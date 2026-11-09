"""视频出方案第二步的接口（spec [2026-10-23] §201）：按一段视频文字生成原博主方案和按本人改的方案，本人选存不存。

- 生成用本人自己填的模型（§167，没填回 428）；在后台跑，同一个人同时只生成一个，进度放内存。
- 结果和文字同一个编号存在 "videoplan"；选完（存了任意一份或"都不要了"）就删掉文字和结果（§200 需求 5）。
- 存进"我的方案"：原版标"原版·"、按你改的标"按你·"；原版里库里没有的动作加进"我的动作"；
  每项的出处（第几秒、原话）和每处改动的理由写在动作备注里，以后在对话里问为什么，IronM 照这些答。
- 两份都不会自动换成当前计划，只有本人在"我的方案"里点"用这套"才换（§198，能撤销）。
- 日志只记状态、耗时、交了几次，不记文字和方案内容。
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, date, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from ironm_agent import video_plan
from ironm_agent.web_video import KIND, PLAN_KIND, STORE, _alive
from ironm_agent.web_voice import _verify

_log = logging.getLogger("ironm_agent.video_plan")
router: APIRouter = APIRouter()
_jobs: dict[str, dict[str, Any]] = {}  # 用户 → 正在生成的（或刚失败、还没看到的）
_tasks: set[asyncio.Task[None]] = set()
STEP = {"original": "正在整理博主原版…", "adapt": "正在按你的情况改…"}


def plan_job_view(user_id: str) -> dict[str, Any] | None:
    job = _jobs.get(user_id)
    if job is None:
        return None
    if job.get("state") == "failed":
        _jobs.pop(user_id, None)  # 失败的说一次就清掉
    return {k: job[k] for k in ("id", "state", "step", "error") if k in job}


def training_dep() -> Any:
    from ironm_agent.web import training_store

    return training_store()


def memory_dep() -> Any:
    from ironm_agent.web import memory_store

    return memory_store()


def pain_dep() -> Any:
    from ironm_agent.web_more import pain_store

    return pain_store()


TRAINING = Depends(training_dep)
MEMORY = Depends(memory_dep)
PAIN = Depends(pain_dep)


async def who_for(user_id: str, memory: Any, training: Any, pain: Any) -> video_plan.Who:
    """这个人：档案（和对话里给 Agent 看的同一份）、器械伤处特殊情况（和计划页改计划同一份）、每周几天每次多久、力量记录。"""
    from ironm_agent.memory_view import Recall
    from ironm_agent.web_plan import _limits

    equipment, injured, special = await _limits(user_id, memory, pain)
    try:
        items = Recall.of(await memory.items(user_id)).items
    except Exception as e:
        _log.warning("memory unavailable: error_type=%s", type(e).__name__)
        items = ()
    days = minutes = None
    novice = False
    for it in items:
        data = getattr(it, "data", None) or {}
        if it.category == "schedule":
            days = data.get("per_week") or days
            minutes = data.get("minutes") or minutes
        if it.category == "level" and any(w in it.content for w in ("零基础", "新手", "刚开始", "没练过", "不到")):
            novice = True
    text = ""
    try:
        from ironm_agent.casefile import build
        from ironm_agent.web import snapshot_for

        text = build(await snapshot_for(user_id), Recall.of(list(items)), "").text()
    except Exception as e:
        _log.warning("casefile unavailable: error_type=%s", type(e).__name__)
        text = "\n".join(f"- {it.content}" for it in items)
    return video_plan.Who(text=text or "（还没建档）", equipment=equipment, injured=tuple(injured), special=frozenset(special),
                          days=days, minutes=minutes, novice=novice, strength=await _strength(user_id, training))  # fmt: skip


async def _strength(user_id: str, training: Any, k: int = 8) -> str:
    """最近 90 天每个动作最好的一组（按估的最大重量排），给 %1RM 换算用。"""
    from ironm_agent.protools.strength import epley

    try:
        logs = await training.workouts(user_id, date.today() - timedelta(days=90))
    except Exception:
        return ""
    best: dict[str, tuple[float, float, int]] = {}
    for w in logs:
        for it in w.items:
            for st in it.done_sets or []:
                if st.weight_kg and st.reps and 1 <= st.reps <= 20:
                    e = epley(st.weight_kg, st.reps)
                    if e > best.get(it.exercise, (0, 0, 0))[0]:
                        best[it.exercise] = (e, st.weight_kg, st.reps)
    rows = sorted(best.items(), key=lambda kv: -kv[1][0])[:k]
    return "\n".join(f"- {n}：{kg:g} kg × {r} 次（估最大约 {e:.0f} kg）" for n, (e, kg, r) in rows)


async def _run(user_id: str, job: dict[str, Any], llm: Any, sentences: list[Any], who: video_plan.Who, store: Any) -> None:
    def step(name: str) -> None:
        job["step"] = name

    try:
        out = await video_plan.generate(llm, sentences, who, step)
        if out.get("status") == "failed":
            job.update(state="failed", error=out.get("reason") or "这次没整理出来，可以再生成一次")
            _log.info("video plan failed: tries=%s", out.get("tries"))
            return
        out["created"] = datetime.now(UTC).isoformat(timespec="seconds")
        await store.put(user_id, PLAN_KIND, job["id"], {"id": job["id"], **out})
        _jobs.pop(user_id, None)
        _log.info("video plan done: status=%s tries=%s ms=%s adapted=%s",
                  out.get("status"), out.get("tries"), out.get("ms"), "adapted" in out)  # fmt: skip
    except Exception as e:
        job.update(state="failed", error="这次没能调用你的模型，稍后再试")
        _log.warning("video plan error: error_type=%s", type(e).__name__)


@router.post("/iron/api/video-text/{item_id}/plan")
async def start_plan(
    item_id: str,
    user_id: str = Depends(_verify()),
    store: Any = STORE,
    memory: Any = MEMORY,
    training: Any = TRAINING,
    pain: Any = PAIN,
) -> dict[str, Any]:
    text = next((x for x in await _alive(store, user_id) if x["id"] == item_id), None)
    if text is None:
        raise HTTPException(404, "没有这条视频文字（可能过了 7 天删掉了）")
    if (job := _jobs.get(user_id)) and job.get("state") != "failed":
        raise HTTPException(429, "上一个还在生成，好了再来")
    from ironm_agent.web_more import need_llm

    llm = await need_llm(user_id)  # 用他自己的模型（§167），没填回 428
    who = await who_for(user_id, memory, training, pain)
    await store.delete(user_id, PLAN_KIND, item_id)  # 重新生成：旧的那份作废
    job = {"id": item_id, "state": "working", "step": "original"}
    _jobs[user_id] = job
    task = asyncio.create_task(_run(user_id, job, llm, list(text.get("sentences") or []), who, store))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return {"plan_job": plan_job_view(user_id)}


@router.get("/iron/api/video-text/{item_id}/plan")
async def get_plan(item_id: str, user_id: str = Depends(_verify()), store: Any = STORE) -> dict[str, Any]:
    if not any(x["id"] == item_id for x in await _alive(store, user_id)):
        raise HTTPException(404, "没有这条视频文字（可能过了 7 天删掉了）")
    got = await store.get(user_id, PLAN_KIND, item_id)
    if not got:
        raise HTTPException(404, "还没生成方案")
    return got  # type: ignore[no-any-return]


class Choice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original: list[int] = []  # 存哪几份原版（讲了两套时可以都存）
    adapted: bool = False


def _marked(title: str, mark: str) -> str:
    return (mark + title)[:24]


@router.post("/iron/api/video-text/{item_id}/plan/save")
async def save_plan(item_id: str, choice: Choice, user_id: str = Depends(_verify()), store: Any = STORE) -> dict[str, Any]:
    """存选中的进"我的方案"（都不选就是"都不要了"），然后删掉这段文字和结果。"""
    from ironm_agent.mine import MOVE, PLAN, LimitError, MyMove, MyPlan, check_move, check_plan, check_room, load, new_id
    from ironm_agent.training import Plan

    got = await store.get(user_id, PLAN_KIND, item_id)
    if not got:
        raise HTTPException(404, "还没生成方案")
    originals = got.get("originals") or []
    if any(not 0 <= i < len(originals) for i in choice.original) or (choice.adapted and not got.get("adapted")):
        raise HTTPException(422, "选的方案不存在")
    plans: list[Plan] = []
    moves: list[MyMove] = []
    mine_moves = await load(store, user_id, MOVE)
    for i in dict.fromkeys(choice.original):
        o = originals[i]
        p = Plan.model_validate(o["plan"])
        p.title = _marked(p.title, "原版·")
        plans.append(p)
        for u in o.get("unknown") or []:
            if any(m.name == u["name"] for m in [*mine_moves, *moves]):
                continue
            part = u.get("part") if u.get("part") in ("腿臀", "胸", "背", "肩", "手臂", "核心", "全身", "有氧", "拉伸放松") else "全身"
            gear = u.get("equipment") if u.get("equipment") in ("自重", "哑铃", "壶铃", "弹力带", "杠铃", "单杠", "器械", "泡沫轴") else "自重"
            mode = u.get("mode") if u.get("mode") in ("strength", "timed", "cardio") else "strength"
            moves.append(MyMove(id=new_id(), name=u["name"], part=part, equipment=[gear], mode=mode,
                                note="从视频方案里加的（博主的叫法，动作库里没有）"))  # fmt: skip
    if choice.adapted:
        a = got["adapted"]
        p = Plan.model_validate(a["plan"])
        why = {}
        for c in a.get("changes") or []:
            if isinstance(c, dict) and c.get("exercise") and c.get("why"):
                why.setdefault(str(c["exercise"]), str(c["why"]))
        for s in p.sessions:
            for it in s.items:
                if it.exercise in why and "改：" not in it.note:
                    it.note = ("；".join(x for x in (it.note, "改：" + why[it.exercise]) if x))[: video_plan.NOTE_MAX]
        p.title = _marked(p.title, "按你·")
        plans.append(p)
    try:
        have = await load(store, user_id, PLAN)
        for _ in plans:
            check_room(have, PLAN, "")
            have.append(None)
        all_moves = list(mine_moves)
        for m in moves:
            check_move(m, all_moves)
            check_room(all_moves, MOVE, "")
            all_moves.append(m)
        for p in plans:
            check_plan(p, all_moves)
    except LimitError as e:
        raise HTTPException(409, str(e)) from None
    for m in moves:
        await store.put(user_id, MOVE, m.id, m.model_dump(mode="json"))
    saved = []
    for p in plans:
        mp = MyPlan(id=new_id(), plan=p)
        await store.put(user_id, PLAN, mp.id, mp.model_dump(mode="json"))
        saved.append({"id": mp.id, "title": p.title})
    await store.delete(user_id, PLAN_KIND, item_id)
    await store.delete(user_id, KIND, item_id)
    _log.info("video plan chosen: plans=%d moves=%d", len(saved), len(moves))
    return {"saved": saved, "moves": [m.name for m in moves]}
