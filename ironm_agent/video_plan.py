"""视频出方案（spec [2026-10-23] §201）：视频文字 → 原博主方案（照原话、标秒数）→ 按这个人改的方案。

判断交给模型，按页面规程「视频出方案」（page_skills/video-plan/SKILL.md）；确定的事由代码做：
- 原版核对：每项的原话要在那一秒附近的转写里找得到，组数、次数、重量、休息、每周几次的数字也要在附近的原文里有
  （推算的写明怎么算），找不到就退回重写；离谱的数（30 组、900 秒休息）标出来请他看一眼。
- 动作名对到动作库（protools.movement.find），对不上的留原叫法、存的时候加进"我的动作"。
- 改的版本过现有的计划检查（器械、伤处、特殊人群、天数时长），过不了退回重改；不出时写明原因。
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ironm_agent.exercises import EQUIPMENT, Equipment
from ironm_agent.llm import LLM, parse_json
from ironm_agent.protools.movement import find
from ironm_agent.skills import page_skill
from ironm_agent.training import Plan, PlanItem, Session

SKILL_ID = "video-plan"
WINDOW = 25.0  # 原话、数字在那一秒前后多少秒里找
EXTRACT_TRIES = 3  # 原版最多交几次（第一次 + 退回两次）
ADAPT_TRIES = 4
NEXT_PAD = 10.0  # 数字可以在这个动作开头到下一个动作开头（再多 10 秒）之间说
LAST_SPAN = 120.0  # 最后一个动作往后找多久
DEFAULT_DAYS = (1, 3, 5)  # 博主没说每周几次：按每周 3 次排（周一三五）
SPREAD = {1: (1,), 2: (1, 4), 3: (1, 3, 5), 4: (1, 2, 4, 5), 5: (1, 2, 3, 5, 6), 6: (1, 2, 3, 4, 5, 6), 7: (1, 2, 3, 4, 5, 6, 7)}
DEFAULT_SETS = 3
NOTE_MAX = 80
_PUNCT = re.compile(r"[\s，。？！、,.?!；;：:“”\"'（）()《》「」…—\-]")
_CN = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_NUM = re.compile(r"[零〇一二两三四五六七八九十百]+")
_AR_NUM = re.compile(r"\d+(?:\.\d+)?")


# ---- 数字和原文 ---------------------------------------------------------------------------------


def _cn_value(s: str) -> int | None:
    """中文数字（到几百）→ 整数；"十二"→12，"八十"→80，"一百二十"→120，"两"→2。"""
    if not s:
        return None
    total, cur = 0, 0
    for ch in s:
        if ch in _CN:
            cur = _CN[ch]
        elif ch == "十":
            total += (cur or 1) * 10
            cur = 0
        elif ch == "百":
            total += (cur or 1) * 100
            cur = 0
    return total + cur


def numbers_in(text: str) -> set[float]:
    """一段话里出现的数：阿拉伯数字和中文数字都算（"12次""十二次""两轮"）。"""
    out = {float(x) for x in _AR_NUM.findall(text or "")}
    for m in _CN_NUM.findall(text or ""):
        v = _cn_value(m)
        if v is not None:
            out.add(float(v))
    return out


def _plain(s: str) -> str:
    return _PUNCT.sub("", s or "")


def window(sentences: Sequence[Mapping[str, Any]], t: float, span: float = WINDOW) -> str:
    """那一秒前后的原文（接起来，便于找原话和数字）。"""
    return "".join(str(x.get("text", "")) for x in sentences if abs(float(x.get("t", 0)) - t) <= span)


def transcript_lines(sentences: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(f"[{float(x['t']):.0f}] {x['text']}" for x in sentences)


# ---- 原版：格式和核对 ----------------------------------------------------------------------------


@dataclass
class Checked:
    problems: list[str] = field(default_factory=list)  # 退回重写的
    flags: list[str] = field(default_factory=list)  # 请他看一眼的（离谱的数、推算的）


def _num_ok(value: Any, text: str) -> bool:
    nums = numbers_in(str(value))
    return not nums or nums <= numbers_in(text)


def check_original(data: Mapping[str, Any], sentences: Sequence[Mapping[str, Any]]) -> Checked:
    out = Checked()
    end = max((float(x.get("t", 0)) for x in sentences), default=0.0) + 30
    kind = data.get("kind")
    if kind not in ("plan", "not_plan"):
        out.problems.append('kind 只能是 "plan" 或 "not_plan"')
        return out
    plans = data.get("plans") or []
    if kind == "not_plan":
        if not str(data.get("about") or "").strip():
            out.problems.append("不是训练计划时 about 要写一两句它讲了什么")
        return out
    if not isinstance(plans, list) or not plans:
        out.problems.append("kind 是 plan 时 plans 至少一份")
        return out
    derived_sets: dict[str, list[str]] = {}  # 同一个算法推出来的组数合成一句（"一轮加两轮，共 3 轮"：五个动作说一次）
    for pi, p in enumerate(plans, start=1):
        tag = f"第 {pi} 份" if len(plans) > 1 else ""
        starts = sorted(float(it["t"]) for s in p.get("sessions") or [] for it in s.get("items") or []
                        if isinstance(it.get("t"), int | float))  # fmt: skip
        dq = p.get("days_quote") or None
        if p.get("days_per_week") is not None:
            if not isinstance(dq, Mapping) or not _num_ok(p["days_per_week"], window(sentences, float(dq.get("t", -999)))):
                out.problems.append(f"{tag}每周 {p['days_per_week']} 次在原文里找不到；博主没说就写 null")
        sessions = p.get("sessions") or []
        if not sessions:
            out.problems.append(f"{tag}没有训练内容")
        for s in sessions:
            names = [str(it.get("exercise") or "") for it in s.get("items") or []]
            twice = [n for n in dict.fromkeys(names) if n and names.count(n) > 1]
            if twice:
                out.problems.append(f"「{'、'.join(twice)}」在同一天写了两项：先报动作、过一会儿说做几次的，合成一项（t 写开始讲这个动作的那一秒）")
            for it in s.get("items") or []:
                name = str(it.get("exercise") or "").strip() or "?"
                try:
                    t = float(it.get("t"))
                except (TypeError, ValueError):
                    out.problems.append(f"「{name}」没写出自第几秒")
                    continue
                if not 0 <= t <= end:
                    out.problems.append(f"「{name}」的秒数 {t:.0f} 超出视频长度")
                    continue
                near = window(sentences, t)
                nxt = next((x for x in starts if x > t), t + LAST_SPAN)  # 跟练常是先报动作、过一会儿才说做几次
                said = "".join(str(x.get("text", "")) for x in sentences if t - WINDOW <= float(x.get("t", 0)) <= nxt + NEXT_PAD)
                quote = _plain(str(it.get("quote") or ""))
                if not quote or quote not in _plain(near):
                    out.problems.append(f"「{name}」的原话在第 {t:.0f} 秒附近找不到，原话要照抄转写里的字")
                derived = bool(str(it.get("how") or "").strip())
                for key in ("reps", "load", "rest") + (() if derived else ("sets",)):
                    v = it.get(key)
                    if v not in (None, "") and not _num_ok(v, said):
                        out.problems.append(f"「{name}」的 {key}「{v}」在讲这个动作的那段原文里没有；没说就写 null")
                if derived and it.get("sets") is not None:
                    derived_sets.setdefault(str(it["how"]).strip(), []).append(name)
                sets = it.get("sets")
                if isinstance(sets, int | float) and not 1 <= sets <= 10:
                    out.flags.append(f"「{name}」{sets} 组，不太常见，看一眼视频")
                reps_n = max(numbers_in(str(it.get("reps") or "")), default=0)
                if reps_n > 100:
                    out.flags.append(f"「{name}」{it.get('reps')}，不太常见，看一眼视频")
                rest = str(it.get("rest") or "")
                if rest and max(numbers_in(rest), default=0) > (10 if "分" in rest else 600):
                    out.flags.append(f"「{name}」休息 {rest}，不太常见，看一眼视频")
    for how, names in derived_sets.items():
        who = "、".join(f"「{n}」" for n in dict.fromkeys(names))
        out.flags.insert(0, f"{who}的组数是推算的：{how}")
    return out


# ---- 原版 → 能存的计划 -----------------------------------------------------------------------------


def _mmss(t: float) -> str:
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def _clip(s: str, n: int) -> str:
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def library_name(spoken: str) -> str | None:
    m = find(spoken)
    return m.name if m is not None else None


def to_plan(p: Mapping[str, Any]) -> tuple[Plan, list[str], list[dict[str, Any]]]:
    """原版的一份 → (能存进"我的方案"的计划, 存的时候补了什么, 库里没有的动作)。没说的按常规补并写进备注。"""
    filled: list[str] = []
    unknown: list[dict[str, Any]] = []
    raw_sessions = [s for s in p.get("sessions") or [] if s.get("items")]
    n = p.get("days_per_week")
    days = SPREAD.get(int(n), DEFAULT_DAYS) if isinstance(n, int | float) and 1 <= n <= 7 else DEFAULT_DAYS
    if not isinstance(n, int | float):
        filled.append("没说每周几次，排在周一、三、五")
    elif n < len(raw_sessions):
        days = SPREAD.get(len(raw_sessions), DEFAULT_DAYS)
    sessions: list[Session] = []
    for i, day in enumerate(days[: max(len(days), len(raw_sessions))]):
        s = raw_sessions[i % len(raw_sessions)]
        items: list[PlanItem] = []
        for it in (s.get("items") or [])[:10]:
            spoken = _clip(str(it.get("exercise") or "动作"), 20)  # 库里没有的存成"我的动作"，名字最长 20 字
            lib = library_name(spoken) if it.get("in_library", True) else None
            name = lib or spoken
            if lib is None and spoken not in {u["name"] for u in unknown}:
                unknown.append({"name": spoken, "part": it.get("part") or "", "equipment": it.get("equipment") or "",
                                "mode": it.get("mode") or "strength"})  # fmt: skip
            sets = it.get("sets")
            notes = []
            if not isinstance(sets, int | float) or not 1 <= sets <= 10:
                sets = DEFAULT_SETS
                notes.append("博主没说组数，按常规填的")
            reps = str(it.get("reps") or "").strip()
            if not reps:
                reps = "30 秒" if it.get("mode") in ("timed", "cardio") else "8–12"
                notes.append("博主没说次数，按常规填的")
            if it.get("how"):
                notes.append(str(it["how"]))
            src = f"{_mmss(float(it.get('t') or 0))}「{it.get('quote') or ''}」"
            note = "；".join([*notes, src])
            items.append(PlanItem(exercise=name, sets=int(sets), reps=_clip(reps, 20),
                                  load=_clip(str(it.get("load") or ""), 40), note=_clip(note, NOTE_MAX)))  # fmt: skip
        if items:
            sessions.append(Session(day=day, name=_clip(str(s.get("name") or "训练"), 16), items=items))
    if not raw_sessions or not sessions:
        raise ValueError("原版里没有能排的训练")
    plan = Plan(title=_clip(str(p.get("title") or "博主方案"), 24), goal=_clip(str(p.get("goal") or ""), 16),
                days_per_week=len(sessions), sessions=sessions)  # fmt: skip
    return plan, filled, unknown


# ---- 这个人 -----------------------------------------------------------------------------------


@dataclass
class Who:
    """改方案时看的这个人：给模型看的档案，和代码检查用的限制（与对话里改计划同一份）。"""

    text: str
    equipment: frozenset[Equipment] | None = None
    injured: tuple[str, ...] = ()
    special: frozenset[str] = frozenset()
    days: int | None = None
    minutes: int | None = None
    novice: bool = False
    goal: str | None = None
    strength: str = ""  # 力量记录（每个动作最好的一组），换算 %1RM 用

    def limits_text(self) -> str:
        parts = []
        if self.equipment is not None:
            parts.append("他有的器械：" + "、".join(e for e in EQUIPMENT if e in self.equipment))
        if self.injured:
            parts.append("要避开的部位：" + "、".join(self.injured))
        if self.days:
            parts.append(f"每周最多练 {self.days} 天")
        if self.minutes:
            parts.append(f"每次最多 {self.minutes} 分钟")
        if self.novice:
            parts.append("新手")
        return "；".join(parts)


# ---- 调模型 -----------------------------------------------------------------------------------


def _system(sections: Sequence[str]) -> str:
    sk = page_skill(SKILL_ID)
    body = dict(sk.sections())
    picked = "\n\n".join(f"## {name}\n{body[name]}" for name in sections if name in body)
    return (
        f"你在按规程「{sk.name}」工作。只输出一个 JSON 对象，不要别的文字。\n\n## 步骤卡\n{sk.card}\n\n{picked}"
    )


async def _ask(llm: LLM, messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    reply = await llm.chat("expert", messages, json_mode=True)
    messages.append({"role": "assistant", "content": reply.content})
    return parse_json(reply.content)


async def extract(llm: LLM, sentences: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], Checked, int]:
    """出原版：交上来先核对，没过的把问题列给它重交（最多 EXTRACT_TRIES 次）。返回（原版, 核对结果, 交了几次）。"""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system(["一、是不是训练计划", "二、出原版", "参考：交的格式"])},
        {"role": "user", "content": "这是视频转成的文字，每行前面是这句话开始的秒数。按规程交原版（第一个格式）。\n\n"
                                    + transcript_lines(sentences)},
    ]  # fmt: skip
    got: dict[str, Any] = {}
    checked = Checked(problems=["没交上来"])
    for tries in range(1, EXTRACT_TRIES + 1):
        data = await _ask(llm, messages)
        if data is None:
            checked = Checked(problems=["交的不是 JSON"])
        else:
            got, checked = data, check_original(data, sentences)
        if not checked.problems:
            return got, checked, tries
        messages.append({"role": "user", "content": "核对没过，改了重交整份（还是只要 JSON）：\n- " + "\n- ".join(checked.problems[:12])})
    return got, checked, EXTRACT_TRIES


def _adapted_plan(data: Mapping[str, Any], who: Who | None = None) -> tuple[Plan | None, list[str]]:
    """模型交的改版 → 计划。每次多长由代码定：有他的上限就用上限（检查按估的时长比它），没有就按估的时长。"""
    from ironm_agent.protools.plan_check import estimate_minutes

    raw = data.get("plan")
    if not isinstance(raw, Mapping):
        return None, ["没有 plan"]
    problems: list[str] = []
    sessions = []
    for s in raw.get("sessions") or []:
        items = []
        for it in (s.get("items") or [])[:10]:
            name = str(it.get("exercise") or "")
            lib = library_name(name)
            if lib is None:
                problems.append(f"「{name}」不在动作库里，换成库里的动作")
                continue
            try:
                items.append(PlanItem(exercise=lib, sets=int(it.get("sets") or 0), reps=str(it.get("reps") or "8–12"),
                                      load=_clip(str(it.get("load") or ""), 40), note=_clip(str(it.get("note") or ""), NOTE_MAX)))  # fmt: skip
            except Exception as e:
                problems.append(f"「{name}」写得不对：{type(e).__name__}（组数 1–10）")
        days = s.get("days") if isinstance(s.get("days"), list) else [s.get("day")]
        for d in days if items else []:
            try:
                sess = Session(day=int(d or 0), name=_clip(str(s.get("name") or "训练"), 16),
                               items=[x.model_copy() for x in items])  # fmt: skip
                need = estimate_minutes(sess)
                sess.minutes = int(min(180, max(10, who.minutes if who and who.minutes else round(need))))
                sessions.append(sess)
            except Exception:
                problems.append(f"「{s.get('name')}」这一天写得不对（day 1–7、minutes 10–180）")
    if not sessions:
        return None, problems or ["没有训练内容"]
    try:
        plan = Plan(title=_clip(str(raw.get("title") or "按你调整"), 24), goal=_clip(str(raw.get("goal") or ""), 16),
                    days_per_week=len(sessions), sessions=sessions)  # fmt: skip
    except Exception as e:
        return None, [*problems, f"计划写得不对：{e}"[:120]]
    return plan, problems


def usable_moves(who: Who) -> str:
    """他的器械做得了的动作库名字（按部位），给改方案时挑。"""
    from ironm_agent.exercises import library

    by: dict[str, list[str]] = {}
    for e in library().exercises:
        if who.equipment is None or any(g in who.equipment for g in e.equipment):  # 和计划检查一样：列的器械有一样就能做
            by.setdefault(e.part, []).append(e.name)
    return "\n".join(f"{k}：{'、'.join(v)}" for k, v in by.items())


def check_adapted(plan: Plan, who: Who) -> list[str]:
    from ironm_agent.protools.plan_check import check_plan

    r = check_plan(plan, goal=who.goal, novice=who.novice, injured=who.injured, equipment=who.equipment,
                   available_days=who.days, available_minutes=who.minutes, special=who.special)  # fmt: skip
    return list(r.problems)


async def adapt(
    llm: LLM, original: Mapping[str, Any], who: Who
) -> tuple[dict[str, Any], Plan | None, list[str], int]:
    """按这个人改：交上来过计划检查，没过退回重改。返回（模型交的, 计划或 None, 最后还剩的问题, 交了几次）。"""
    asked = (
        f"他的档案：\n{who.text}\n\n代码检查会按这些限制查：{who.limits_text() or '没有特别的'}"
        + (f"\n\n他的力量记录：\n{who.strength}" if who.strength else "")
        + "\n\n动作只能从这些动作库名字里挑（他的器械做得了的）：\n" + usable_moves(who)
        + "\n\n代码估每次时长：热身 10 分钟，加上每组约 3.5 分钟（多关节动作）、2.5 分钟（单关节、核心）、"
          "1.5 分钟（备注写了「循环」的），有氧按写的分钟数；不能超过他每次能练的时长。"
        + "\n\n博主原版（照原话整理的）：\n" + json.dumps(original, ensure_ascii=False)
        + "\n\n按规程「三、改成他的」交改的版本（第二个格式）。"
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system(["三、改成他的", "参考：交的格式"])},
        {"role": "user", "content": asked},
    ]
    data: dict[str, Any] = {}
    problems: list[str] = ["没交上来"]
    for tries in range(1, ADAPT_TRIES + 1):
        got = await _ask(llm, messages)
        if got is None:
            problems = ["交的不是 JSON"]
        else:
            data = got
            if str(data.get("blocked") or "").strip():
                return data, None, [], tries
            plan, problems = _adapted_plan(data, who)
            if plan is not None and not problems:
                problems = check_adapted(plan, who) + kept_constraints(plan, original)
                if problems and all("分钟" in x for x in problems):  # 只差时长：代码按时长减组数（纯算术），记成改动
                    plan, trims = fit_time(plan, who)
                    problems = check_adapted(plan, who)
                    if not problems:
                        data = {**data, "changes": [*(data.get("changes") or []), *trims]}
                changes = data.get("changes")
                for c in changes if isinstance(changes, list) else []:
                    if isinstance(c, dict) and not c.get("why"):
                        c["why"] = c.get("reason") or c.get("理由") or ""
                if not isinstance(changes, list) or not changes or not all(
                    isinstance(c, Mapping) and str(c.get("why") or "").strip() for c in changes
                ):
                    problems.append("changes 要列出每处改动（至少一条），每条写 why；没说的组数次数按常规补的也算一条")
                if not problems:
                    return data, plan, [], tries
        tip = "\n- ".join(problems[:12])
        if plan is not None and any("分钟" in p for p in problems):
            tip += "\n代码估的时长（分钟）：\n" + time_breakdown(plan)
        messages.append({"role": "user", "content": "检查没过，改了重交整份（还是只要 JSON）：\n- " + tip})
    return data, None, problems, ADAPT_TRIES


def fit_time(plan: Plan, who: Who) -> tuple[Plan, list[dict[str, Any]]]:
    """每次超了他的时长：从组数最多的那项起每次减一组（最少 1 组），直到估的时长放得下。返回（新计划, 记下的改动）。"""
    from ironm_agent.protools.plan_check import SLACK, estimate_minutes

    if not who.minutes:
        return plan, []
    p = plan.model_copy(deep=True)
    trims: list[dict[str, Any]] = []
    for s in p.sessions:
        before = {x.exercise: x.sets for x in s.items}
        while estimate_minutes(s) > who.minutes * SLACK:
            item = max(s.items, key=lambda x: x.sets)
            if item.sets <= 1:
                break
            item.sets -= 1
        for x in s.items:
            if x.sets != before[x.exercise]:
                trims.append({"day": s.day, "exercise": x.exercise, "from": f"{before[x.exercise]} 组", "to": f"{x.sets} 组",
                              "why": f"你每次只有 {who.minutes} 分钟，按时长减了组数"})  # fmt: skip
    return p, trims


def kept_constraints(plan: Plan, original: Mapping[str, Any]) -> list[str]:
    """博主特意说的做法（现在查"不跳"）：改的版本换动作时不能丢。"""
    said = json.dumps(original, ensure_ascii=False)
    if "不跳" not in said:
        return []
    jumps = list(dict.fromkeys(x.exercise for s in plan.sessions for x in s.items
                               if (m := find(x.exercise)) is not None and m.tag.pattern == "plyo"))  # fmt: skip
    return [f"博主说了不跳：{'、'.join(jumps)}是跳跃类，换成不跳的动作（比如自重深蹲）"] if jumps else []


def time_breakdown(plan: Plan) -> str:
    """每天每项估多少分钟（退回时给它看，好知道该减哪里）。"""
    from ironm_agent.protools.plan_check import WARMUP_MIN, _minutes, estimate_minutes

    rows = []
    for s in plan.sessions:
        parts = "、".join(f"{x.exercise} {x.sets} 组 {_minutes(x, find(x.exercise)):.1f}" for x in s.items)
        rows.append(f"{s.name}：热身 {WARMUP_MIN} + {parts} = {estimate_minutes(s):.0f}")
    return "\n".join(rows)


async def generate(
    llm: LLM, sentences: Sequence[Mapping[str, Any]], who: Who, progress: Callable[[str], None] | None = None
) -> dict[str, Any]:
    """整条：原版 → 选一份改 → 两份都能存的计划。出不来的写明原因，不抛（模型调不通时照常抛给调用方）。
    progress：走到哪一步（"original" 整理原版、"adapt" 按他改）。"""
    started = time.monotonic()
    if progress:
        progress("original")
    original, checked, n1 = await extract(llm, sentences)
    out: dict[str, Any] = {"kind": original.get("kind"), "about": original.get("about") or "", "tries": [n1]}
    if checked.problems:
        out.update(status="failed", reason="原版没核对过，这次没整理出来，可以再生成一次", problems=checked.problems[:6])
        return out
    if original.get("kind") == "not_plan":
        out.update(status="not_plan", reason="这个视频讲的不是训练计划")
        return out
    plans = []
    for p in original.get("plans") or []:
        try:
            plan, filled, unknown = to_plan(p)
        except Exception as e:
            out.update(status="failed", reason="原版排不成计划，可以再生成一次", problems=[str(e)[:120]])
            return out
        plans.append({"raw": p, "plan": plan.model_dump(mode="json"), "filled": filled, "unknown": unknown})
    out.update(originals=plans, flags=checked.flags, unclear=original.get("unclear") or [])
    pick = plans[0]["raw"] if len(plans) == 1 else {"两套": [x["raw"] for x in plans]}
    if progress:
        progress("adapt")
    data, plan, left, n2 = await adapt(llm, pick, who)
    out["tries"].append(n2)
    if plan is not None:
        out["adapted"] = {"plan": plan.model_dump(mode="json"), "changes": data.get("changes") or [],
                          "why_this": data.get("why_this") or ""}  # fmt: skip
    elif str(data.get("blocked") or "").strip():
        out["blocked"] = str(data["blocked"])
    else:
        out["adapt_failed"] = left[:6]
    out.update(status="ok", ms=round((time.monotonic() - started) * 1000))
    return out
