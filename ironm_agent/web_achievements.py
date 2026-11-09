"""成就勋章的接口（spec [2026-11-09] §196）：从训练、体重和围度、目标、计划、饮食、打卡照里取数，交给 achievements 算。

每样数据读不到就当作没有，不影响别的勋章；日志只记类别，不记内容。
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

from fastapi import APIRouter, Depends

from ironm_agent.achievements import Inputs, compute, summary
from ironm_agent.web import _verify
from ironm_agent.web_more import GOAL, MEALS, PHOTOS, READINGS, TRAINING, _web

_log = logging.getLogger("ironm_agent.achievements")
router: APIRouter = APIRouter()
SINCE = date(2000, 1, 1)
EXTRA_KINDS = ("hip", "chest", "arm", "thigh")  # 快照里没有的四项围度：从读数记录里取


async def _try(what: str, coro: Any, default: Any) -> Any:
    try:
        return await coro
    except Exception as e:
        _log.warning("achievements %s unavailable: error_type=%s", what, type(e).__name__)
        return default


def _day(text: Any) -> date | None:
    try:
        return date.fromisoformat(str(text)[:10])
    except ValueError:
        return None


async def gather(user_id: str, training: Any, meals: Any, photos: Any, goals: Any, readings: Any,
                 today: date | None = None, snapshot: Any = None) -> Inputs:  # fmt: skip
    """snapshot：调用方已经读好的这一刻的数据（管家作答时），不再读一遍。"""
    today = today or date.today()
    snap = snapshot if snapshot is not None else await _try("snapshot", _web().snapshot_for(user_id), None)
    logs = await _try("training", training.workouts(user_id, SINCE), [])
    plans = await _try("plans", training.plans(user_id), [])
    goal = await _try("goal", goals.get(user_id), None)
    body: dict[str, list[tuple[date, float]]] = {}
    if snap is not None:
        r = snap.readings
        body = {"weight": list(r.weight), "body_fat": list(r.body_fat), "waist": list(r.waist)}
    for kind in EXTRA_KINDS + (() if body.get("waist") else ("waist",)):
        rows = await _try(kind, readings.recent(user_id, kind, limit=400), [])
        body[kind] = [(d, float(x["value"])) for x in rows if (d := _day(x.get("day"))) is not None]
    protein: dict[date, float] = defaultdict(float)
    for m in await _try("meals", meals.since(user_id, today - timedelta(days=365)), []):
        protein[m.day] += m.meal.macros.protein_g
    target = next((float(m.value) for m in snap.metrics if m.key == "protein_target" and m.value), None) if snap else None
    with_photo = await _try("photos", photos.having(user_id, [x.id for x in logs]), set())
    return Inputs(
        today=today,
        goal=str(snap.profile.goal or "") if snap is not None else None,
        target_kg=goal.value if goal is not None else None,
        workouts=[x.log for x in logs],
        plans=[(p.created.date(), p.plan.days_per_week) for p in plans],
        body=body,
        protein_days=sorted(protein.items()),
        protein_target=target,
        photo_days=[(x.log.day, x.id) for x in logs if x.id in with_photo],
    )


@router.get("/iron/api/achievements")
async def get_achievements(
    user_id: str = Depends(_verify()),
    training: Any = TRAINING,
    meals: Any = MEALS,
    photos: Any = PHOTOS,
    goals: Any = GOAL,
    readings: Any = READINGS,
) -> dict[str, Any]:
    inp = await gather(user_id, training, meals, photos, goals, readings)
    out = summary(compute(inp), inp.today)
    out["direction"] = "体重、围度往下算进步" if inp.goal in (None, "", "fat_loss", "health", "endurance") else "体重、臂围胸围往上算进步"
    return out
