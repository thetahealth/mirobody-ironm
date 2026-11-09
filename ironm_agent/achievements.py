"""成就勋章（spec [2026-11-09] §196）：只记"最"和里程碑，按已有的记录算出来，不读写存储。

四组：身体之最（按目标看方向、按一周平均）、坚持（按周不按天）、力量之最、第一次。
只奖励做到的：没拿到的灰着、写怎么拿或还差多少，不提醒、不倒计时。离谱的数不算。
"""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Literal

from ironm_agent.training import WorkoutLog

type Tier = Literal["bronze", "silver", "gold", "best"] | None  # best：之最类，拿到就是这枚，没有档次
GROUPS = (("body", "身体之最"), ("streak", "坚持"), ("power", "力量之最"), ("first", "第一次"))
LOWER_GOALS = (None, "", "fat_loss", "health", "endurance")  # 体重、围度往下是进步
NEW_DAYS = 7  # 几天内拿到的标"新"
MAX_KG = 500
MAX_REPS = 100
_NUM = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True)
class Badge:
    id: str
    group: str
    title: str
    icon: str
    tier: Tier = None
    value: str = ""  # 成绩，如 78 cm、80 kg × 5
    day: date | None = None  # 哪天拿到（最近一次刷新）
    detail: str = ""  # 比开始好了多少
    hint: str = ""  # 没拿到：怎么拿、还差多少；拿到了：下一档
    progress: float | None = None  # 到下一档的进度 0–1
    how: str = ""  # 怎么算的
    series: tuple[tuple[str, float], ...] = ()  # 这一路的曲线（日期、值）
    link: str = ""  # 跳到哪看完整数据
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def earned(self) -> bool:
        return self.tier is not None

    def json(self, today: date) -> dict[str, Any]:
        return {
            "id": self.id, "group": self.group, "title": self.title, "icon": self.icon, "tier": self.tier,
            "earned": self.earned, "value": self.value, "day": self.day.isoformat() if self.day else None,
            "new": bool(self.day and self.earned and (today - self.day).days < NEW_DAYS), "detail": self.detail,
            "hint": self.hint, "progress": None if self.progress is None else round(min(1.0, max(0.0, self.progress)), 2),
            "how": self.how, "series": [list(x) for x in self.series], "link": self.link, **self.extra,
        }  # fmt: skip


@dataclass(frozen=True)
class Inputs:
    today: date
    goal: str | None = None  # 档案里的目标：fat_loss / muscle / strength / endurance / health
    target_kg: float | None = None  # 目标体重
    workouts: Sequence[WorkoutLog] = ()
    plans: Sequence[tuple[date, int]] = ()  # （从哪天起，每周练几天）
    body: Mapping[str, Sequence[tuple[date, float]]] = field(default_factory=dict)  # weight、body_fat、waist、hip…
    protein_days: Sequence[tuple[date, float]] = ()  # 每天记下的蛋白质（克）
    protein_target: float | None = None
    photo_days: Sequence[tuple[date, str]] = ()  # 带打卡照的训练：（哪天，训练编号）


# ---- 小工具 ----------------------------------------------------------------------------------------


def _monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def weekly(points: Sequence[tuple[date, float]]) -> list[tuple[date, float, date]]:
    """按周平均：（这周一，平均值，这周最后一次记录的日期）。"""
    by: dict[date, list[tuple[date, float]]] = defaultdict(list)
    for d, v in points:
        by[_monday(d)].append((d, float(v)))
    return [(m, statistics.fmean(v for _, v in xs), max(d for d, _ in xs)) for m, xs in sorted(by.items())]


def _tier(value: float, steps: Sequence[float]) -> Tier:
    names: tuple[Tier, ...] = ("bronze", "silver", "gold")
    got = [n for n, s in zip(names, steps, strict=True) if value >= s - 1e-9]
    return got[-1] if got else None


def _next(value: float, steps: Sequence[float]) -> float | None:
    return next((s for s in steps if value < s - 1e-9), None)


def _g(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


def _md(d: date) -> str:
    return f"{d.month} 月 {d.day} 日"


def _lower(goal: str | None) -> bool:
    return goal in LOWER_GOALS


# ---- 身体之最 --------------------------------------------------------------------------------------


def _body(inp: Inputs) -> list[Badge]:
    lower = _lower(inp.goal)
    specs = [  # （编号、名字、图标、读数、单位、往下好吗、档次、只在哪个方向出现）
        ("weight-low", "最轻一周", "scale", "weight", " kg", True, (2, 5, 10), "lower"),
        ("weight-high", "最重一周", "scale", "weight", " kg", False, (1, 3, 6), "upper"),
        ("waist", "最小腰围", "tape", "waist", " cm", True, (3, 6, 10), "lower"),
        ("body-fat", "最低体脂", "drop", "body_fat", "%", True, (2, 4, 6), "both"),
        ("arm", "最大上臂围", "tape", "arm", " cm", False, (1, 2, 4), "upper"),
        ("chest", "最大胸围", "tape", "chest", " cm", False, (1, 2, 4), "upper"),
        ("hip", "最小臀围", "tape", "hip", " cm", True, (2, 4, 6), "lower"),
        ("thigh", "最小大腿围", "tape", "thigh", " cm", True, (2, 4, 6), "lower"),
    ]
    out = []
    for bid, title, icon, kind, unit, down, steps, side in specs:
        if (side == "lower" and not lower) or (side == "upper" and lower):
            continue
        weeks = weekly(inp.body.get(kind, ()))
        how = "按一周的平均算，单天的起伏不算" + ("；体脂秤个人误差很大，看长期趋势就好" if kind == "body_fat" else "")
        link = "#record"  # 身体数据在记录页
        word = "少" if down else "多"
        u = " 个百分点" if kind == "body_fat" else unit
        if len(weeks) < 2:
            hint = "还没记过" if not weeks else "再记一周就能和开始比"
            out.append(Badge(f"body:{bid}", "body", title, icon, hint=f"{hint}；比开始{word} {_g(steps[0])}{u} 拿铜", how=how, link=link))
            continue
        start = weeks[0][1]
        best = (min if down else max)(weeks, key=lambda w: w[1])
        gain = (start - best[1]) if down else (best[1] - start)
        tier = _tier(gain, steps)
        nxt = _next(gain, steps)
        series = tuple((w[0].isoformat(), round(w[1], 1)) for w in weeks)
        detail = f"比开始{word} {_g(gain)}{u}" if gain > 0 else "还没比开始好"
        hint = f"再{word} {_g(nxt - gain)}{u} 升一档" if tier and nxt else (f"比开始{word} {_g(nxt)}{u} 拿铜，还差 {_g(nxt - max(gain, 0))}{u}" if nxt else "已经最高一档")
        out.append(Badge(f"body:{bid}", "body", title, icon, tier, f"{_g(best[1])}{unit}", best[2] if tier else None, detail, hint,
                         None if not nxt else max(gain, 0) / nxt, how, series, link))  # fmt: skip
    target = _target(inp, lower)
    if target is not None:
        out.insert(min(2, len(out)), target)
    return out


def _target(inp: Inputs, lower: bool) -> Badge | None:
    if not inp.target_kg:
        return None
    weeks = weekly(inp.body.get("weight", ()))
    how = "一周的平均体重到了你定的目标体重"
    if not weeks:
        return Badge("body:target", "body", "到达目标体重", "target", hint=f"目标 {_g(inp.target_kg)} kg", how=how, link="#me")
    start, tgt = weeks[0][1], inp.target_kg
    down = tgt < start
    hit = next((w for w in weeks if (w[1] <= tgt if down else w[1] >= tgt)), None)
    latest = weeks[-1][1]
    span = abs(start - tgt) or 1
    done = (start - latest) if down else (latest - start)
    if hit:
        return Badge("body:target", "body", "到达目标体重", "target", "gold", f"{_g(tgt)} kg", hit[2],
                     f"从 {_g(start)} kg 到 {_g(hit[1])} kg", how=how, series=tuple((w[0].isoformat(), round(w[1], 1)) for w in weeks), link="#me")  # fmt: skip
    return Badge("body:target", "body", "到达目标体重", "target", hint=f"目标 {_g(tgt)} kg，还差 {_g(abs(latest - tgt))} kg",
                 progress=max(0.0, done) / span, how=how, link="#me")  # fmt: skip


# ---- 坚持 ------------------------------------------------------------------------------------------


def _done(inp: Inputs) -> list[WorkoutLog]:
    return sorted((w for w in inp.workouts if w.done and w.rest is None and w.day <= inp.today), key=lambda w: w.day)


def _met_weeks(inp: Inputs, done: Sequence[WorkoutLog]) -> list[date]:
    """练满计划的那几周（周一）：这周练的天数不少于当时计划的每周天数。"""
    days_by: dict[date, set[date]] = defaultdict(set)
    for w in done:
        days_by[_monday(w.day)].add(w.day)
    plans = sorted(inp.plans)
    out = []
    for m, days in sorted(days_by.items()):
        end = m + timedelta(days=6)
        plan = next((n for start, n in reversed(plans) if start <= end), None)
        if plan and len(days) >= plan:
            out.append(m)
    return out


def _streak(inp: Inputs) -> list[Badge]:
    done = _done(inp)
    out = []
    n = len(done)
    steps = (10, 50, 100)
    tier = _tier(n, steps)
    after = next((s for s in (200, 365) if n < s), None) if n >= 100 else None
    nxt = _next(n, steps) or after
    reach = done[(steps[("bronze", "silver", "gold").index(tier)] - 1)].day if tier else None
    out.append(Badge("streak:count", "streak", f"累计 {n} 次" if tier else "累计训练", "flame", tier, f"{n} 次", reach,
                     f"一共练了 {n} 次", f"还差 {nxt - n} 次到 {nxt} 次" if nxt else "", None if not nxt else n / nxt,
                     "练完打卡或记了训练就算一次；没练、记了休息的不算", link="#record/timeline"))  # fmt: skip
    hours = sum((w.minutes or 0) for w in done) / 60
    htier = _tier(hours, (10, 50, 100))
    hnxt = _next(hours, (10, 50, 100))
    out.append(Badge("streak:hours", "streak", "累计时长", "clock", htier, f"{_g(round(hours, 1))} 小时",
                     done[-1].day if htier else None, f"一共练了 {_g(round(hours, 1))} 小时",
                     f"还差 {_g(round(hnxt - hours, 1))} 小时到 {hnxt} 小时" if hnxt else "", None if not hnxt else hours / hnxt,
                     "按每次记下的训练时长加起来；没记时长的不算"))  # fmt: skip
    weeks = sorted({_monday(w.day) for w in done})
    best_run, run, end = 0, 0, None
    for i, m in enumerate(weeks):
        run = run + 1 if i and (m - weeks[i - 1]).days == 7 else 1
        if run > best_run:
            best_run, end = run, m
    wsteps = (4, 8, 16)
    wtier = _tier(best_run, wsteps)
    wnxt = _next(best_run, wsteps) or (next((s for s in (26, 52) if best_run < s), None) if best_run >= 16 else None)
    out.append(Badge("streak:weeks", "streak", f"连续 {best_run} 周" if wtier else "连续每周都练", "calendar", wtier,
                     f"{best_run} 周", (end + timedelta(days=6)) if (wtier and end) else None, "最长连续多少周，每周都练了",
                     f"连续 {wnxt} 周升一档" if wnxt else "", None if not wnxt else best_run / wnxt,
                     "按周算，每周练了一次以上就算；按天算容易断，断了也不扣"))  # fmt: skip
    months = Counter((w.day.year, w.day.month) for w in done)
    if months:
        (y, mo), c = max(months.items(), key=lambda kv: (kv[1], kv[0]))
        out.append(Badge("streak:month", "streak", "最多的一个月", "flame", "best", f"{c} 次", date(y, mo, 1),
                         f"{y} 年 {mo} 月练了 {c} 次", how="一个月里练得最多的那次；被刷新就更新"))  # fmt: skip
    else:
        out.append(Badge("streak:month", "streak", "最多的一个月", "flame", hint="练一次就开始算"))
    met = _met_weeks(inp, done)
    msteps = (4, 12, 26)
    mtier = _tier(len(met), msteps)
    mnxt = _next(len(met), msteps)
    out.append(Badge("streak:plan", "streak", "说到做到", "check", mtier, f"{len(met)} 周", (met[-1] + timedelta(days=6)) if mtier else None,
                     f"有 {len(met)} 周练满了计划", f"还差 {mnxt - len(met)} 周到 {mnxt} 周" if mnxt else "",
                     None if not mnxt else len(met) / mnxt, "一周练的天数不少于当时计划的每周天数"))  # fmt: skip
    back = next((b for a, b in zip(reversed(done[:-1]), reversed(done[1:]), strict=True) if (b.day - a.day).days > 14), None)
    out.append(Badge("streak:back", "streak", "回来了", "return", "best" if back else None, _md(back.day) if back else "",
                     back.day if back else None, "停了一阵又练起来了" if back else "",
                     "" if back else "停一阵再回来练也算，不着急", how="两次训练隔了两周以上，又练了就算"))  # fmt: skip
    return out


# ---- 力量之最 --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Set:
    day: date
    weight: float | None
    reps: int
    seconds: int | None


def _first_num(text: str) -> float | None:
    m = _NUM.search(text or "")
    return float(m.group()) if m else None


def _sets(w: WorkoutLog) -> dict[str, tuple[str, list[_Set], Any]]:
    """这次训练每个动作的各组：动作库认得的归到同一个动作（叫法不同也算一个）。"""
    from ironm_agent.protools.movement import find

    out: dict[str, tuple[str, list[_Set], Any]] = {}
    for it in w.items:
        move = find(it.exercise)
        key = move.exercise.id if move else it.exercise
        name = move.name if move else it.exercise
        rows: list[_Set] = []
        if it.done_sets:
            rows = [_Set(w.day, s.weight_kg, s.reps, s.seconds) for s in it.done_sets]
        else:
            reps = it.reps or ""
            n = _first_num(reps)
            secs = None
            if n is not None and "秒" in reps:
                secs, n = int(n), 0
            elif n is not None and "分" in reps:
                secs, n = int(n * 60), 0
            rows = [_Set(w.day, it.weight_kg, int(n or 0), secs)] * (it.sets or 1)
        rows = [r for r in rows if (r.weight is None or 0 <= r.weight <= MAX_KG) and 0 <= r.reps <= MAX_REPS]
        if rows:
            prev = out.get(key)
            out[key] = (name, (prev[1] if prev else []) + rows, move)
    return out


_NOT_LIFTS = re.compile(r"跑|骑|走|游泳|跳绳|椭圆|划船机|单车|有氧|拉伸|放松|泡沫轴")


def _lifts(name: str, move: Any) -> bool:
    """算不算举重量的动作：动作库认得的看动作模式，认不得的看名字。"""
    if move is not None:
        return move.tag.pattern not in ("cardio", "stretch")
    return not _NOT_LIFTS.search(name)


_RATIO = {  # 和体重比：动作编号 → （名字、档次）
    "squat": ("深蹲是体重的几倍", ("back-squat", "front-squat"), (1.0, 1.5, 2.0)),
    "deadlift": ("硬拉是体重的几倍", ("deadlift", "trap-bar-deadlift"), (1.0, 1.5, 2.0)),
    "bench": ("卧推是体重的几倍", ("bench-press",), (0.75, 1.0, 1.25)),
}
_REPS = {  # 自重一组最多几个
    "pullup": ("引体向上最多", ("pull-up", "chin-up"), (1, 5, 10)),
    "pushup": ("俯卧撑最多", ("push-up",), (10, 30, 50)),
    "dip": ("双杠臂屈伸最多", ("parallel-dip",), (5, 10, 20)),
}


def _power(inp: Inputs, done: Sequence[WorkoutLog]) -> tuple[list[Badge], date | None, date | None]:
    """返回（勋章，第一次破纪录那天，第一个标准引体那天）。"""
    per: dict[str, tuple[str, list[_Set], Any]] = {}
    for w in done:
        for key, (name, rows, move) in _sets(w).items():
            prev = per.get(key)
            per[key] = (name, (prev[1] if prev else []) + rows, move)
    out: list[Badge] = []
    first_pr: date | None = None
    for key, (name, rows, move) in per.items():
        if not _lifts(name, move):  # 有氧、拉伸记了"重量"也不算（骑车 7.5 kg × 2 那种，§196 复核）
            continue
        weighted = [r for r in rows if r.weight and r.reps > 0]
        if not weighted:
            continue
        by_day: dict[date, _Set] = {}
        for r in weighted:
            cur = by_day.get(r.day)
            if cur is None or (r.weight, r.reps) > (cur.weight, cur.reps):
                by_day[r.day] = r
        days = sorted(by_day)
        best = by_day[days[0]]
        for d in days[1:]:
            r = by_day[d]
            if (r.weight, r.reps) > (best.weight, best.reps):  # type: ignore[operator]
                first_pr = d if first_pr is None or d < first_pr else first_pr
                best = r
        first = by_day[days[0]]
        detail = (f"第一次 {_g(first.weight or 0)} kg × {first.reps}，现在 {_g(best.weight or 0)} kg × {best.reps}"
                  if (best.weight, best.reps) != (first.weight, first.reps) else f"练了 {len(days)} 次")  # fmt: skip
        out.append(Badge(f"power:best:{key}", "power", f"{name}之最", "barbell", "best", f"{_g(best.weight or 0)} kg × {best.reps}",
                         best.day, detail, how="记下的最重的一组；一样重的，次数多的算",
                         series=tuple((d.isoformat(), by_day[d].weight or 0) for d in days),
                         extra={"exercise": name}))  # fmt: skip
    out.sort(key=lambda b: b.day or date.min, reverse=True)
    bw_weeks = weekly(inp.body.get("weight", ()))
    bw = bw_weeks[-1][1] if bw_weeks else None
    special: list[Badge] = []
    for rid, (title, ids, steps) in _RATIO.items():
        rows = [r for k in ids if k in per for r in per[k][1] if r.weight and r.reps > 0]
        how = "最重一组的重量 ÷ 最近一周的平均体重"
        if not rows or not bw:
            special.append(Badge(f"power:ratio:{rid}", "power", title, "barbell",
                                 hint="要有体重记录和这个动作的重量记录" if rows or not bw else "记一次这个动作的重量就开始算", how=how))  # fmt: skip
            continue
        top = max(rows, key=lambda r: (r.weight, r.reps))
        ratio = (top.weight or 0) / bw
        tier = _tier(ratio, steps)
        nxt = _next(ratio, steps)
        special.append(Badge(f"power:ratio:{rid}", "power", title, "barbell", tier, f"{ratio:.2f} 倍", top.day if tier else None,
                             f"{_g(top.weight or 0)} kg，体重 {_g(round(bw, 1))} kg",
                             f"到 {nxt:g} 倍（约 {_g(round(nxt * bw / 2.5) * 2.5)} kg）升一档" if nxt else "已经最高一档",
                             None if not nxt else ratio / nxt, how))  # fmt: skip
    first_pullup: date | None = None
    for rid, (title, ids, steps) in _REPS.items():
        rows = [r for k in ids if k in per for r in per[k][1] if r.reps > 0 and not r.weight]
        if rid == "pullup" and rows:
            first_pullup = min(r.day for r in rows)
        if not rows:
            special.append(Badge(f"power:reps:{rid}", "power", title, "body", hint=f"一组做到 {steps[0]} 个拿铜", how="不负重的一组最多做几个"))
            continue
        top = max(rows, key=lambda r: (r.reps, -r.day.toordinal()))
        tier = _tier(top.reps, steps)
        nxt = _next(top.reps, steps)
        special.append(Badge(f"power:reps:{rid}", "power", title, "body", tier, f"{top.reps} 个", top.day if tier else None,
                             f"一组最多 {top.reps} 个", f"还差 {nxt - top.reps} 个到 {nxt} 个" if nxt else "已经最高一档",
                             None if not nxt else top.reps / nxt, "不负重的一组最多做几个"))  # fmt: skip
    plank = [r for k in ("plank",) if k in per for r in per[k][1] if r.seconds]
    psteps = (60, 120, 180)
    if plank:
        top = max(plank, key=lambda r: r.seconds or 0)
        s = top.seconds or 0
        tier, nxt = _tier(s, psteps), _next(s, psteps)
        special.append(Badge("power:plank", "power", "撑得最久", "clock", tier, f"{s // 60} 分 {s % 60} 秒" if s >= 60 else f"{s} 秒",
                             top.day if tier else None, "平板支撑一次最长", f"还差 {nxt - s} 秒到 {nxt // 60} 分钟" if nxt else "已经最高一档",
                             None if not nxt else s / nxt, "平板支撑计时的一组"))  # fmt: skip
    else:
        special.append(Badge("power:plank", "power", "撑得最久", "clock", hint="平板支撑撑到 1 分钟拿铜", how="平板支撑计时的一组"))
    cardio: list[tuple[date, float]] = []
    for w in done:
        for _, rows, move in _sets(w).values():
            if move is not None and move.tag.pattern == "cardio":
                mins = sum((r.seconds or 0) for r in rows) / 60 or (w.minutes or 0)
                if mins:
                    cardio.append((w.day, mins))
    csteps = (20, 40, 60)
    if cardio:
        d, m = max(cardio, key=lambda x: x[1])
        tier, nxt = _tier(m, csteps), _next(m, csteps)
        special.append(Badge("power:cardio", "power", "有氧最长", "run", tier, f"{round(m)} 分钟", d if tier else None,
                             "一次有氧最长", f"还差 {round(nxt - m)} 分钟到 {nxt} 分钟" if nxt else "已经最高一档",
                             None if not nxt else m / nxt, "跑步、骑车、椭圆机这类有氧，一次练了多久"))  # fmt: skip
    else:
        special.append(Badge("power:cardio", "power", "有氧最长", "run", hint="一次有氧练到 20 分钟拿铜", how="一次有氧练了多久"))
    return special + out, first_pr, first_pullup


# ---- 第一次 ----------------------------------------------------------------------------------------


def _first(inp: Inputs, done: Sequence[WorkoutLog], first_pr: date | None, first_pullup: date | None) -> list[Badge]:
    def one(bid: str, title: str, icon: str, day: date | None, hint: str, value: str = "", extra: dict[str, Any] | None = None) -> Badge:
        return Badge(f"first:{bid}", "first", title, icon, "best" if day else None, value or (_md(day) if day else ""), day,
                     hint="" if day else hint, extra=extra or {})  # fmt: skip

    met = _met_weeks(inp, done)
    photos = sorted(inp.photo_days)
    span = photos and (photos[-1][0] - photos[0][0]).days >= 90
    protein = next((d for d, g in sorted(inp.protein_days) if inp.protein_target and g >= inp.protein_target), None)
    return [
        one("workout", "第一次训练", "star", done[0].day if done else None, "练完打卡就拿到"),
        one("pullup", "第一个标准引体", "body", first_pullup, "不借力做完一个引体向上"),
        one("pr", "第一次破纪录", "up", first_pr, "任一个动作比自己之前做得更重或更多"),
        one("plan", "照计划练满一周", "check", (met[0] + timedelta(days=6)) if met else None, "一周练够计划的次数"),
        one("photos", "前后对比", "photo", photos[-1][0] if span else None, "打卡照第一张和最近一张隔三个月以上",
            extra={"photos": [photos[0][1], photos[-1][1]]} if span else None),
        one("protein", "第一次吃够蛋白质", "egg", protein, "一天记下的蛋白质到目标"),
    ]


# ---- 汇总 ------------------------------------------------------------------------------------------


def compute(inp: Inputs) -> list[Badge]:
    done = _done(inp)
    power, first_pr, first_pullup = _power(inp, done)
    return [*_body(inp), *_streak(inp), *power, *_first(inp, done, first_pr, first_pullup)]


def summary(badges: Sequence[Badge], today: date) -> dict[str, Any]:
    earned = [b for b in badges if b.earned]
    recent = sorted((b for b in earned if b.day), key=lambda b: b.day or date.min, reverse=True)
    return {
        "earned": len(earned), "total": len(badges),
        "recent": [b.json(today) for b in recent[:4]],
        "groups": [{"key": k, "title": t, "badges": [b.json(today) for b in badges if b.group == k]} for k, t in GROUPS],
    }  # fmt: skip


def recent_line(badges: Sequence[Badge], today: date) -> str:
    """档案里一行：最近 7 天新拿到的勋章（§196 需求 5）；没有时为空。"""
    new = [b for b in badges if b.earned and b.day and (today - b.day).days < NEW_DAYS]
    new.sort(key=lambda b: b.day or date.min, reverse=True)
    return "；".join(f"{b.title} {b.value}（{_md(b.day)}）" for b in new[:5] if b.day)
