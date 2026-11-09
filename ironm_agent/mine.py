"""自定义（spec [2026-11-09] §198）：我的食物、我的动作、我的方案，只属于本人。

和自建食谱存在同一张加密小表里（recipes.DietStore，按种类分）：食物 "food"、动作 "move"、方案 "myplan"。
不碰公共的食物库、动作库、方案库；读出来对不上格式的跳过，不报错。
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ironm_agent.exercises import EQUIPMENT, Equipment, Part, library
from ironm_agent.meals import Macros
from ironm_agent.training import Plan

FOOD, MOVE, PLAN = "food", "move", "myplan"
LIMITS = {FOOD: 50, MOVE: 50, PLAN: 20}
FOOD_PREFIX = "my-"  # 我的食物的编号带这个前缀，和食物库的编号分开
KCAL_OFF = 0.2  # 热量和三大营养素算出来的差两成以上：提示再看看包装
type Mode = Literal["strength", "timed", "cardio"]  # 力量记重量×次数；计时记秒；有氧记时长
MODES: tuple[Mode, ...] = ("strength", "timed", "cardio")


class LimitError(ValueError):
    """超过上限、重名、动作不认识：给人看的话。"""


def _name(v: Any) -> Any:
    return " ".join(v.split()) if isinstance(v, str) else v


class Per100In(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kcal: float = Field(ge=0, le=900)
    protein_g: float = Field(ge=0, le=100)
    carbs_g: float = Field(ge=0, le=100)
    fat_g: float = Field(ge=0, le=100)


class MyFood(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = ""
    name: str = Field(min_length=1, max_length=20)
    unit: str = Field(default="份", min_length=1, max_length=6)  # 一份叫什么：个、勺、杯、包
    unit_g: float = Field(default=100, ge=1, le=2000)  # 一份多少克
    per100: Per100In

    _n = field_validator("name", "unit", mode="before")(_name)

    def macros(self) -> Macros:
        return Macros(**self.per100.model_dump())

    def check(self) -> str:
        """热量和三大营养素对得上吗：对不上时给一句提示（照样能存）。"""
        p = self.per100
        calc = p.protein_g * 4 + p.carbs_g * 4 + p.fat_g * 9
        if max(calc, p.kcal) < 20:  # 很少的（茶、黑咖啡）不比
            return ""
        if abs(p.kcal - calc) > KCAL_OFF * max(calc, p.kcal):
            return f"数对不上：按三大营养素算每 100 克约 {round(calc)} 千卡，你填的是 {round(p.kcal)} 千卡，再看看包装"
        return ""

    def shown(self) -> dict[str, Any]:
        """和食物库的食物一个样子（页面、选着记一餐都能直接用），多一个 mine。"""
        p = self.per100
        return {
            "id": self.id, "name": self.name, "cat": "我的", "dish": False, "unit": self.unit,
            "unit_g": self.unit_g, "oil_g": 0, "image": None, "mine": True,
            "per100": {"kcal": round(p.kcal), "protein_g": round(p.protein_g, 1),
                       "carbs_g": round(p.carbs_g, 1), "fat_g": round(p.fat_g, 1)},
            "warning": self.check(),
        }  # fmt: skip


class MyMove(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = ""
    name: str = Field(min_length=1, max_length=20)
    part: Part
    equipment: list[Equipment] = Field(min_length=1, max_length=len(EQUIPMENT))
    mode: Mode = "strength"
    note: str = Field(default="", max_length=200)

    _n = field_validator("name", mode="before")(_name)

    @field_validator("equipment")
    @classmethod
    def _dedupe(cls, v: list[Equipment]) -> list[Equipment]:
        return list(dict.fromkeys(v))

    def shown(self) -> dict[str, Any]:
        return {**self.model_dump(), "mine": True}


class MyPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = ""
    plan: Plan

    def shown(self) -> dict[str, Any]:
        return {"id": self.id, "mine": True, **self.plan.model_dump(mode="json")}


def new_id(prefix: str = "") -> str:
    return prefix + uuid.uuid4().hex[:12]


async def load(store: Any, user_id: str, kind: str) -> list[Any]:
    """读出来对不上格式的跳过（以后改了格式也不报错）。"""
    model = {FOOD: MyFood, MOVE: MyMove, PLAN: MyPlan}[kind]
    out = []
    for d in await store.all(user_id, kind):
        try:
            out.append(model.model_validate(d))
        except ValidationError:
            continue
    return out


def library_names() -> set[str]:
    return {x.name for x in library().exercises}


def check_move(move: MyMove, others: list[MyMove]) -> None:
    if move.name in library_names():
        raise LimitError(f"动作库里已经有「{move.name}」了，直接用库里的就好")
    if any(x.name == move.name and x.id != move.id for x in others):
        raise LimitError(f"你已经加过「{move.name}」了")


def check_food(food: MyFood, others: list[MyFood]) -> None:
    if any(x.name == food.name and x.id != food.id for x in others):
        raise LimitError(f"你已经加过「{food.name}」了")


def check_plan(plan: Plan, moves: list[MyMove]) -> None:
    """方案里的动作要在动作库或我的动作里。"""
    known = library_names() | {m.name for m in moves}
    bad = list(
        dict.fromkeys(
            it.exercise for s in plan.sessions for it in s.items if it.exercise not in known
        )
    )
    if bad:
        raise LimitError(f"动作库和你的动作里都没有：{'、'.join(bad[:3])}")


def check_room(have: list[Any], kind: str, item_id: str) -> None:
    if not item_id and len(have) >= LIMITS[kind]:
        what = {FOOD: "食物", MOVE: "动作", PLAN: "方案"}[kind]
        raise LimitError(f"最多存 {LIMITS[kind]} 个{what}，先删掉几个不用的")


def modes_of(moves: list[MyMove]) -> dict[str, Mode]:
    """训练记录器用：我的动作怎么记。"""
    return {m.name: m.mode for m in moves}
