"""菜谱解析 Agent 的结构化数据模型。"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


QuantityKind = Literal["exact", "range", "qualitative", "formula", "unknown"]
RecipeCategory = Literal["素菜", "荤菜", "水产", "早餐", "主食", "汤类", "甜品", "饮料", "调料", "半成品", "其他"]


class Quantity(BaseModel):
    raw: str = Field(min_length=1, description="原文中的用量表达")
    kind: QuantityKind = "unknown"
    value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    unit: str = ""
    canonical_value: Optional[float] = None
    canonical_min_value: Optional[float] = None
    canonical_max_value: Optional[float] = None
    canonical_unit: str = ""
    formula: str = ""
    note: str = ""

    @model_validator(mode="after")
    def validate_shape(self) -> "Quantity":
        if self.kind == "exact" and self.value is None:
            raise ValueError("exact 用量必须提供 value")
        if self.kind == "range":
            if self.min_value is None or self.max_value is None:
                raise ValueError("range 用量必须提供 min_value 和 max_value")
            if self.min_value > self.max_value:
                raise ValueError("用量区间下限不能大于上限")
        if self.kind == "formula" and not self.formula:
            raise ValueError("formula 用量必须保留公式")
        return self


class IngredientRecord(BaseModel):
    name: str = Field(min_length=1)
    normalized_name: str = Field(min_length=1)
    category: str = "其他"
    is_main: bool = True
    quantity: Quantity
    evidence: str = Field(min_length=1, description="必须逐字来自源 Markdown")


class CookingStepRecord(BaseModel):
    step_number: int = Field(ge=1)
    description: str = Field(min_length=1)
    methods: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    time_estimate: str = ""
    evidence: str = Field(min_length=1, description="必须逐字来自源 Markdown")


class RecipeRecord(BaseModel):
    name: str = Field(min_length=1)
    categories: List[RecipeCategory] = Field(min_length=1)
    difficulty: Optional[int] = Field(default=None, ge=1, le=5)
    cuisine_type: str = ""
    prep_time: str = ""
    cook_time: str = ""
    servings: str = ""
    ingredients: List[IngredientRecord] = Field(min_length=1)
    steps: List[CookingStepRecord] = Field(min_length=1)
    tags: List[str] = Field(default_factory=list)
    source_path: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_steps(self) -> "RecipeRecord":
        numbers = [step.step_number for step in self.steps]
        if len(numbers) != len(set(numbers)):
            raise ValueError("步骤编号不能重复")
        if numbers != sorted(numbers):
            raise ValueError("步骤必须按 step_number 升序排列")
        return self


class AgentTrace(BaseModel):
    source_path: str
    round: int = Field(ge=0)
    event: str
    tool_name: str = ""
    arguments_summary: str = ""
    result_summary: str = ""
    validation_errors: List[str] = Field(default_factory=list)
    latency_ms: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
