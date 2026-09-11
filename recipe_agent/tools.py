"""菜谱解析 Agent 可调用的确定性工具。"""

import re
from typing import Any, Dict, List

from pydantic import ValidationError

from .models import Quantity, RecipeRecord


SECTION_NAMES = {
    "必备原料和工具": "ingredients",
    "计算": "quantities",
    "操作": "steps",
    "附加内容": "additional",
}

INGREDIENT_ALIASES = {
    "番茄": "西红柿",
    "蕃茄": "西红柿",
    "马铃薯": "土豆",
    "生粉": "淀粉",
    "细砂糖": "白砂糖",
    "味极鲜": "生抽",
}

QUALITATIVE_WORDS = (
    "适量", "少许", "依个人口味", "依照个人口味", "按个人口味", "根据个人口味",
    "看个人口味", "酌量", "稍微多一点", "少量", "若干",
)
UNIT_PATTERN = r"kg|KG|Kg|千克|公斤|斤|两|g|克|L|l|升|ml|mL|毫升|个|只|颗|棵|根|片|瓣|勺|匙|茶匙|汤匙|杯"


def normalize_ingredient_name(name: str) -> str:
    cleaned = re.sub(r"[（(].*?[）)]", "", name).strip(" -*\t")
    return INGREDIENT_ALIASES.get(cleaned, cleaned)


def parse_markdown_sections(markdown: str) -> Dict[str, Any]:
    """按二级标题拆分文档；保留原始行，供模型引用证据。"""
    title_match = re.search(r"(?m)^#\s+(.+?)\s*$", markdown)
    title = title_match.group(1).strip() if title_match else ""
    title = re.sub(r"的做法$", "", title)
    difficulty_match = re.search(r"预估烹饪难度\s*[：:]\s*([★☆]+)", markdown)
    difficulty = difficulty_match.group(1).count("★") if difficulty_match else None

    sections: Dict[str, List[str]] = {value: [] for value in SECTION_NAMES.values()}
    other: List[str] = []
    current: List[str] = other
    for line in markdown.splitlines():
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading:
            key = SECTION_NAMES.get(heading.group(1).strip())
            current = sections[key] if key else other
            continue
        if line.strip():
            current.append(line.rstrip())

    return {
        "title": title,
        "difficulty": difficulty,
        "ingredients": sections["ingredients"],
        "quantities": sections["quantities"],
        "steps": sections["steps"],
        "additional": sections["additional"],
    }


def _canonicalize(value: float, unit: str) -> tuple[float, str]:
    mapping = {
        "kg": (1000.0, "g"), "KG": (1000.0, "g"), "Kg": (1000.0, "g"),
        "千克": (1000.0, "g"), "公斤": (1000.0, "g"),
        "斤": (500.0, "g"), "两": (50.0, "g"),
        "g": (1.0, "g"), "克": (1.0, "g"),
        "L": (1000.0, "ml"), "l": (1000.0, "ml"), "升": (1000.0, "ml"),
        "ml": (1.0, "ml"), "mL": (1.0, "ml"), "毫升": (1.0, "ml"),
    }
    factor, canonical_unit = mapping.get(unit, (1.0, unit))
    return value * factor, canonical_unit


def normalize_quantity(raw: str) -> Quantity:
    """规范化用量，无法可靠换算的表达保留原文。"""
    text = raw.strip()
    if not text:
        return Quantity(raw="未知", kind="unknown", note="原文未给出用量")

    half_match = re.search(r"半\s*(斤|两|升|个|只|杯)", text)
    if half_match:
        unit = half_match.group(1)
        canonical_value, canonical_unit = _canonicalize(0.5, unit)
        return Quantity(
            raw=text, kind="exact", value=0.5, unit=unit,
            canonical_value=canonical_value, canonical_unit=canonical_unit,
        )

    formula_match = re.search(
        rf"(\d+(?:\.\d+)?)\s*({UNIT_PATTERN})?[^\n*×xX]{{0,30}}(?:\*|×|x|X)\s*(份数|人数)", text
    )
    if formula_match:
        value = float(formula_match.group(1))
        unit = formula_match.group(2) or ""
        canonical_value, canonical_unit = _canonicalize(value, unit)
        return Quantity(
            raw=text, kind="formula", value=value, unit=unit,
            canonical_value=canonical_value, canonical_unit=canonical_unit,
            formula=formula_match.group(0), note="按份数计算，未展开最终数值",
        )

    if any(word in text for word in QUALITATIVE_WORDS):
        return Quantity(raw=text, kind="qualitative", note="定性用量，未推断数值")

    range_match = re.search(
        rf"(\d+(?:\.\d+)?)\s*(?:-|~|～|至|到)\s*(\d+(?:\.\d+)?)\s*({UNIT_PATTERN})?", text
    )
    if range_match:
        lower = float(range_match.group(1))
        upper = float(range_match.group(2))
        unit = range_match.group(3) or ""
        canonical_lower, canonical_unit = _canonicalize(lower, unit)
        canonical_upper, _ = _canonicalize(upper, unit)
        return Quantity(
            raw=text, kind="range", min_value=lower, max_value=upper, unit=unit,
            canonical_min_value=canonical_lower, canonical_max_value=canonical_upper,
            canonical_unit=canonical_unit,
        )

    exact_match = re.search(rf"(\d+(?:\.\d+)?)\s*({UNIT_PATTERN})", text)
    if exact_match:
        value = float(exact_match.group(1))
        unit = exact_match.group(2)
        canonical_value, canonical_unit = _canonicalize(value, unit)
        return Quantity(
            raw=text, kind="exact", value=value, unit=unit,
            canonical_value=canonical_value, canonical_unit=canonical_unit,
        )

    return Quantity(raw=text, kind="unknown", note="无法可靠解析，保留原文")


def validate_recipe_payload(payload: Dict[str, Any], markdown: str, source_path: str) -> tuple[RecipeRecord | None, List[str]]:
    errors: List[str] = []
    candidate = dict(payload)
    candidate["source_path"] = source_path
    try:
        recipe = RecipeRecord.model_validate(candidate)
    except ValidationError as exc:
        return None, [error["msg"] for error in exc.errors()]

    for ingredient in recipe.ingredients:
        if ingredient.evidence not in markdown:
            errors.append(f"食材 {ingredient.name} 的 evidence 不在源文档中")
            continue
        evidence = ingredient.evidence
        if any(word in evidence for word in QUALITATIVE_WORDS) and ingredient.quantity.kind != "qualitative":
            errors.append(
                f"食材 {ingredient.name} 的证据包含模糊用量，必须把包含限定词的原文交给 normalize_quantity"
            )
        if re.search(r"(?:\*|×|x|X)\s*(?:份数|人数)", evidence) and ingredient.quantity.kind != "formula":
            errors.append(f"食材 {ingredient.name} 的证据包含份数公式，quantity.kind 必须为 formula")
    for step in recipe.steps:
        if step.evidence not in markdown:
            errors.append(f"步骤 {step.step_number} 的 evidence 不在源文档中")

    if errors:
        return None, errors
    return recipe, []
