import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from recipe_agent.agent import RecipeParsingAgent
from recipe_agent.artifacts import RecipeArtifactStore, stable_numeric_id
from recipe_agent.models import RecipeRecord
from recipe_agent.tools import normalize_quantity, parse_markdown_sections, validate_recipe_payload


MARKDOWN = """# 测试菜的做法

预估烹饪难度：★★

## 必备原料和工具

- 土豆

## 计算

- 土豆 2～3 个

## 操作

- 土豆切块。
"""


def candidate(evidence: str = "- 土豆 2～3 个"):
    return {
        "name": "测试菜",
        "categories": ["素菜"],
        "difficulty": 2,
        "ingredients": [{
            "name": "土豆",
            "normalized_name": "土豆",
            "category": "蔬菜",
            "is_main": True,
            "quantity": normalize_quantity("2～3 个").model_dump(),
            "evidence": evidence,
        }],
        "steps": [{
            "step_number": 1,
            "description": "土豆切块。",
            "methods": ["切"],
            "tools": ["刀"],
            "evidence": "- 土豆切块。",
        }],
        "source_path": "ignored.md",
    }


def tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments, ensure_ascii=False)),
    )


def response(*calls):
    message = SimpleNamespace(content="", tool_calls=list(calls))
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class QuantityTests(unittest.TestCase):
    def test_exact_and_unit_conversion(self):
        quantity = normalize_quantity("牛肉 1.5 kg")
        self.assertEqual(quantity.kind, "exact")
        self.assertEqual(quantity.canonical_value, 1500)
        self.assertEqual(quantity.canonical_unit, "g")
        self.assertEqual(normalize_quantity("半斤").canonical_value, 250)

    def test_range_qualitative_and_formula(self):
        interval = normalize_quantity("蒜瓣 2～3 个")
        self.assertEqual((interval.min_value, interval.max_value, interval.unit), (2, 3, "个"))
        self.assertEqual(normalize_quantity("盐适量").kind, "qualitative")
        formula = normalize_quantity("咖喱块 15g × 份数")
        self.assertEqual(formula.kind, "formula")
        self.assertEqual(formula.canonical_value, 15)
        self.assertIn("份数", formula.formula)
        annotated_formula = normalize_quantity("肉蟹 1 只（大约 300g） * 份数")
        self.assertEqual(annotated_formula.kind, "formula")
        self.assertEqual((annotated_formula.value, annotated_formula.unit), (1, "只"))
        self.assertEqual(normalize_quantity("干辣椒（依照个人口味）2-3 个").kind, "qualitative")

    def test_sections_and_evidence_validation(self):
        sections = parse_markdown_sections(MARKDOWN)
        self.assertEqual(sections["title"], "测试菜")
        self.assertEqual(sections["difficulty"], 2)
        recipe, errors = validate_recipe_payload(candidate(), MARKDOWN, "sample.md")
        self.assertIsNotNone(recipe)
        self.assertEqual(errors, [])
        recipe, errors = validate_recipe_payload(candidate("不存在的证据"), MARKDOWN, "sample.md")
        self.assertIsNone(recipe)
        self.assertTrue(any("evidence" in error for error in errors))

    def test_quantity_kind_must_match_evidence_context(self):
        payload = candidate()
        payload["ingredients"][0]["evidence"] = "- 土豆依个人口味放 2～3 个"
        contextual_markdown = MARKDOWN + "\n- 土豆依个人口味放 2～3 个\n"
        recipe, errors = validate_recipe_payload(payload, contextual_markdown, "sample.md")
        self.assertIsNone(recipe)
        self.assertTrue(any("模糊用量" in error for error in errors))

    def test_quantity_must_preserve_qualitative_context(self):
        markdown = MARKDOWN.replace("- 土豆 2～3 个", "- 土豆依照个人口味放 2～3 个")
        payload = candidate("- 土豆依照个人口味放 2～3 个")
        recipe, errors = validate_recipe_payload(payload, markdown, "sample.md")
        self.assertIsNone(recipe)
        self.assertTrue(any("模糊用量" in error for error in errors))


class AgentLoopTests(unittest.TestCase):
    def test_agent_requires_tools_and_submit(self):
        data = candidate()
        client = FakeClient([
            response(
                tool_call("1", "parse_markdown_sections", {}),
                tool_call("2", "normalize_quantity", {"raw": "2～3 个"}),
            ),
            response(tool_call("3", "validate_recipe", {"recipe": data})),
            response(tool_call("4", "submit_recipe", {"recipe": data})),
        ])
        result = RecipeParsingAgent(client, max_rounds=4, sleeper=lambda _: None).parse(MARKDOWN, "sample.md")
        self.assertIsNotNone(result.recipe)
        self.assertEqual(result.recipe.source_path, "sample.md")
        self.assertEqual([trace.tool_name for trace in result.traces], [
            "parse_markdown_sections", "normalize_quantity", "validate_recipe", "submit_recipe"
        ])

    def test_agent_retries_api_and_stops_without_submit(self):
        client = FakeClient([
            RuntimeError("temporary"),
            response(tool_call("1", "parse_markdown_sections", {})),
            response(tool_call("2", "unknown_tool", {})),
        ])
        result = RecipeParsingAgent(
            client, max_rounds=2, max_retries=2, sleeper=lambda _: None
        ).parse(MARKDOWN, "sample.md")
        self.assertIsNone(result.recipe)
        self.assertEqual(result.retries, 1)
        self.assertIn("最大工具调用轮数", result.error)
        self.assertTrue(any(trace.validation_errors for trace in result.traces))


class ArtifactTests(unittest.TestCase):
    def test_stable_ids_and_crlf_artifacts(self):
        self.assertEqual(stable_numeric_id("21", "a"), stable_numeric_id("21", "a"))
        recipe = RecipeRecord.model_validate({**candidate(), "source_path": "sample.md"})
        with __import__("tempfile").TemporaryDirectory() as directory:
            output = Path(directory)
            RecipeArtifactStore(output).write([recipe], [], [])
            self.assertIn(b"\r\n", (output / "recipes.jsonl").read_bytes())
            self.assertTrue((output / "nodes.csv").exists())
            self.assertTrue((output / "relationships.csv").exists())


if __name__ == "__main__":
    unittest.main()
