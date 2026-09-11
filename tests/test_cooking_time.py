import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from api import app as api


class CookingTimeTests(unittest.TestCase):
    def test_legacy_step_time_is_not_exposed_as_total(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipes.json"
            path.write_text(json.dumps([{
                "name": "braised pork", "file_path": "missing.md", "cooking_time": 1
            }]), encoding="utf-8")
            with patch.object(api, "DATA_FILE", path):
                service = api.GraphRAGWebService()
                card = service.get_recommendations()[0]
        self.assertIsNone(card["cooking_time"])
        self.assertEqual(card["cooking_time_label"], "耗时待确认")

    def test_pork_step_time_and_stirring_interval_are_not_summed(self):
        service = api.GraphRAGWebService()
        recipe = service.recipe_by_name["湖南家常红烧肉"]
        source = (api.PROJECT_ROOT / recipe["file_path"]).read_text(encoding="utf-8")
        self.assertIn("翻炒 1 分钟", source)
        self.assertIn("间隔 10 分钟", source)
        card = service._build_card(recipe)
        self.assertIsNone(card["cooking_time"])
        self.assertEqual(card["cooking_time_label"], "耗时待确认")

    def test_reviewed_totals_preserve_units_and_source(self):
        service = api.GraphRAGWebService()
        self.assertEqual(service.recipe_by_name["农家一碗香"]["cooking_time"], 17)
        self.assertEqual(service.recipe_by_name["无骨鸡爪"]["cooking_time"], 495)
        self.assertEqual(service.recipe_by_name["黄油煎虾"]["cooking_time_label"], "总耗时 1 小时内")
        self.assertIsNone(service.recipe_by_name["黄油煎虾"]["cooking_time"])
        for recipe in service.recipes:
            if recipe["cooking_time_source"]:
                source = (api.PROJECT_ROOT / recipe["file_path"]).read_text(encoding="utf-8")
                self.assertIn(recipe["cooking_time_source"], source)
            else:
                self.assertIsNone(recipe["cooking_time"])


if __name__ == "__main__":
    unittest.main()
