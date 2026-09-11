"""解析产物、Neo4j CSV 与增量写入支持。"""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from neo4j import GraphDatabase

from .models import AgentTrace, RecipeRecord


NODE_HEADERS = [
    "nodeId", "labels", "name", "preferredTerm", "fsn", "conceptType", "synonyms",
    "category", "difficulty", "cuisineType", "prepTime", "cookTime", "servings", "tags",
    "filePath", "amount", "unit", "isMain", "description", "stepNumber", "methods", "tools",
    "timeEstimate",
]
RELATION_HEADERS = [
    "startNodeId", "endNodeId", "relationshipType", "relationshipId", "amount", "unit", "step_order",
]
CATEGORY_IDS = {
    "素菜": "710000000", "荤菜": "720000000", "水产": "730000000", "早餐": "740000000",
    "主食": "750000000", "汤类": "760000000", "甜品": "770000000", "饮料": "780000000",
    "调料": "790000000", "半成品": "795000000",
}


def stable_numeric_id(prefix: str, key: str) -> str:
    """生成跨进程稳定、与旧 201... ID 范围错开的数字 ID。"""
    digest = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:12], 16) % 99_999_999
    return f"{prefix}{digest:08d}"


def stable_relation_id(key: str) -> str:
    return "R_AGENT_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\r\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class RecipeArtifactStore:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        recipes: List[RecipeRecord],
        failures: List[Dict[str, Any]],
        traces: List[AgentTrace],
    ) -> Dict[str, Any]:
        _write_jsonl(self.output_dir / "recipes.jsonl", [recipe.model_dump() for recipe in recipes])
        _write_jsonl(self.output_dir / "failures.jsonl", failures)
        _write_jsonl(self.output_dir / "agent_traces.jsonl", [trace.model_dump() for trace in traces])
        self._write_graph_csv(recipes)

        error_counts = Counter(
            error
            for trace in traces
            for error in trace.validation_errors
        )
        round_latencies: Dict[tuple[str, int], int] = {}
        for trace in traces:
            key = (trace.source_path, trace.round)
            round_latencies[key] = max(round_latencies.get(key, 0), trace.latency_ms)
        summary = {
            "total": len(recipes) + len(failures),
            "succeeded": len(recipes),
            "failed": len(failures),
            "success_rate": round(len(recipes) / max(1, len(recipes) + len(failures)), 4),
            "retry_count": sum(
                int(trace.result_summary)
                for trace in traces
                if trace.event == "api_retry" and trace.result_summary.isdigit()
            ) + sum(
                int(trace.arguments_summary)
                for trace in traces
                if trace.event == "api_error" and trace.arguments_summary.isdigit()
            ),
            "validation_error_distribution": dict(error_counts),
            "total_duration_ms": sum(round_latencies.values()),
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\r\n", encoding="utf-8", newline=""
        )
        return summary

    def _write_graph_csv(self, recipes: List[RecipeRecord]) -> None:
        nodes: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []
        for recipe in recipes:
            recipe_id = stable_numeric_id("21", recipe.source_path)
            nodes.append({
                "nodeId": recipe_id,
                "labels": "Recipe",
                "name": recipe.name,
                "preferredTerm": recipe.name,
                "fsn": f"{recipe.name} (Recipe)",
                "conceptType": "Recipe",
                "category": ",".join(recipe.categories),
                "difficulty": recipe.difficulty or "",
                "cuisineType": recipe.cuisine_type,
                "prepTime": recipe.prep_time,
                "cookTime": recipe.cook_time,
                "servings": recipe.servings,
                "tags": ",".join(recipe.tags),
                "filePath": recipe.source_path,
            })
            for index, ingredient in enumerate(recipe.ingredients, 1):
                ingredient_id = stable_numeric_id("31", f"{recipe.source_path}:ingredient:{index}:{ingredient.normalized_name}")
                quantity = ingredient.quantity
                amount = quantity.canonical_value if quantity.canonical_value is not None else quantity.raw
                unit = quantity.canonical_unit or quantity.unit
                nodes.append({
                    "nodeId": ingredient_id,
                    "labels": "Ingredient",
                    "name": ingredient.normalized_name,
                    "preferredTerm": ingredient.normalized_name,
                    "fsn": f"{ingredient.normalized_name} (Ingredient)",
                    "conceptType": "Ingredient",
                    "category": ingredient.category,
                    "amount": amount,
                    "unit": unit,
                    "isMain": ingredient.is_main,
                    "description": ingredient.evidence,
                })
                relationships.append({
                    "startNodeId": recipe_id,
                    "endNodeId": ingredient_id,
                    "relationshipType": "801000001",
                    "relationshipId": stable_relation_id(f"{recipe_id}:REQUIRES:{ingredient_id}"),
                    "amount": amount,
                    "unit": unit,
                })
            for step in recipe.steps:
                step_id = stable_numeric_id("41", f"{recipe.source_path}:step:{step.step_number}")
                nodes.append({
                    "nodeId": step_id,
                    "labels": "CookingStep",
                    "name": f"步骤{step.step_number}",
                    "preferredTerm": f"步骤{step.step_number}",
                    "fsn": f"步骤{step.step_number} (Cooking Step)",
                    "conceptType": "CookingStep",
                    "description": step.description,
                    "stepNumber": step.step_number,
                    "methods": ",".join(step.methods),
                    "tools": ",".join(step.tools),
                    "timeEstimate": step.time_estimate,
                })
                relationships.append({
                    "startNodeId": recipe_id,
                    "endNodeId": step_id,
                    "relationshipType": "801000003",
                    "relationshipId": stable_relation_id(f"{recipe_id}:CONTAINS_STEP:{step_id}"),
                    "step_order": step.step_number,
                })
            for category in recipe.categories:
                if category in CATEGORY_IDS:
                    relationships.append({
                        "startNodeId": recipe_id,
                        "endNodeId": CATEGORY_IDS[category],
                        "relationshipType": "801000004",
                        "relationshipId": stable_relation_id(f"{recipe_id}:CATEGORY:{category}"),
                    })

        self._write_csv(self.output_dir / "nodes.csv", NODE_HEADERS, nodes)
        self._write_csv(self.output_dir / "relationships.csv", RELATION_HEADERS, relationships)

    @staticmethod
    def _write_csv(path: Path, headers: List[str], rows: List[Dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore", lineterminator="\r\n")
            writer.writeheader()
            writer.writerows(rows)


class Neo4jRecipeWriter:
    """逐菜谱事务 upsert；不删除任何旧节点或关系。"""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def apply(self, recipe: RecipeRecord) -> str:
        with self.driver.session(database=self.database) as session:
            return session.execute_write(self._upsert_recipe, recipe)

    @staticmethod
    def _upsert_recipe(tx: Any, recipe: RecipeRecord) -> str:
        existing = tx.run(
            "MATCH (r:Recipe) WHERE r.filePath = $path OR r.name = $name RETURN r.nodeId AS id LIMIT 1",
            path=recipe.source_path,
            name=recipe.name,
        ).single()
        recipe_id = existing["id"] if existing and existing["id"] else stable_numeric_id("21", recipe.source_path)
        tx.run(
            """
            MERGE (r:Recipe {nodeId: $id})
            SET r.name=$name, r.preferredTerm=$name, r.filePath=$path,
                r.category=$category, r.difficulty=$difficulty, r.cuisineType=$cuisine,
                r.prepTime=$prep, r.cookTime=$cook, r.servings=$servings,
                r.tags=$tags, r.conceptType='Recipe', r.agentManaged=true
            """,
            id=recipe_id, name=recipe.name, path=recipe.source_path,
            category=",".join(recipe.categories), difficulty=recipe.difficulty,
            cuisine=recipe.cuisine_type, prep=recipe.prep_time, cook=recipe.cook_time,
            servings=recipe.servings, tags=",".join(recipe.tags),
        ).consume()

        for index, ingredient in enumerate(recipe.ingredients, 1):
            found = tx.run(
                """
                MATCH (r:Recipe {nodeId:$recipe_id})-[:REQUIRES]->(i:Ingredient)
                WHERE i.name=$name RETURN i.nodeId AS id LIMIT 1
                """,
                recipe_id=recipe_id, name=ingredient.normalized_name,
            ).single()
            ingredient_id = (
                found["id"] if found and found["id"]
                else stable_numeric_id("31", f"{recipe.source_path}:ingredient:{index}:{ingredient.normalized_name}")
            )
            quantity = ingredient.quantity
            amount = quantity.canonical_value if quantity.canonical_value is not None else quantity.raw
            unit = quantity.canonical_unit or quantity.unit
            tx.run(
                """
                MATCH (r:Recipe {nodeId:$recipe_id})
                MERGE (i:Ingredient {nodeId:$ingredient_id})
                SET i.name=$name, i.preferredTerm=$name, i.category=$category,
                    i.isMain=$is_main, i.description=$evidence, i.agentManaged=true
                MERGE (r)-[rel:REQUIRES]->(i)
                SET rel.relationshipId=$relationship_id, rel.amount=$amount, rel.unit=$unit
                """,
                recipe_id=recipe_id, ingredient_id=ingredient_id, name=ingredient.normalized_name,
                category=ingredient.category, is_main=ingredient.is_main, evidence=ingredient.evidence,
                relationship_id=stable_relation_id(f"{recipe_id}:REQUIRES:{ingredient_id}"),
                amount=str(amount), unit=unit,
            ).consume()

        for step in recipe.steps:
            found = tx.run(
                """
                MATCH (r:Recipe {nodeId:$recipe_id})-[:CONTAINS_STEP]->(s:CookingStep)
                WHERE s.stepNumber=$number RETURN s.nodeId AS id LIMIT 1
                """,
                recipe_id=recipe_id, number=step.step_number,
            ).single()
            step_id = (
                found["id"] if found and found["id"]
                else stable_numeric_id("41", f"{recipe.source_path}:step:{step.step_number}")
            )
            tx.run(
                """
                MATCH (r:Recipe {nodeId:$recipe_id})
                MERGE (s:CookingStep {nodeId:$step_id})
                SET s.name=$name, s.description=$description, s.stepNumber=$number,
                    s.methods=$methods, s.tools=$tools, s.timeEstimate=$time, s.agentManaged=true
                MERGE (r)-[rel:CONTAINS_STEP]->(s)
                SET rel.relationshipId=$relationship_id, rel.stepOrder=$number
                """,
                recipe_id=recipe_id, step_id=step_id, name=f"步骤{step.step_number}",
                description=step.description, number=step.step_number,
                methods=",".join(step.methods), tools=",".join(step.tools), time=step.time_estimate,
                relationship_id=stable_relation_id(f"{recipe_id}:CONTAINS_STEP:{step_id}"),
            ).consume()

        for category in recipe.categories:
            tx.run(
                """
                MATCH (r:Recipe {nodeId:$recipe_id})
                MERGE (c:Category {name:$category})
                MERGE (r)-[:BELONGS_TO_CATEGORY]->(c)
                """,
                recipe_id=recipe_id, category=category,
            ).consume()
        return str(recipe_id)
