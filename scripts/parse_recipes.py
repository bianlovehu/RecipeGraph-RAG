#!/usr/bin/env python3
"""批量运行菜谱解析 Agent，并生成可审查产物。"""

import argparse
import sys
from pathlib import Path
from typing import List

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import GraphRAGConfig
from recipe_agent import RecipeParsingAgent, RecipeRecord
from recipe_agent.artifacts import Neo4jRecipeWriter, RecipeArtifactStore, read_jsonl
from recipe_agent.models import AgentTrace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用受控 DeepSeek Agent 解析中文菜谱 Markdown")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "dishes")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "recipe_agent")
    parser.add_argument("--file", type=Path, action="append", help="只解析指定文件，可重复传入")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true", help="与 --resume 同用时重新解析显式指定的文件")
    parser.add_argument("--apply", action="store_true", help="校验成功后增量写入 Neo4j")
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    return parser


def source_key(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def collect_files(args: argparse.Namespace) -> List[Path]:
    files = [path.resolve() for path in args.file] if args.file else sorted(args.input.rglob("*.md"))
    files = [path for path in files if "template" not in {part.lower() for part in path.parts}]
    if args.limit is not None:
        files = files[:max(0, args.limit)]
    return files


def main() -> int:
    args = build_parser().parse_args()
    config = GraphRAGConfig.from_env()
    api_key = __import__("os").getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("缺少 DEEPSEEK_API_KEY，无法运行真实解析")

    output_dir = args.output.resolve()
    existing_recipe_rows = read_jsonl(output_dir / "recipes.jsonl") if args.resume else []
    existing_failure_rows = read_jsonl(output_dir / "failures.jsonl") if args.resume else []
    existing_trace_rows = read_jsonl(output_dir / "agent_traces.jsonl") if args.resume else []
    recipes = [RecipeRecord.model_validate(row) for row in existing_recipe_rows]
    failures = list(existing_failure_rows)
    traces = [AgentTrace.model_validate(row) for row in existing_trace_rows]
    completed = {recipe.source_path for recipe in recipes}
    recipes_by_source = {recipe.source_path: recipe for recipe in recipes}

    client = OpenAI(api_key=api_key, base_url=config.deepseek_base_url)
    agent = RecipeParsingAgent(
        client=client,
        model=config.llm_model,
        max_rounds=args.max_rounds or config.agent_max_rounds,
        max_retries=args.max_retries or config.agent_max_retries,
    )
    writer = None
    if args.apply:
        writer = Neo4jRecipeWriter(
            config.neo4j_uri, config.neo4j_user, config.neo4j_password, config.neo4j_database
        )

    try:
        for path in collect_files(args):
            key = source_key(path)
            if args.resume and key in completed and not args.force:
                if writer:
                    node_id = writer.apply(recipes_by_source[key])
                    print(f"复用已有产物写入 Neo4j，Recipe.nodeId={node_id}")
                print(f"跳过已完成: {key}")
                continue
            print(f"解析: {key}")
            markdown = path.read_text(encoding="utf-8")
            result = agent.parse(markdown, key)
            traces.extend(result.traces)
            if result.recipe is None:
                failures.append({
                    "source_path": key,
                    "error": result.error,
                    "retries": result.retries,
                    "duration_ms": result.duration_ms,
                })
                print(f"失败: {result.error}")
                continue
            recipes = [recipe for recipe in recipes if recipe.source_path != key]
            recipes.append(result.recipe)
            if writer:
                node_id = writer.apply(result.recipe)
                print(f"已写入 Neo4j，Recipe.nodeId={node_id}")
            print(f"成功: {result.recipe.name}")
    finally:
        if writer:
            writer.close()

    summary = RecipeArtifactStore(output_dir).write(recipes, failures, traces)
    print(
        f"完成: 成功 {summary['succeeded']}，失败 {summary['failed']}，"
        f"产物目录 {output_dir}"
    )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
