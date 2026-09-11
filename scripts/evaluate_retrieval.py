#!/usr/bin/env python3
"""比较纯向量、BM25+向量和完整多路检索。"""

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import AdvancedGraphRAGSystem
from rag_modules.hybrid_retrieval import reciprocal_rank_fusion


def recipe_names(documents: List[Any]) -> List[str]:
    return [str(document.metadata.get("recipe_name", "")) for document in documents]


def score_case(expected: List[str], retrieved: List[str]) -> Dict[str, float]:
    expected_set = set(expected)
    hits = expected_set & set(retrieved)
    reciprocal_rank = 0.0
    for index, name in enumerate(retrieved, 1):
        if name in expected_set:
            reciprocal_rank = 1.0 / index
            break
    return {
        "recall_at_5": len(hits) / max(1, len(expected_set)),
        "mrr_at_5": reciprocal_rank,
    }


def run_strategy(system: AdvancedGraphRAGSystem, strategy: str, query: str, top_k: int) -> List[Any]:
    retriever = system.traditional_retrieval
    if strategy == "vector":
        return retriever.vector_search_enhanced(query, top_k)
    if strategy == "bm25_vector":
        candidate_k = top_k * system.config.retrieval_candidate_multiplier
        return reciprocal_rank_fusion({
            "bm25": retriever.bm25_search(query, candidate_k),
            "vector": retriever.vector_search_enhanced(query, candidate_k),
        }, top_k, system.config.rrf_k)
    channels = retriever.retrieve_channels(query, top_k)
    graph_documents = system.graph_rag_retrieval.graph_rag_search(
        query, top_k * system.config.retrieval_candidate_multiplier
    )
    if graph_documents:
        channels["graph"] = graph_documents
    return reciprocal_rank_fusion(
        channels,
        top_k,
        system.config.rrf_k,
        system.config.rrf_channel_weights(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="运行菜谱检索离线评测")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "evaluation" / "retrieval_cases.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "evaluation")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    cases = json.loads(args.dataset.read_text(encoding="utf-8"))
    system = AdvancedGraphRAGSystem()
    try:
        system.initialize_system()
        system.build_knowledge_base()
        details: List[Dict[str, Any]] = []
        for strategy in ("vector", "bm25_vector", "full"):
            for case in cases:
                started = time.perf_counter()
                documents = run_strategy(system, strategy, case["query"], args.top_k)
                latency_ms = int((time.perf_counter() - started) * 1000)
                retrieved = recipe_names(documents)
                details.append({
                    "strategy": strategy,
                    "case_id": case["id"],
                    "category": case["category"],
                    "query": case["query"],
                    "expected_recipes": case["expected_recipes"],
                    "retrieved_recipes": retrieved,
                    "latency_ms": latency_ms,
                    **score_case(case["expected_recipes"], retrieved),
                })
    finally:
        system._cleanup()

    summaries = []
    for strategy in ("vector", "bm25_vector", "full"):
        rows = [row for row in details if row["strategy"] == strategy]
        summaries.append({
            "strategy": strategy,
            "recall_at_5": round(mean(row["recall_at_5"] for row in rows), 4),
            "mrr_at_5": round(mean(row["mrr_at_5"] for row in rows), 4),
            "average_latency_ms": round(mean(row["latency_ms"] for row in rows), 2),
            "case_count": len(rows),
        })

    args.output.mkdir(parents=True, exist_ok=True)
    rrf_config = {
        "k": system.config.rrf_k,
        "candidate_multiplier": system.config.retrieval_candidate_multiplier,
        "channel_weights": system.config.rrf_channel_weights(),
    }
    result = {
        "top_k": args.top_k,
        "rrf_config": rrf_config,
        "summary": summaries,
        "details": details,
    }
    (args.output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\r\n", encoding="utf-8", newline=""
    )
    table = [
        "# 检索评测报告",
        "",
        "| 策略 | Recall@5 | MRR@5 | 平均延迟(ms) | 样例数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        table.append(
            f"| {row['strategy']} | {row['recall_at_5']:.4f} | {row['mrr_at_5']:.4f} | "
            f"{row['average_latency_ms']:.2f} | {row['case_count']} |"
        )
    weights = rrf_config["channel_weights"]
    table.extend([
        "",
        (
            f"RRF 配置：`k={rrf_config['k']}`，候选倍数 `{rrf_config['candidate_multiplier']}`，"
            f"BM25/向量/实体主题/图权重分别为 "
            f"`{weights['bm25']}/{weights['vector']}/{weights['entity_topic']}/{weights['graph']}`。"
        ),
        "",
        "详细命中记录见 `results.json`。",
        "",
    ])
    (args.output / "report.md").write_text("\r\n".join(table), encoding="utf-8", newline="")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
