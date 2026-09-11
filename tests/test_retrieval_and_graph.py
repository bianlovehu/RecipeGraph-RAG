import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from rag_modules.graph_rag_retrieval import GraphPath, GraphQuery, GraphRAGRetrieval, KnowledgeSubgraph, QueryType
from rag_modules.hybrid_retrieval import (
    HybridRetrievalModule,
    chinese_tokenize,
    document_identity,
    reciprocal_rank_fusion,
)
from rag_modules.intelligent_query_router import IntelligentQueryRouter, QueryAnalysis, SearchStrategy
from rag_modules.llm_json import request_json_object


class FakeNode(dict):
    def __init__(self, node_id, name, labels):
        super().__init__(nodeId=node_id, name=name)
        self.labels = set(labels)
        self.element_id = node_id


class FakeRelationship(dict):
    def __init__(self, rel_type, start_node, end_node, **properties):
        super().__init__(properties)
        self.type = rel_type
        self.start_node = start_node
        self.end_node = end_node


class FakePath:
    def __init__(self, nodes, relationships):
        self.nodes = nodes
        self.relationships = relationships


class FusionTests(unittest.TestCase):
    def test_document_identity_prefers_recipe_over_chunk_ids(self):
        first = Document(page_content="a", metadata={"recipe_name": "麻婆豆腐", "parent_id": "chunk-parent-a"})
        second = Document(page_content="b", metadata={"recipe_name": "麻婆豆腐", "parent_id": "chunk-parent-b"})
        self.assertEqual(document_identity(first), document_identity(second))

    def test_rrf_deduplicates_per_recipe_and_records_sources(self):
        a1 = Document(page_content="A bm25", metadata={"parent_id": "A", "recipe_name": "A"})
        a2 = Document(page_content="A duplicate", metadata={"parent_id": "A", "recipe_name": "A"})
        a3 = Document(page_content="A vector", metadata={"parent_id": "A", "recipe_name": "A"})
        b = Document(page_content="B", metadata={"parent_id": "B", "recipe_name": "B"})
        results = reciprocal_rank_fusion({"bm25": [a1, a2, b], "vector": [a3]}, 5, rrf_k=60)
        self.assertEqual([doc.metadata["document_id"] for doc in results], ["A", "B"])
        self.assertEqual(results[0].metadata["retrieval_sources"], ["bm25", "vector"])
        self.assertEqual(results[0].metadata["source_ranks"]["bm25"], 1)
        self.assertAlmostEqual(results[0].metadata["source_contributions"]["bm25"], 1 / 61)

    def test_rrf_applies_channel_weights(self):
        bm25 = Document(page_content="A", metadata={"recipe_name": "A"})
        graph = Document(page_content="B", metadata={"recipe_name": "B"})
        results = reciprocal_rank_fusion(
            {"bm25": [bm25], "graph": [graph]},
            2,
            rrf_k=60,
            channel_weights={"bm25": 1.0, "graph": 0.5},
        )
        self.assertEqual([doc.metadata["recipe_name"] for doc in results], ["A", "B"])
        self.assertAlmostEqual(results[1].metadata["source_contributions"]["graph"], 0.5 / 61)

    def test_chinese_bm25_is_really_callable(self):
        module = HybridRetrievalModule.__new__(HybridRetrievalModule)
        module.bm25_retriever = BM25Retriever.from_documents([
            Document(page_content="麻婆豆腐 川菜 花椒 豆腐", metadata={"recipe_name": "麻婆豆腐"}),
            Document(page_content="清蒸鲈鱼 蒸锅 鲈鱼", metadata={"recipe_name": "清蒸鲈鱼"}),
            Document(page_content="牛奶燕麦 早餐 牛奶", metadata={"recipe_name": "牛奶燕麦"}),
        ], preprocess_func=chinese_tokenize)
        documents = module.bm25_search("花椒", 1)
        self.assertEqual(documents[0].metadata["recipe_name"], "麻婆豆腐")
        self.assertEqual(documents[0].metadata["search_type"], "bm25")


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.graph = GraphRAGRetrieval(
            SimpleNamespace(neo4j_database="neo4j", max_graph_nodes=50),
            llm_client=Mock(),
        )

    def test_path_preserves_relation_type_and_direction(self):
        recipe = FakeNode("r1", "土豆烧肉", ["Recipe"])
        ingredient = FakeNode("i1", "土豆", ["Ingredient"])
        relationship = FakeRelationship("REQUIRES", recipe, ingredient, amount="2", unit="个")
        parsed = self.graph._parse_neo4j_path({
            "path": FakePath([ingredient, recipe], [relationship]),
            "path_len": 1,
        }, "entity_relation")
        self.assertEqual(parsed.relationships[0]["type"], "REQUIRES")
        self.assertEqual(parsed.relationships[0]["start_id"], "r1")
        self.assertIn("<-[REQUIRES]-", self.graph._build_path_description(parsed))

    def test_graph_document_contains_traceable_evidence(self):
        path = GraphPath(
            nodes=[
                {"id": "i1", "name": "土豆", "labels": ["Ingredient"], "properties": {}},
                {"id": "r1", "name": "土豆烧肉", "labels": ["Recipe"], "properties": {}},
            ],
            relationships=[{"type": "REQUIRES", "start_id": "r1", "end_id": "i1", "properties": {}}],
            path_length=1,
            relevance_score=0.8,
            path_type="entity_relation",
        )
        document = self.graph._paths_to_documents([path], "土豆有什么菜")[0]
        self.assertEqual(document.metadata["node_id"], "r1")
        self.assertEqual(document.metadata["relation_types"], ["REQUIRES"])
        self.assertTrue(document.metadata["path_signature"])
        self.assertEqual(document.metadata["evidence"], [document.page_content])

    def test_reasoning_is_derived_from_relations(self):
        subgraph = KnowledgeSubgraph(
            central_nodes=[{"id": "r1", "name": "土豆烧肉", "labels": ["Recipe"], "properties": {}}],
            connected_nodes=[
                {"id": "i1", "name": "土豆", "labels": ["Ingredient"], "properties": {}},
                {"id": "i2", "name": "猪肉", "labels": ["Ingredient"], "properties": {}},
            ],
            relationships=[
                {"type": "REQUIRES", "start_id": "r1", "end_id": "i1", "properties": {}},
                {"type": "REQUIRES", "start_id": "r1", "end_id": "i2", "properties": {}},
            ],
            graph_metrics={},
            reasoning_chains=[],
        )
        chains = self.graph.graph_structure_reasoning(subgraph, "土豆和猪肉")
        self.assertTrue(any("土豆烧肉 需要 土豆" in chain for chain in chains))
        self.assertTrue(any("共同出现" in chain for chain in chains))
        self.assertFalse(any("因果" in chain for chain in chains))

    def test_shortest_path_requires_target(self):
        query = GraphQuery(QueryType.PATH_FINDING, ["土豆"], target_entities=[])
        self.assertEqual(self.graph._find_shortest_paths(query, Mock()), [])

    def test_subgraph_documents_only_emit_recipe_candidates(self):
        subgraph = KnowledgeSubgraph(
            central_nodes=[{"id": "i1", "name": "土豆", "labels": ["Ingredient"], "properties": {}}],
            connected_nodes=[
                {"id": "r1", "name": "土豆烧肉", "labels": ["Recipe"], "properties": {}},
                {"id": "i2", "name": "猪肉", "labels": ["Ingredient"], "properties": {}},
            ],
            relationships=[
                {"type": "REQUIRES", "start_id": "r1", "end_id": "i1", "properties": {}},
                {"type": "REQUIRES", "start_id": "r1", "end_id": "i2", "properties": {}},
            ],
            graph_metrics={"density": 0.5},
            reasoning_chains=[],
        )
        documents = self.graph._subgraph_to_documents(subgraph, [], "土豆有什么菜")
        self.assertEqual([doc.metadata["recipe_name"] for doc in documents], ["土豆烧肉"])
        self.assertTrue(all(doc.metadata["node_id"].startswith("r") for doc in documents))

    def test_relation_allowlist_includes_real_similarity_edges(self):
        self.assertEqual(self.graph._safe_relation_types(["SIMILAR", "DROP_ALL"]), ["SIMILAR"])


class JsonRequestTests(unittest.TestCase):
    def test_empty_json_response_is_retried_with_thinking_disabled(self):
        empty = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=""), finish_reason="stop")]
        )
        valid = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'), finish_reason="stop")]
        )
        client = Mock()
        client.chat.completions.create.side_effect = [empty, valid]
        self.assertEqual(request_json_object(client, "model", "return JSON"), {"ok": True})
        self.assertEqual(client.chat.completions.create.call_count, 2)
        self.assertEqual(
            client.chat.completions.create.call_args.kwargs["extra_body"],
            {"thinking": {"type": "disabled"}},
        )


class RouterFallbackTests(unittest.TestCase):
    def test_empty_graph_result_falls_back_to_hybrid(self):
        fallback_document = Document(page_content="fallback", metadata={"recipe_name": "土豆烧肉"})
        traditional = Mock()
        traditional.hybrid_search.return_value = [fallback_document]
        graph = Mock()
        graph.graph_rag_search.return_value = []
        router = IntelligentQueryRouter(traditional, graph, Mock(), SimpleNamespace())
        router.analyze_query = Mock(return_value=QueryAnalysis(
            query_complexity=0.8,
            relationship_intensity=0.8,
            reasoning_required=True,
            entity_count=2,
            recommended_strategy=SearchStrategy.GRAPH_RAG,
            confidence=0.9,
            reasoning="关系查询",
        ))
        documents, analysis = router.route_query("土豆和猪肉", 5)
        self.assertEqual(documents, [fallback_document])
        self.assertTrue(analysis.fallback_used)
        self.assertEqual(analysis.fallback_reason, "graph_rag_empty")


if __name__ == "__main__":
    unittest.main()
