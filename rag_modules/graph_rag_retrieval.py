"""
真正的图RAG检索模块
基于图结构的知识推理和检索，而非简单的关键词匹配
"""

import logging
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document
from neo4j import GraphDatabase

from .llm_json import request_json_object

logger = logging.getLogger(__name__)

ALLOWED_RELATION_TYPES = {
    "REQUIRES",
    "CONTAINS_STEP",
    "BELONGS_TO",
    "BELONGS_TO_CATEGORY",
    "DIFFICULTY_LEVEL",
    "HAS_DIFFICULTY_LEVEL",
    "HAS_CONCEPT_TYPE",
    "NEXT_STEP",
    "SIMILAR",
    "USES_SAME_METHOD",
    "USES_SAME_TOOL",
}

class QueryType(Enum):
    """查询类型枚举"""
    ENTITY_RELATION = "entity_relation"  # 实体关系查询：A和B有什么关系？
    MULTI_HOP = "multi_hop"  # 多跳查询：A通过什么连接到C？
    SUBGRAPH = "subgraph"  # 子图查询：A相关的所有信息
    PATH_FINDING = "path_finding"  # 路径查找：从A到B的最佳路径
    CLUSTERING = "clustering"  # 聚类查询：和A相似的都有什么？

@dataclass
class GraphQuery:
    """图查询结构"""
    query_type: QueryType
    source_entities: List[str]
    target_entities: List[str] = None
    relation_types: List[str] = None
    max_depth: int = 2
    max_nodes: int = 50
    constraints: Dict[str, Any] = None

@dataclass
class GraphPath:
    """图路径结构"""
    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    path_length: int
    relevance_score: float
    path_type: str

@dataclass
class KnowledgeSubgraph:
    """知识子图结构"""
    central_nodes: List[Dict[str, Any]]
    connected_nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    graph_metrics: Dict[str, float]
    reasoning_chains: List[List[str]]

class GraphRAGRetrieval:
    """
    真正的图RAG检索系统
    核心特点：
    1. 查询意图理解：识别图查询模式
    2. 多跳图遍历：深度关系探索
    3. 子图提取：相关知识网络
    4. 图结构推理：基于拓扑的推理
    5. 动态查询规划：自适应遍历策略
    """
    
    def __init__(self, config, llm_client):
        self.config = config
        self.llm_client = llm_client
        self.driver = None
        
        # 图结构缓存
        self.entity_cache = {}
        self.relation_cache = {}
        self.subgraph_cache = {}
        
    def initialize(self):
        """初始化图RAG检索系统"""
        logger.info("初始化图RAG检索系统...")
        
        # 连接Neo4j
        try:
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri, 
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            # 测试连接
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j连接成功")
        except Exception as e:
            logger.error(f"Neo4j连接失败: {e}")
            return
        
        # 预热：构建实体和关系索引
        self._build_graph_index()
        
    def _build_graph_index(self):
        """构建图索引以加速查询"""
        logger.info("构建图结构索引...")
        
        try:
            with self.driver.session() as session:
                # 构建实体索引 - 修复Neo4j语法兼容性问题
                entity_query = """
                MATCH (n)
                WHERE n.nodeId IS NOT NULL
                WITH n, COUNT { (n)--() } as degree
                RETURN labels(n) as node_labels, n.nodeId as node_id, 
                       n.name as name, n.category as category, degree
                ORDER BY degree DESC
                LIMIT 1000
                """
                
                result = session.run(entity_query)
                for record in result:
                    node_id = record["node_id"]
                    self.entity_cache[node_id] = {
                        "labels": record["node_labels"],
                        "name": record["name"],
                        "category": record["category"],
                        "degree": record["degree"]
                    }
                
                # 构建关系类型索引
                relation_query = """
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as frequency
                ORDER BY frequency DESC
                """
                
                result = session.run(relation_query)
                for record in result:
                    rel_type = record["rel_type"]
                    self.relation_cache[rel_type] = record["frequency"]
                    
                logger.info(f"索引构建完成: {len(self.entity_cache)}个实体, {len(self.relation_cache)}个关系类型")
                
        except Exception as e:
            logger.error(f"构建图索引失败: {e}")
    
    def understand_graph_query(self, query: str) -> GraphQuery:
        """
        理解查询的图结构意图
        这是图RAG的核心：从自然语言到图查询的转换
        """
        prompt = f"""
        作为图数据库专家，分析以下查询的图结构意图，并将自然语言问题映射到**已有图结构**上。
        
        已知图中大致有以下节点和关系：
        - 节点类型：
          - Recipe：菜谱节点，包含 name、description、cuisineType（如"川菜"）、category、tags、prepTime、cookTime 等属性
          - Ingredient：食材节点，包含 name、category（如"蔬菜"、"蛋白质" 等）
          - Category：菜品分类（如"川菜"、"家常菜"、"素菜"）
          - CookingStep：烹饪步骤
        - 主要关系：
          - (Recipe)-[:REQUIRES]->(Ingredient)
          - (Recipe)-[:BELONGS_TO_CATEGORY]->(Category)
          - (Recipe)-[:CONTAINS_STEP]->(CookingStep)
        
        请根据上述图结构分析下面的查询：
        
        查询：{query}
        
        请识别：
        1. 查询类型：
           - entity_relation: 询问实体间的直接关系（如：鸡肉和胡萝卜能一起做菜吗？）
           - multi_hop: 需要多跳推理（如：鸡肉配什么蔬菜？需要：鸡肉→菜品→食材→蔬菜）
           - subgraph: 需要完整子图（如：川菜有什么特色？需要川菜相关的完整知识网络）
           - path_finding: 路径查找（如：从食材到成品菜的制作路径）
           - clustering: 聚类相似性（如：和宫保鸡丁类似的菜有哪些？）
        
        2. source_entities：
           - 只包含在图中**很有可能有对应节点**的具体实体名称
           - 优先选择：菜系（如"川菜"）、具体菜名（如"宫保鸡丁"）、食材名（如"鸡肉"、"豆腐"）
           - 不要把抽象概念或约束（如"糖尿病饮食限制"、"具体川菜菜品"、"健康饮食"、"30分钟内"）放进 source_entities
        
        3. target_entities：
           - 只在确实需要限制「路径终点」时填写
           - 同样只能使用可能出现在 Recipe / Ingredient / Category 节点上的名称（如"蔬菜"、"素菜"、具体菜名）
           - 如果不确定目标实体怎么映射到图中，请返回空列表 []
        
        4. relation_types：本次推理中希望优先考虑的关系类型列表
           - 例如：["REQUIRES", "BELONGS_TO_CATEGORY"]
        
        5. max_depth：建议的图遍历深度（1-3 之间的整数）
        
        6. constraints：可选的**属性级约束**，用于表达图结构之外的过滤条件，例如：
           - 健康/饮食限制（如"糖尿病"、"低糖"）
           - 时间限制（如"30分钟内"）
           - 口味偏好（如"清淡"、"少油"）
           用一个字典描述，例如：
           {{
             "health": ["糖尿病", "低糖"],
             "time": {{"max_minutes": 30}},
             "style": ["川菜"]
           }}
        
        示例1：
        查询："鸡肉配什么蔬菜好？"
        期望分析：这是 multi_hop 查询，需要通过"鸡肉→使用鸡肉的菜品→这些菜品使用的蔬菜"的路径推理。
        
        返回JSON示例：
        {{
          "query_type": "multi_hop",
          "source_entities": ["鸡肉"],
          "target_entities": ["蔬菜"],
          "relation_types": ["REQUIRES", "BELONGS_TO_CATEGORY"],
          "max_depth": 3,
          "constraints": {{}}
        }}
        
        示例2：
        查询："适合糖尿病人吃的低糖川菜有哪些，并且制作时间不超过30分钟？"
        期望分析：
          - 图中可以直接对应的实体：主要是菜系 "川菜"
          - 糖尿病/低糖/30分钟 属于属性级约束，不能当作节点
          - 可以使用 subgraph 或 multi_hop，以 "川菜" 为核心实体，结合属性约束做后续过滤
        
        返回JSON示例：
        {{
          "query_type": "subgraph",
          "source_entities": ["川菜"],
          "target_entities": [],
          "relation_types": ["BELONGS_TO_CATEGORY", "REQUIRES"],
          "max_depth": 2,
          "constraints": {{
            "health": ["糖尿病", "低糖"],
            "time": {{"max_minutes": 30}}
          }}
        }}
        
        请严格返回一个合法的 JSON 对象，不要包含任何多余的说明文字。
        """
        
        try:
            result = request_json_object(
                self.llm_client,
                self.config.llm_model,
                prompt,
                max_tokens=1200,
            )
            
            requested_relations = {
                str(value) for value in result.get("relation_types", [])
                if str(value) in ALLOWED_RELATION_TYPES
            }
            max_depth = min(max(int(result.get("max_depth", 2)), 1), 4)
            query_type = QueryType(result.get("query_type", "subgraph"))
            source_entities = [str(value) for value in result.get("source_entities", []) if str(value).strip()]
            target_entities = [str(value) for value in result.get("target_entities", []) if str(value).strip()]
            if query_type == QueryType.PATH_FINDING and not target_entities:
                if len(source_entities) >= 2:
                    source_entities, target_entities = source_entities[:1], source_entities[1:]
                else:
                    query_type = QueryType.MULTI_HOP
            return GraphQuery(
                query_type=query_type,
                source_entities=source_entities,
                target_entities=target_entities,
                relation_types=sorted(requested_relations) or sorted(ALLOWED_RELATION_TYPES),
                max_depth=max_depth,
                max_nodes=min(max(getattr(self.config, "max_graph_nodes", 50), 1), 100),
                constraints=result.get("constraints", {}),
            )
            
        except Exception as e:
            logger.error(f"查询意图理解失败: {e}")
            # 降级方案：默认子图查询
            return GraphQuery(
                query_type=QueryType.SUBGRAPH,
                source_entities=[query],
                max_depth=2
            )
    
    def multi_hop_traversal(self, graph_query: GraphQuery) -> List[GraphPath]:
        """
        多跳图遍历：这是图RAG的核心优势
        通过图结构发现隐含的知识关联
        """
        logger.info(f"执行多跳遍历: {graph_query.source_entities} -> {graph_query.target_entities}")
        
        paths: List[GraphPath] = []
        if not self.driver:
            logger.error("Neo4j连接未建立")
            return paths

        graph_query.max_depth = min(max(int(graph_query.max_depth or 2), 1), 4)
        graph_query.max_nodes = min(max(int(graph_query.max_nodes or 50), 1), 100)
        relation_types = self._safe_relation_types(graph_query.relation_types)
        try:
            with self.driver.session(database=getattr(self.config, "neo4j_database", "neo4j")) as session:
                if graph_query.query_type == QueryType.MULTI_HOP:
                    if len(graph_query.source_entities) >= 2:
                        pair_query = GraphQuery(
                            query_type=QueryType.PATH_FINDING,
                            source_entities=graph_query.source_entities[:1],
                            target_entities=graph_query.source_entities[1:],
                            relation_types=relation_types,
                            max_depth=graph_query.max_depth,
                            max_nodes=graph_query.max_nodes,
                            constraints=graph_query.constraints,
                        )
                        paths.extend(self._find_shortest_paths(pair_query, session))
                    else:
                        paths.extend(self._run_multi_hop_query(graph_query, session, relation_types))
                        if not paths and graph_query.target_entities:
                            fallback_query = GraphQuery(
                                query_type=QueryType.MULTI_HOP,
                                source_entities=graph_query.source_entities,
                                target_entities=[],
                                relation_types=relation_types,
                                max_depth=graph_query.max_depth,
                                max_nodes=graph_query.max_nodes,
                                constraints=graph_query.constraints,
                            )
                            paths.extend(self._run_multi_hop_query(fallback_query, session, relation_types))
                elif graph_query.query_type == QueryType.ENTITY_RELATION:
                    paths.extend(self._find_entity_relations(graph_query, session))
                elif graph_query.query_type == QueryType.PATH_FINDING:
                    paths.extend(self._find_shortest_paths(graph_query, session))
        except Exception as e:
            logger.error(f"多跳遍历失败: {e}")

        unique: Dict[str, GraphPath] = {}
        for path in paths:
            signature = self._path_signature(path)
            if signature not in unique or path.relevance_score > unique[signature].relevance_score:
                unique[signature] = path
        paths = sorted(unique.values(), key=lambda item: item.relevance_score, reverse=True)
        logger.info(f"多跳遍历完成，找到 {len(paths)} 条路径")
        return paths[:graph_query.max_nodes]

    def _run_multi_hop_query(
        self, graph_query: GraphQuery, session: Any, relation_types: List[str]
    ) -> List[GraphPath]:
        """执行单源多跳查询；目标约束无命中时由调用方负责降级。"""
        target_clause = ""
        if graph_query.target_entities:
            target_clause = """
            AND ANY(keyword IN $targets WHERE
                (target.name IS NOT NULL AND
                 (toString(target.name) CONTAINS keyword OR keyword CONTAINS toString(target.name)))
                OR (target.category IS NOT NULL AND toString(target.category) CONTAINS keyword)
            )
            """
        query = f"""
        UNWIND $sources AS source_name
        MATCH (source)
        WHERE source.nodeId = source_name
           OR (source.name IS NOT NULL AND
               (toString(source.name) CONTAINS source_name OR source_name CONTAINS toString(source.name)))
        MATCH path=(source)-[*1..{graph_query.max_depth}]-(target)
        WHERE source <> target
          AND ALL(rel IN relationships(path) WHERE type(rel) IN $relation_types)
          {target_clause}
        RETURN path, length(path) AS path_len
        LIMIT $limit
        """
        result = session.run(query, {
            "sources": graph_query.source_entities,
            "targets": graph_query.target_entities or [],
            "relation_types": relation_types,
            "limit": graph_query.max_nodes,
        })
        paths: List[GraphPath] = []
        for record in result:
            path_data = self._parse_neo4j_path(record, "multi_hop")
            if path_data:
                path_data.relevance_score = self._score_path(path_data, graph_query)
                paths.append(path_data)
        return paths
    
    def extract_knowledge_subgraph(self, graph_query: GraphQuery) -> KnowledgeSubgraph:
        """
        提取知识子图：获取实体相关的完整知识网络
        这体现了图RAG的整体性思维
        """
        logger.info(f"提取知识子图: {graph_query.source_entities}")
        
        if not self.driver:
            logger.error("Neo4j连接未建立")
            return self._fallback_subgraph_extraction(graph_query)
        
        max_depth = min(max(int(graph_query.max_depth or 2), 1), 4)
        max_nodes = min(max(int(graph_query.max_nodes or 50), 1), 100)
        relation_types = self._safe_relation_types(graph_query.relation_types)
        try:
            with self.driver.session(database=getattr(self.config, "neo4j_database", "neo4j")) as session:
                cypher_query = f"""
                UNWIND $source_entities AS entity_name
                MATCH (source)
                WHERE source.nodeId = entity_name
                   OR (source.name IS NOT NULL AND
                       (toString(source.name) CONTAINS entity_name OR entity_name CONTAINS toString(source.name)))
                CALL {{
                    WITH source
                    MATCH path=(source)-[*1..{max_depth}]-(neighbor)
                    WHERE ALL(rel IN relationships(path) WHERE type(rel) IN $relation_types)
                    RETURN path
                    LIMIT $max_nodes
                }}
                RETURN source, collect(path) AS paths
                LIMIT 1
                """
                record = session.run(cypher_query, {
                    "source_entities": graph_query.source_entities,
                    "relation_types": relation_types,
                    "max_nodes": max_nodes,
                }).single()
                if record:
                    return self._build_knowledge_subgraph(record)
        except Exception as e:
            logger.error(f"子图提取失败: {e}")

        return self._fallback_subgraph_extraction(graph_query)
    
    def graph_structure_reasoning(self, subgraph: KnowledgeSubgraph, query: str) -> List[str]:
        """
        基于图结构的推理：这是图RAG的智能之处
        不仅检索信息，还能进行逻辑推理
        """
        reasoning_chains = []
        
        try:
            for pattern in self._identify_reasoning_patterns(subgraph):
                reasoning_chains.extend(self._build_reasoning_chain(pattern, subgraph))
            validated_chains = self._validate_reasoning_chains(reasoning_chains, query)
            
            logger.info(f"图结构推理完成，生成 {len(validated_chains)} 条推理链")
            return validated_chains
            
        except Exception as e:
            logger.error(f"图结构推理失败: {e}")
            return []
    
    def adaptive_query_planning(self, query: str) -> List[GraphQuery]:
        """
        自适应查询规划：根据查询复杂度动态调整策略
        """
        # 分析查询复杂度
        complexity_score = self._analyze_query_complexity(query)
        
        query_plans = []
        
        if complexity_score < 0.3:
            # 简单查询：直接邻居查询
            plan = GraphQuery(
                query_type=QueryType.ENTITY_RELATION,
                source_entities=[query],
                max_depth=1,
                max_nodes=20
            )
            query_plans.append(plan)
            
        elif complexity_score < 0.7:
            # 中等复杂度：多跳查询
            plan = GraphQuery(
                query_type=QueryType.MULTI_HOP,
                source_entities=[query],
                max_depth=2,
                max_nodes=50
            )
            query_plans.append(plan)
            
        else:
            # 复杂查询：子图提取 + 推理
            plan1 = GraphQuery(
                query_type=QueryType.SUBGRAPH,
                source_entities=[query],
                max_depth=3,
                max_nodes=100
            )
            plan2 = GraphQuery(
                query_type=QueryType.MULTI_HOP,
                source_entities=[query],
                max_depth=3,
                max_nodes=50
            )
            query_plans.extend([plan1, plan2])
            
        return query_plans
    
    def graph_rag_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        图RAG主搜索接口：整合所有图RAG能力
        """
        logger.info(f"开始图RAG检索: {query}")
        
        if not self.driver:
            logger.warning("Neo4j连接未建立，返回空结果")
            return []
        
        # 1. 查询意图理解
        graph_query = self.understand_graph_query(query)
        logger.info(f"查询类型: {graph_query.query_type.value}")
        
        results = []
        
        try:
            # 2. 根据查询类型执行不同策略
            if graph_query.query_type in [QueryType.MULTI_HOP, QueryType.PATH_FINDING]:
                # 多跳遍历 / 路径查找
                paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths, query))
                
            elif graph_query.query_type in [QueryType.SUBGRAPH, QueryType.CLUSTERING]:
                # 子图提取 / 聚类查询：都视为“围绕核心实体的局部知识网络”
                subgraph = self.extract_knowledge_subgraph(graph_query)
                
                # 图结构推理
                reasoning_chains = self.graph_structure_reasoning(subgraph, query)
                
                results.extend(self._subgraph_to_documents(subgraph, reasoning_chains, query))
                
            elif graph_query.query_type == QueryType.ENTITY_RELATION:
                # 实体关系查询（可以视为一跳 / 少量跳的路径查询）
                paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths, query))
            
            # 3. 图结构相关性排序
            results = self._rank_by_graph_relevance(results, query)
            
            logger.info(f"图RAG检索完成，返回 {len(results[:top_k])} 个结果")
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"图RAG检索失败: {e}")
            return []
    
    # ========== 辅助方法 ==========
    
    @staticmethod
    def _safe_relation_types(relation_types: Optional[List[str]]) -> List[str]:
        requested = {str(value) for value in (relation_types or [])}
        safe = sorted(requested & ALLOWED_RELATION_TYPES)
        return safe or sorted(ALLOWED_RELATION_TYPES)

    @staticmethod
    def _node_dict(node: Any) -> Dict[str, Any]:
        properties = dict(node)
        return {
            "id": str(properties.get("nodeId", getattr(node, "element_id", ""))),
            "name": str(properties.get("name", "")),
            "labels": list(getattr(node, "labels", [])),
            "properties": properties,
        }

    def _relationship_dict(self, relationship: Any) -> Dict[str, Any]:
        properties = dict(relationship)
        start_node = getattr(relationship, "start_node", None)
        end_node = getattr(relationship, "end_node", None)
        start_id = self._node_dict(start_node)["id"] if start_node is not None else ""
        end_id = self._node_dict(end_node)["id"] if end_node is not None else ""
        relation_type = getattr(relationship, "type", None) or properties.get("type", "RELATED")
        return {
            "type": str(relation_type),
            "start_id": start_id,
            "end_id": end_id,
            "properties": properties,
        }

    def _parse_neo4j_path(self, record: Any, path_type: str = "multi_hop") -> Optional[GraphPath]:
        """解析 Neo4j Path，同时保留真实关系类型、方向和属性。"""
        try:
            path = record.get("path") if hasattr(record, "get") else record["path"]
            if path is not None:
                raw_nodes = list(path.nodes)
                raw_relationships = list(path.relationships)
            else:
                raw_nodes = list(record.get("path_nodes", []))
                raw_relationships = list(record.get("rels", []))
            path_nodes = [self._node_dict(node) for node in raw_nodes]
            relationships = [self._relationship_dict(rel) for rel in raw_relationships]
            path_length = int(record.get("path_len", len(relationships)))
            relevance = float(record.get("relevance", 1.0 / (1.0 + max(1, path_length))))
            return GraphPath(
                nodes=path_nodes,
                relationships=relationships,
                path_length=path_length,
                relevance_score=relevance,
                path_type=path_type,
            )
        except Exception as e:
            logger.error(f"路径解析失败: {e}")
            return None
    
    def _build_knowledge_subgraph(self, record) -> KnowledgeSubgraph:
        """构建知识子图对象"""
        try:
            central = self._node_dict(record["source"])
            nodes_by_id: Dict[str, Dict[str, Any]] = {central["id"]: central}
            relationships_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
            for path in record.get("paths", []):
                for node in path.nodes:
                    node_data = self._node_dict(node)
                    nodes_by_id[node_data["id"]] = node_data
                for rel in path.relationships:
                    rel_data = self._relationship_dict(rel)
                    key = (rel_data["start_id"], rel_data["type"], rel_data["end_id"])
                    relationships_by_key[key] = rel_data
            connected_nodes = [node for node_id, node in nodes_by_id.items() if node_id != central["id"]]
            relationships = list(relationships_by_key.values())
            node_count = len(connected_nodes)
            relation_count = len(relationships)
            total_node_count = len(nodes_by_id)
            density = (
                relation_count / (total_node_count * (total_node_count - 1) / 2)
                if total_node_count > 1 else 0.0
            )
            return KnowledgeSubgraph(
                central_nodes=[central],
                connected_nodes=connected_nodes,
                relationships=relationships,
                graph_metrics={
                    "node_count": node_count,
                    "relationship_count": relation_count,
                    "density": density,
                },
                reasoning_chains=[],
            )
        except Exception as e:
            logger.error(f"构建知识子图失败: {e}")
            return KnowledgeSubgraph(
                central_nodes=[],
                connected_nodes=[],
                relationships=[],
                graph_metrics={},
                reasoning_chains=[]
            )
    
    def _paths_to_documents(self, paths: List[GraphPath], query: str) -> List[Document]:
        """将图路径转换为Document对象"""
        documents = []
        
        for i, path in enumerate(paths):
            path_desc = self._build_path_description(path)
            signature = self._path_signature(path)
            recipe_node = next((node for node in path.nodes if "Recipe" in node.get("labels", [])), None)
            primary_node = recipe_node or (path.nodes[0] if path.nodes else {})
            doc = Document(
                page_content=path_desc,
                metadata={
                    "search_type": "graph_path",
                    "document_id": primary_node.get("id", signature),
                    "node_id": primary_node.get("id", ""),
                    "parent_id": primary_node.get("id", ""),
                    "path_length": path.path_length,
                    "relevance_score": path.relevance_score,
                    "path_type": path.path_type,
                    "path_signature": signature,
                    "relation_types": [rel.get("type", "") for rel in path.relationships],
                    "node_count": len(path.nodes),
                    "relationship_count": len(path.relationships),
                    "recipe_name": primary_node.get("name", "图结构结果"),
                    "evidence": [path_desc],
                    "reasoning_chains": [path_desc],
                },
            )
            documents.append(doc)
            
        return documents
    
    def _subgraph_to_documents(self, subgraph: KnowledgeSubgraph, 
                              reasoning_chains: List[str], query: str) -> List[Document]:
        """将子图投影为菜谱级文档，避免把食材或分类混入菜谱候选。"""
        all_nodes = subgraph.central_nodes + subgraph.connected_nodes
        nodes_by_id = {node.get("id", ""): node for node in all_nodes}
        recipe_nodes = [node for node in all_nodes if "Recipe" in node.get("labels", [])]
        central_ids = {node.get("id", "") for node in subgraph.central_nodes}
        documents: List[Document] = []

        seen_recipe_ids: Set[str] = set()
        for recipe in recipe_nodes:
            recipe_id = recipe.get("id", "")
            if not recipe_id or recipe_id in seen_recipe_ids:
                continue
            seen_recipe_ids.add(recipe_id)
            recipe_name = recipe.get("name", "未知菜品")
            incident = [
                rel for rel in subgraph.relationships
                if recipe_id in {rel.get("start_id", ""), rel.get("end_id", "")}
            ]
            relation_evidence: List[str] = []
            for rel in incident:
                start = nodes_by_id.get(rel.get("start_id", ""), {})
                end = nodes_by_id.get(rel.get("end_id", ""), {})
                relation_evidence.append(
                    f"{start.get('name', '未知')} -[{rel.get('type', 'RELATED')}]-> "
                    f"{end.get('name', '未知')}"
                )
            chain_evidence = [chain for chain in reasoning_chains if recipe_name in chain]
            evidence = list(dict.fromkeys(relation_evidence + chain_evidence))[:12]
            if not evidence:
                evidence = [self._build_subgraph_description(subgraph)]
            documents.append(Document(
                page_content=f"菜品: {recipe_name}\n图谱证据:\n- " + "\n- ".join(evidence),
                metadata={
                    "search_type": "knowledge_subgraph",
                    "document_id": recipe_id,
                    "node_id": recipe_id,
                    "parent_id": recipe_id,
                    "node_count": len(subgraph.connected_nodes),
                    "relationship_count": len(incident),
                    "relation_types": sorted({rel.get("type", "") for rel in incident}),
                    "graph_density": subgraph.graph_metrics.get("density", 0.0),
                    "reasoning_chains": chain_evidence,
                    "relevance_score": 0.9 if recipe_id in central_ids else 0.75,
                    "recipe_name": recipe_name,
                    "evidence": evidence,
                },
            ))

        return documents
    
    def _build_path_description(self, path: GraphPath) -> str:
        """构建路径的自然语言描述"""
        if not path.nodes:
            return "空路径"
            
        desc_parts: List[str] = []
        for i, node in enumerate(path.nodes):
            desc_parts.append(node.get("name", f"节点{i}"))
            if i < len(path.relationships):
                rel = path.relationships[i]
                rel_type = rel.get("type", "相关")
                current_id = node.get("id", "")
                next_id = path.nodes[i + 1].get("id", "") if i + 1 < len(path.nodes) else ""
                if rel.get("start_id") == current_id and rel.get("end_id") == next_id:
                    desc_parts.append(f" -[{rel_type}]-> ")
                elif rel.get("start_id") == next_id and rel.get("end_id") == current_id:
                    desc_parts.append(f" <-[{rel_type}]- ")
                else:
                    desc_parts.append(f" -[{rel_type}]- ")
        return "".join(desc_parts)
    
    def _build_subgraph_description(self, subgraph: KnowledgeSubgraph) -> str:
        """构建子图的自然语言描述"""
        central_names = [node.get("name", "未知") for node in subgraph.central_nodes]
        node_count = len(subgraph.connected_nodes)
        rel_count = len(subgraph.relationships)
        
        related_names = [node.get("name", "未知") for node in subgraph.connected_nodes[:10]]
        return (
            f"关于 {', '.join(central_names)} 的知识网络，包含 {node_count} 个相关概念和 {rel_count} 个关系。"
            f"相关节点：{', '.join(related_names)}。"
        )
    
    def _rank_by_graph_relevance(self, documents: List[Document], query: str) -> List[Document]:
        """基于图结构相关性排序"""
        return sorted(documents, 
                     key=lambda x: x.metadata.get("relevance_score", 0.0), 
                     reverse=True)
    
    def _analyze_query_complexity(self, query: str) -> float:
        """分析查询复杂度"""
        complexity_indicators = ["什么", "如何", "为什么", "哪些", "关系", "影响", "原因"]
        score = sum(1 for indicator in complexity_indicators if indicator in query)
        return min(score / len(complexity_indicators), 1.0)
    
    def _identify_reasoning_patterns(self, subgraph: KnowledgeSubgraph) -> List[str]:
        """识别推理模式"""
        relation_types = {rel.get("type") for rel in subgraph.relationships}
        patterns = []
        if "REQUIRES" in relation_types:
            patterns.extend(["composition", "cooccurrence"])
        if relation_types & {"BELONGS_TO", "BELONGS_TO_CATEGORY"}:
            patterns.append("classification")
        if relation_types & {"CONTAINS_STEP", "NEXT_STEP"}:
            patterns.append("process")
        return patterns
    
    def _build_reasoning_chain(self, pattern: str, subgraph: KnowledgeSubgraph) -> List[str]:
        """构建推理链"""
        nodes = {
            node.get("id", ""): node
            for node in subgraph.central_nodes + subgraph.connected_nodes
        }
        chains: List[str] = []
        ingredients_by_recipe: Dict[str, List[str]] = defaultdict(list)
        for rel in subgraph.relationships:
            start = nodes.get(rel.get("start_id", ""), {})
            end = nodes.get(rel.get("end_id", ""), {})
            rel_type = rel.get("type")
            start_name = start.get("name", "未知")
            end_name = end.get("name", "未知")
            if rel_type == "REQUIRES":
                ingredients_by_recipe[start.get("id", "")].append(end_name)
            if pattern == "composition" and rel_type == "REQUIRES":
                chains.append(f"{start_name} 需要 {end_name}（REQUIRES）")
            elif pattern == "classification" and rel_type in {"BELONGS_TO", "BELONGS_TO_CATEGORY"}:
                chains.append(f"{start_name} 属于 {end_name}（{rel_type}）")
            elif pattern == "process" and rel_type == "CONTAINS_STEP":
                chains.append(f"{start_name} 包含步骤 {end_name}（CONTAINS_STEP）")
            elif pattern == "process" and rel_type == "NEXT_STEP":
                chains.append(f"{start_name} 之后是 {end_name}（NEXT_STEP）")
        if pattern == "cooccurrence":
            for recipe_id, ingredient_names in ingredients_by_recipe.items():
                recipe_name = nodes.get(recipe_id, {}).get("name", "该菜谱")
                unique_names = list(dict.fromkeys(ingredient_names))
                if len(unique_names) >= 2:
                    chains.append(f"{', '.join(unique_names[:5])} 在 {recipe_name} 中共同出现（菜谱共现）")
        return chains
    
    def _validate_reasoning_chains(self, chains: List[str], query: str) -> List[str]:
        """验证推理链"""
        return list(dict.fromkeys(chain.strip() for chain in chains if chain.strip()))[:10]
    
    def _find_entity_relations(self, graph_query: GraphQuery, session) -> List[GraphPath]:
        """查找实体间关系"""
        if not graph_query.source_entities:
            return []
        relation_types = self._safe_relation_types(graph_query.relation_types)
        max_depth = min(max(int(graph_query.max_depth or 1), 1), 4)
        targets = graph_query.target_entities or []
        if targets:
            query = f"""
            UNWIND $sources AS source_name
            UNWIND $targets AS target_name
            MATCH (source), (target)
            WHERE (source.nodeId=source_name OR
                   (source.name IS NOT NULL AND (toString(source.name) CONTAINS source_name OR source_name CONTAINS toString(source.name))))
              AND (target.nodeId=target_name OR
                   (target.name IS NOT NULL AND (toString(target.name) CONTAINS target_name OR target_name CONTAINS toString(target.name))))
              AND source <> target
            MATCH path=(source)-[*1..{max_depth}]-(target)
            WHERE ALL(rel IN relationships(path) WHERE type(rel) IN $relation_types)
            RETURN path, length(path) AS path_len
            LIMIT $limit
            """
        else:
            query = """
            UNWIND $sources AS source_name
            MATCH (source)
            WHERE source.nodeId=source_name OR
                  (source.name IS NOT NULL AND (toString(source.name) CONTAINS source_name OR source_name CONTAINS toString(source.name)))
            MATCH path=(source)-[rel]-(target)
            WHERE type(rel) IN $relation_types
            RETURN path, 1 AS path_len
            LIMIT $limit
            """
        result = session.run(query, {
            "sources": graph_query.source_entities,
            "targets": targets,
            "relation_types": relation_types,
            "limit": min(max(graph_query.max_nodes, 1), 100),
        })
        paths = []
        for record in result:
            parsed = self._parse_neo4j_path(record, "entity_relation")
            if parsed:
                parsed.relevance_score = self._score_path(parsed, graph_query)
                paths.append(parsed)
        return paths
    
    def _find_shortest_paths(self, graph_query: GraphQuery, session) -> List[GraphPath]:
        """查找最短路径"""
        if not graph_query.source_entities or not graph_query.target_entities:
            logger.warning("最短路径查询必须同时提供源实体和目标实体")
            return []
        max_depth = min(max(int(graph_query.max_depth or 2), 1), 4)
        query = f"""
        UNWIND $sources AS source_name
        UNWIND $targets AS target_name
        MATCH (source), (target)
        WHERE (source.nodeId=source_name OR
               (source.name IS NOT NULL AND (toString(source.name) CONTAINS source_name OR source_name CONTAINS toString(source.name))))
          AND (target.nodeId=target_name OR
               (target.name IS NOT NULL AND (toString(target.name) CONTAINS target_name OR target_name CONTAINS toString(target.name))))
          AND source <> target
        MATCH path=allShortestPaths((source)-[*1..{max_depth}]-(target))
        WHERE ALL(rel IN relationships(path) WHERE type(rel) IN $relation_types)
        RETURN path, length(path) AS path_len
        LIMIT $limit
        """
        result = session.run(query, {
            "sources": graph_query.source_entities,
            "targets": graph_query.target_entities,
            "relation_types": self._safe_relation_types(graph_query.relation_types),
            "limit": min(max(graph_query.max_nodes, 1), 100),
        })
        paths = []
        for record in result:
            parsed = self._parse_neo4j_path(record, "shortest_path")
            if parsed:
                parsed.relevance_score = self._score_path(parsed, graph_query)
                paths.append(parsed)
        return paths
    
    def _fallback_subgraph_extraction(self, graph_query: GraphQuery) -> KnowledgeSubgraph:
        """降级子图提取"""
        if not self.driver or not graph_query.source_entities:
            return KnowledgeSubgraph([], [], [], {}, [])
        try:
            with self.driver.session(database=getattr(self.config, "neo4j_database", "neo4j")) as session:
                record = session.run(
                    """
                    UNWIND $sources AS source_name
                    MATCH (source)
                    WHERE source.nodeId=source_name OR
                          (source.name IS NOT NULL AND toString(source.name) CONTAINS source_name)
                    MATCH path=(source)-[rel]-(neighbor)
                    WHERE type(rel) IN $relation_types
                    RETURN source, collect(path)[0..20] AS paths
                    LIMIT 1
                    """,
                    sources=graph_query.source_entities,
                    relation_types=self._safe_relation_types(graph_query.relation_types),
                ).single()
                if record:
                    return self._build_knowledge_subgraph(record)
        except Exception as exc:
            logger.error(f"一跳子图降级查询失败: {exc}")
        return KnowledgeSubgraph([], [], [], {}, [])

    @staticmethod
    def _path_signature(path: GraphPath) -> str:
        parts = []
        for index, node in enumerate(path.nodes):
            parts.append(str(node.get("id", node.get("name", ""))))
            if index < len(path.relationships):
                parts.append(str(path.relationships[index].get("type", "RELATED")))
        return "|".join(parts)

    def _score_path(self, path: GraphPath, graph_query: GraphQuery) -> float:
        score = 1.0 / (1.0 + max(1, path.path_length))
        target_keywords = graph_query.target_entities or []
        if target_keywords and path.nodes:
            endpoint = path.nodes[-1]
            endpoint_text = f"{endpoint.get('name', '')} {endpoint.get('properties', {}).get('category', '')}"
            if any(keyword in endpoint_text or endpoint.get("name", "") in keyword for keyword in target_keywords):
                score += 0.35
        requested = set(graph_query.relation_types or []) & ALLOWED_RELATION_TYPES
        actual = {rel.get("type") for rel in path.relationships}
        if requested and actual & requested:
            score += 0.15
        return min(score, 1.0)
    
    def close(self):
        """关闭资源连接"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            logger.info("图RAG检索系统已关闭") 
