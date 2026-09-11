"""
混合检索模块
基于双层检索范式：实体级 + 主题级检索
结合BM25、图结构和向量检索，使用RRF排序融合
"""

import logging
import re
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from neo4j import GraphDatabase
import jieba
from .graph_indexing import GraphIndexingModule
from .llm_json import request_json_object

logger = logging.getLogger(__name__)


def chinese_tokenize(text: str) -> List[str]:
    """适用于中文菜谱的 BM25 分词，同时保留英文和数字 token。"""
    normalized = re.sub(r"\s+", " ", str(text).lower()).strip()
    return [token.strip() for token in jieba.lcut(normalized) if token.strip() and not token.isspace()]


def document_identity(document: Document) -> str:
    metadata = document.metadata or {}
    recipe_name = str(metadata.get("recipe_name", "")).strip()
    if recipe_name and recipe_name != "未知菜品":
        return recipe_name
    value = metadata.get("parent_id") or metadata.get("node_id")
    if value:
        return str(value)
    return f"content:{hash(document.page_content)}"


def reciprocal_rank_fusion(
    channels: Dict[str, List[Document]],
    top_k: int,
    rrf_k: int = 60,
    channel_weights: Dict[str, float] | None = None,
) -> List[Document]:
    """按名次融合不同量纲的召回结果，同一路同一菜谱只计分一次。"""
    candidates: Dict[str, Dict[str, Any]] = {}
    weights = channel_weights or {}
    for source, documents in channels.items():
        seen_in_source = set()
        for rank, document in enumerate(documents, 1):
            identity = document_identity(document)
            if identity in seen_in_source:
                continue
            seen_in_source.add(identity)
            contribution = max(0.0, float(weights.get(source, 1.0))) / (max(1, rrf_k) + rank)
            if identity not in candidates:
                candidates[identity] = {
                    "document": Document(
                        page_content=document.page_content,
                        metadata=dict(document.metadata or {}),
                    ),
                    "score": 0.0,
                    "ranks": {},
                    "contributions": {},
                    "evidence": [],
                }
            item = candidates[identity]
            item["score"] += contribution
            item["ranks"][source] = rank
            item["contributions"][source] = contribution
            evidence = (document.metadata or {}).get("evidence")
            if isinstance(evidence, list):
                item["evidence"].extend(str(value) for value in evidence if value)
            elif evidence:
                item["evidence"].append(str(evidence))
            elif document.page_content:
                item["evidence"].append(document.page_content[:300])

    ranked = sorted(
        candidates.items(),
        key=lambda pair: (-pair[1]["score"], min(pair[1]["ranks"].values()), pair[0]),
    )
    results: List[Document] = []
    for identity, item in ranked[:top_k]:
        document = item["document"]
        sources = sorted(item["ranks"], key=item["ranks"].get)
        document.metadata.update({
            "document_id": identity,
            "retrieval_sources": sources,
            "source_ranks": item["ranks"],
            "source_contributions": item["contributions"],
            "rrf_score": item["score"],
            "final_score": item["score"],
            "search_type": "rrf",
            "evidence": list(dict.fromkeys(item["evidence"])),
        })
        results.append(document)
    return results

@dataclass
class RetrievalResult:
    """检索结果数据结构"""
    content: str
    node_id: str
    node_type: str
    relevance_score: float
    retrieval_level: str  # 'low' or 'high'
    metadata: Dict[str, Any]

class HybridRetrievalModule:
    """
    混合检索模块
    核心特点：
    1. 双层检索范式（实体级 + 主题级）
    2. 关键词提取和匹配
    3. 图结构+向量检索结合
    4. 一跳邻居扩展
    5. RRF多路排序融合
    """
    
    def __init__(self, config, milvus_module, data_module, llm_client):
        self.config = config
        self.milvus_module = milvus_module
        self.data_module = data_module
        self.llm_client = llm_client
        self.driver = None
        self.bm25_retriever = None
        
        # 图索引模块
        self.graph_indexing = GraphIndexingModule(config, llm_client)
        self.graph_indexed = False
        
    def initialize(self, chunks: List[Document]):
        """初始化检索系统"""
        logger.info("初始化混合检索模块...")
        
        # 连接Neo4j
        self.driver = GraphDatabase.driver(
            self.config.neo4j_uri, 
            auth=(self.config.neo4j_user, self.config.neo4j_password)
        )
        
        # 初始化BM25检索器
        if chunks:
            self.bm25_retriever = BM25Retriever.from_documents(
                chunks,
                preprocess_func=chinese_tokenize,
            )
            logger.info(f"BM25检索器初始化完成，文档数量: {len(chunks)}")
        
        # 初始化图索引
        self._build_graph_index()
        
    def _build_graph_index(self):
        """构建图索引"""
        if self.graph_indexed:
            return
            
        logger.info("开始构建图索引...")
        
        try:
            # 获取图数据
            recipes = self.data_module.recipes
            ingredients = self.data_module.ingredients
            cooking_steps = self.data_module.cooking_steps
            
            # 创建实体键值对
            self.graph_indexing.create_entity_key_values(recipes, ingredients, cooking_steps)
            
            # 创建关系键值对（这里需要从Neo4j获取关系数据）
            relationships = self._extract_relationships_from_graph()
            self.graph_indexing.create_relation_key_values(relationships)
            
            # 去重优化
            self.graph_indexing.deduplicate_entities_and_relations()
            
            self.graph_indexed = True
            stats = self.graph_indexing.get_statistics()
            logger.info(f"图索引构建完成: {stats}")
            
        except Exception as e:
            logger.error(f"构建图索引失败: {e}")
            
    def _extract_relationships_from_graph(self) -> List[Tuple[str, str, str]]:
        """从Neo4j图中提取关系"""
        relationships = []
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (source)-[r]->(target)
                WHERE source.nodeId >= '200000000' OR target.nodeId >= '200000000'
                RETURN source.nodeId as source_id, type(r) as relation_type, target.nodeId as target_id
                LIMIT 1000
                """
                result = session.run(query)
                
                for record in result:
                    relationships.append((
                        record["source_id"],
                        record["relation_type"],
                        record["target_id"]
                    ))
                    
        except Exception as e:
            logger.error(f"提取图关系失败: {e}")
            
        return relationships
            
    def extract_query_keywords(self, query: str) -> Tuple[List[str], List[str]]:
        """
        提取查询关键词：实体级 + 主题级
        """
        prompt = f"""
        作为烹饪知识助手，请分析以下查询并提取关键词，分为两个层次：

        查询：{query}

        提取规则：
        1. 实体级关键词：具体的食材、菜品名称、工具、品牌等有形实体
           - 例如：鸡胸肉、西兰花、红烧肉、平底锅、老干妈
           - 对于抽象查询，推测相关的具体食材/菜品

        2. 主题级关键词：抽象概念、烹饪主题、饮食风格、营养特点等
           - 例如：减肥、低热量、川菜、素食、下饭菜、快手菜
           - 排除动作词：推荐、介绍、制作、怎么做等

        示例：
        查询："推荐几个减肥菜" 
        {{
            "entity_keywords": ["鸡胸肉", "西兰花", "水煮蛋", "胡萝卜", "黄瓜"],
            "topic_keywords": ["减肥", "低热量", "高蛋白", "低脂"]
        }}

        查询："川菜有什么特色"
        {{
            "entity_keywords": ["麻婆豆腐", "宫保鸡丁", "水煮鱼", "辣椒", "花椒"],
            "topic_keywords": ["川菜", "麻辣", "香辣", "下饭菜"]
        }}

        请严格按照JSON格式返回，不要包含多余的文字：
        {{
            "entity_keywords": ["实体1", "实体2", ...],
            "topic_keywords": ["主题1", "主题2", ...]
        }}
        """
        
        try:
            result = request_json_object(
                self.llm_client,
                self.config.llm_model,
                prompt,
                max_tokens=1000,
            )
            entity_keywords = result.get("entity_keywords", [])
            topic_keywords = result.get("topic_keywords", [])
            
            logger.info(f"关键词提取完成 - 实体级: {entity_keywords}, 主题级: {topic_keywords}")
            return entity_keywords, topic_keywords
            
        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            # 降级方案：简单的关键词分割
            stopwords = {"哪些", "什么", "怎么", "如何", "推荐", "制作", "需要", "使用", "一道"}
            keywords = [
                token for token in chinese_tokenize(query)
                if len(token) > 1 and token not in stopwords
            ]
            keywords = list(dict.fromkeys(keywords)) or [query]
            return keywords[:5], keywords[:5]
    
    def entity_level_retrieval(self, entity_keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        """
        实体级检索：专注于具体实体和关系
        使用图索引的键值对结构进行检索
        """
        results = []
        
        # 1. 使用图索引进行实体检索
        for keyword in entity_keywords:
            # 检索匹配的实体
            entities = self.graph_indexing.get_entities_by_key(keyword)
            
            for entity in entities:
                # 获取邻居信息
                neighbors = self._get_node_neighbors(entity.metadata["node_id"], max_neighbors=2)
                
                # 构建增强内容
                enhanced_content = entity.value_content
                if neighbors:
                    enhanced_content += f"\n相关信息: {', '.join(neighbors)}"
                
                results.append(RetrievalResult(
                    content=enhanced_content,
                    node_id=entity.metadata["node_id"],
                    node_type=entity.entity_type,
                    relevance_score=0.9,  # 精确匹配得分较高
                    retrieval_level="entity",
                    metadata={
                        "entity_name": entity.entity_name,
                        "entity_type": entity.entity_type,
                        "index_keys": entity.index_keys,
                        "matched_keyword": keyword
                    }
                ))
        
        # 2. 将食材/步骤命中投影到所属菜谱，保证下游 RRF 的候选粒度一致
        results = self._project_entity_hits_to_recipes(results, top_k * 2)

        # 3. 如果图索引结果不足，使用Neo4j进行补充检索
        if len(results) < top_k:
            neo4j_results = self._neo4j_entity_level_search(entity_keywords, top_k - len(results))
            results.extend(neo4j_results)
            
        # 4. 按菜谱去重、排序并返回
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        unique_results = []
        seen_recipe_ids = set()
        for result in results:
            if result.node_id in seen_recipe_ids:
                continue
            seen_recipe_ids.add(result.node_id)
            unique_results.append(result)

        logger.info(f"实体级检索完成，返回 {len(unique_results[:top_k])} 个结果")
        return unique_results[:top_k]
    
    def _neo4j_entity_level_search(self, keywords: List[str], limit: int) -> List[RetrievalResult]:
        """Neo4j补充实体的检索"""
        results = []
        
        try:
            with self.driver.session() as session:
                cypher_query = """
                UNWIND $keywords as keyword
                CALL db.index.fulltext.queryNodes('recipe_fulltext_index', keyword + '*') 
                YIELD node, score
                WHERE node:Recipe
                RETURN 
                    node.nodeId as node_id,
                    node.name as name,
                    node.description as description,
                    labels(node) as labels,
                    score
                ORDER BY score DESC
                LIMIT $limit
                """
                
                result = session.run(cypher_query, {
                    "keywords": keywords,
                    "limit": limit
                })
                
                for record in result:
                    content_parts = []
                    if record["name"]:
                        content_parts.append(f"菜品: {record['name']}")
                    if record["description"]:
                        content_parts.append(f"描述: {record['description']}")
                    
                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=record["node_id"],
                        node_type="Recipe",
                        relevance_score=float(record["score"]) * 0.7,  # 补充检索得分较低
                        retrieval_level="entity",
                        metadata={
                            "name": record["name"],
                            "labels": record["labels"],
                            "source": "neo4j_fallback"
                        }
                    ))
                    
        except Exception as e:
            logger.error(f"Neo4j补充检索失败: {e}")
            
        return results

    def _project_entity_hits_to_recipes(
        self, results: List[RetrievalResult], limit: int
    ) -> List[RetrievalResult]:
        """把 Ingredient/CookingStep 命中沿真实图关系映射为 Recipe 候选。"""
        projected = [result for result in results if result.node_type == "Recipe"]
        non_recipe_by_id = {
            str(result.node_id): result for result in results if result.node_type != "Recipe"
        }
        if not non_recipe_by_id or not self.driver:
            return projected
        try:
            with self.driver.session(database=getattr(self.config, "neo4j_database", "neo4j")) as session:
                rows = session.run(
                    """
                    UNWIND $entity_ids AS entity_id
                    MATCH (recipe:Recipe)-[rel]->(entity {nodeId: entity_id})
                    WHERE type(rel) IN ['REQUIRES', 'CONTAINS_STEP']
                    RETURN entity_id, recipe.nodeId AS recipe_id, recipe.name AS recipe_name,
                           type(rel) AS relation_type
                    LIMIT $limit
                    """,
                    entity_ids=list(non_recipe_by_id),
                    limit=max(1, limit),
                )
                for row in rows:
                    original = non_recipe_by_id.get(str(row["entity_id"]))
                    if original is None or not row["recipe_id"]:
                        continue
                    recipe_name = str(row["recipe_name"] or "未知菜品")
                    projected.append(RetrievalResult(
                        content=f"菜品: {recipe_name}\n图谱证据: {original.content}",
                        node_id=str(row["recipe_id"]),
                        node_type="Recipe",
                        relevance_score=original.relevance_score,
                        retrieval_level=original.retrieval_level,
                        metadata={
                            **original.metadata,
                            "name": recipe_name,
                            "recipe_name": recipe_name,
                            "source": "entity_graph_projection",
                            "source_entity_id": original.node_id,
                            "source_entity_type": original.node_type,
                            "relation_type": str(row["relation_type"]),
                            "evidence": [original.content],
                        },
                    ))
        except Exception as exc:
            logger.error(f"实体命中映射菜谱失败: {exc}")
        return projected
    
    def topic_level_retrieval(self, topic_keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        """
        主题级检索：专注于广泛主题和概念
        使用图索引的关系键值对结构进行主题检索
        """
        results = []
        
        # 1. 使用图索引进行关系/主题检索
        for keyword in topic_keywords:
            # 检索匹配的关系
            relations = self.graph_indexing.get_relations_by_key(keyword)
            
            for relation in relations:
                # 获取相关实体信息
                source_entity = self.graph_indexing.entity_kv_store.get(relation.source_entity)
                target_entity = self.graph_indexing.entity_kv_store.get(relation.target_entity)
                
                if source_entity and target_entity:
                    # 构建丰富的主题内容
                    content_parts = [
                        f"主题: {keyword}",
                        relation.value_content,
                        f"相关菜品: {source_entity.entity_name}",
                        f"相关信息: {target_entity.entity_name}"
                    ]
                    
                    # 添加源实体的详细信息
                    if source_entity.entity_type == "Recipe":
                        newline = '\n'
                        content_parts.append(f"菜品详情: {source_entity.value_content.split(newline)[0]}")
                    
                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=relation.source_entity,  # 以主要实体为ID
                        node_type=source_entity.entity_type,
                        relevance_score=0.95,  # 主题匹配得分
                        retrieval_level="topic",
                        metadata={
                            "relation_id": relation.relation_id,
                            "relation_type": relation.relation_type,
                            "source_name": source_entity.entity_name,
                            "target_name": target_entity.entity_name,
                            "matched_keyword": keyword,
                            "index_keys": relation.index_keys
                        }
                    ))
        
        # 2. 使用实体的分类信息进行主题检索
        for keyword in topic_keywords:
            entities = self.graph_indexing.get_entities_by_key(keyword)
            for entity in entities:
                if entity.entity_type == "Recipe":
                    # 构建分类主题内容
                    content_parts = [
                        f"主题分类: {keyword}",
                        entity.value_content
                    ]
                    
                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=entity.metadata["node_id"],
                        node_type=entity.entity_type,
                        relevance_score=0.85,  # 分类匹配得分
                        retrieval_level="topic",
                        metadata={
                            "entity_name": entity.entity_name,
                            "entity_type": entity.entity_type,
                            "matched_keyword": keyword,
                            "source": "category_match"
                        }
                    ))
        
        # 3. 如果结果不足，使用Neo4j进行补充检索
        if len(results) < top_k:
            neo4j_results = self._neo4j_topic_level_search(topic_keywords, top_k - len(results))
            results.extend(neo4j_results)
            
        # 4. 按相关性排序并返回
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"主题级检索完成，返回 {len(results)} 个结果")
        return results[:top_k]
    
    def _neo4j_topic_level_search(self, keywords: List[str], limit: int) -> List[RetrievalResult]:
        """Neo4j主题级检索补充"""
        results = []
        
        try:
            with self.driver.session() as session:
                cypher_query = """
                UNWIND $keywords as keyword
                MATCH (r:Recipe)
                WHERE r.category CONTAINS keyword 
                   OR r.cuisineType CONTAINS keyword
                   OR r.tags CONTAINS keyword
                WITH r, keyword
                OPTIONAL MATCH (r)-[:REQUIRES]->(i:Ingredient)
                WITH r, keyword, collect(i.name)[0..3] as ingredients
                RETURN 
                    r.nodeId as node_id,
                    r.name as name,
                    r.category as category,
                    r.cuisineType as cuisine_type,
                    r.difficulty as difficulty,
                    ingredients,
                    keyword as matched_keyword
                ORDER BY r.difficulty ASC, r.name
                LIMIT $limit
                """
                
                result = session.run(cypher_query, {
                    "keywords": keywords,
                    "limit": limit
                })
                
                for record in result:
                    content_parts = []
                    content_parts.append(f"菜品: {record['name']}")
                    
                    if record["category"]:
                        content_parts.append(f"分类: {record['category']}")
                    if record["cuisine_type"]:
                        content_parts.append(f"菜系: {record['cuisine_type']}")
                    if record["difficulty"]:
                        content_parts.append(f"难度: {record['difficulty']}")
                    
                    if record["ingredients"]:
                        ingredients_str = ', '.join(record["ingredients"][:3])
                        content_parts.append(f"主要食材: {ingredients_str}")
                    
                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=record["node_id"],
                        node_type="Recipe",
                        relevance_score=0.75,  # 补充检索得分
                        retrieval_level="topic",
                        metadata={
                            "name": record["name"],
                            "category": record["category"],
                            "cuisine_type": record["cuisine_type"],
                            "difficulty": record["difficulty"],
                            "matched_keyword": record["matched_keyword"],
                            "source": "neo4j_fallback"
                        }
                    ))
                    
        except Exception as e:
            logger.error(f"Neo4j主题级检索失败: {e}")
            
        return results
        
    def dual_level_retrieval(self, query: str, top_k: int = 5) -> List[Document]:
        """
        双层检索：结合实体级和主题级检索
        """
        logger.info(f"开始双层检索: {query}")
        
        # 1. 提取关键词
        entity_keywords, topic_keywords = self.extract_query_keywords(query)
        
        # 2. 执行双层检索
        entity_results = self.entity_level_retrieval(entity_keywords, top_k)
        topic_results = self.topic_level_retrieval(topic_keywords, top_k)
        
        # 3. 结果合并和排序
        all_results = entity_results + topic_results
        
        # 4. 去重和重排序
        seen_nodes = set()
        unique_results = []
        
        for result in sorted(all_results, key=lambda x: x.relevance_score, reverse=True):
            if result.node_id not in seen_nodes:
                seen_nodes.add(result.node_id)
                unique_results.append(result)
        
        # 5. 转换为Document格式
        documents = []
        for result in unique_results[:top_k]:
            # 确保recipe_name字段正确设置
            recipe_name = result.metadata.get("name") or result.metadata.get("entity_name", "未知菜品")
            
            doc = Document(
                page_content=result.content,
                metadata={
                    "node_id": result.node_id,
                    "node_type": result.node_type,
                    "retrieval_level": result.retrieval_level,
                    "relevance_score": result.relevance_score,
                    "recipe_name": recipe_name,  # 确保有recipe_name字段
                    "search_type": "dual_level",  # 设置搜索类型
                    **result.metadata
                }
            )
            documents.append(doc)
            
        logger.info(f"双层检索完成，返回 {len(documents)} 个文档")
        return documents
    
    def vector_search_enhanced(self, query: str, top_k: int = 5) -> List[Document]:
        """
        增强的向量检索：结合图信息
        """
        try:
            # 使用Milvus进行向量检索
            vector_docs = self.milvus_module.similarity_search(query, k=top_k*2)
            
            # 用图信息增强结果并转换为Document对象
            enhanced_docs = []
            for result in vector_docs:
                # 从Milvus结果创建Document对象
                content = result.get("text", "")
                metadata = result.get("metadata", {})
                node_id = metadata.get("node_id")
                
                if node_id:
                    # 从图中获取邻居信息
                    neighbors = self._get_node_neighbors(node_id)
                    if neighbors:
                        # 将邻居信息添加到内容中
                        neighbor_info = f"\n相关信息: {', '.join(neighbors[:3])}"
                        content += neighbor_info
                
                # 确保recipe_name字段正确设置
                recipe_name = metadata.get("recipe_name", "未知菜品")
                
                # 调试：打印向量得分
                vector_score = result.get("score", 0.0)
                logger.debug(f"向量检索得分: {recipe_name} = {vector_score}")
                
                # 创建Document对象
                doc = Document(
                    page_content=content,
                    metadata={
                        **metadata,
                        "recipe_name": recipe_name,  # 确保有recipe_name字段
                        "score": vector_score,
                        "search_type": "vector_enhanced"
                    }
                )
                enhanced_docs.append(doc)
                
            return enhanced_docs[:top_k]
            
        except Exception as e:
            logger.error(f"增强向量检索失败: {e}")
            return []
    
    def _get_node_neighbors(self, node_id: str, max_neighbors: int = 3) -> List[str]:
        """获取节点的邻居信息"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (n {nodeId: $node_id})-[r]-(neighbor)
                RETURN neighbor.name as name
                LIMIT $limit
                """
                result = session.run(query, {"node_id": node_id, "limit": max_neighbors})
                return [record["name"] for record in result if record["name"]]
        except Exception as e:
            logger.error(f"获取邻居节点失败: {e}")
            return []
    
    def bm25_search(self, query: str, top_k: int = 5) -> List[Document]:
        """执行中文 BM25 检索，并补充统一来源元数据。"""
        if self.bm25_retriever is None:
            logger.warning("BM25检索器尚未初始化")
            return []
        try:
            self.bm25_retriever.k = top_k
            documents = self.bm25_retriever.invoke(query)
            results = []
            for document in documents[:top_k]:
                copied = Document(page_content=document.page_content, metadata=dict(document.metadata or {}))
                copied.metadata["search_type"] = "bm25"
                copied.metadata["recipe_name"] = copied.metadata.get("recipe_name", "未知菜品")
                results.append(copied)
            return results
        except Exception as exc:
            logger.error(f"BM25检索失败: {exc}")
            return []

    def retrieve_channels(self, query: str, top_k: int = 5) -> Dict[str, List[Document]]:
        """返回未融合的召回通道，供路由器加入 GraphRAG 后统一融合。"""
        candidate_k = top_k * getattr(self.config, "retrieval_candidate_multiplier", 3)
        return {
            "bm25": self.bm25_search(query, candidate_k),
            "vector": self.vector_search_enhanced(query, candidate_k),
            "entity_topic": self.dual_level_retrieval(query, candidate_k),
        }

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        混合检索：对 BM25、向量和实体/主题检索执行 RRF。
        """
        logger.info(f"开始混合检索: {query}")
        
        channels = self.retrieve_channels(query, top_k)
        final_docs = reciprocal_rank_fusion(
            channels,
            top_k=top_k,
            rrf_k=getattr(self.config, "rrf_k", 60),
            channel_weights=self.config.rrf_channel_weights(),
        )
        source_count = sum(len(documents) for documents in channels.values())
        logger.info(f"RRF融合：从 {source_count} 个候选融合为 {len(final_docs)} 个文档")
        logger.info(f"混合检索完成，返回 {len(final_docs)} 个文档")
        return final_docs
        
    def close(self):
        """关闭资源连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭") 
