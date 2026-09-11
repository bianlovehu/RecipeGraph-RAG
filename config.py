"""
基于图数据库的RAG系统配置文件
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

from dotenv import load_dotenv

@dataclass
class GraphRAGConfig:
    """基于图数据库的RAG系统配置类"""

    # Neo4j数据库配置
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "all-in-rag"
    neo4j_database: str = "neo4j"

    # Milvus配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "cooking_knowledge"
    milvus_dimension: int = 512  # BGE-small-zh-v1.5的向量维度

    # 模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = "deepseek-v4-flash"
    deepseek_base_url: str = "https://api.deepseek.com"

    # 检索配置（LightRAG Round-robin策略）
    top_k: int = 5
    rrf_k: int = 60
    retrieval_candidate_multiplier: int = 3
    rrf_bm25_weight: float = 1.0
    rrf_vector_weight: float = 1.0
    rrf_entity_topic_weight: float = 0.05
    rrf_graph_weight: float = 0.75

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    # 图数据处理配置
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_graph_depth: int = 2  # 图遍历最大深度
    max_graph_nodes: int = 50

    # 菜谱解析 Agent 配置
    agent_max_rounds: int = 6
    agent_max_retries: int = 3

    def __post_init__(self):
        """初始化后的处理"""
        self.top_k = max(1, self.top_k)
        self.rrf_k = max(1, self.rrf_k)
        self.retrieval_candidate_multiplier = max(1, self.retrieval_candidate_multiplier)
        self.rrf_bm25_weight = max(0.0, self.rrf_bm25_weight)
        self.rrf_vector_weight = max(0.0, self.rrf_vector_weight)
        self.rrf_entity_topic_weight = max(0.0, self.rrf_entity_topic_weight)
        self.rrf_graph_weight = max(0.0, self.rrf_graph_weight)
        self.max_graph_depth = min(max(1, self.max_graph_depth), 4)
        self.max_graph_nodes = min(max(1, self.max_graph_nodes), 100)
        self.agent_max_rounds = max(1, self.agent_max_rounds)
        self.agent_max_retries = max(1, self.agent_max_retries)

    @classmethod
    def from_env(cls) -> 'GraphRAGConfig':
        """从 .env/环境变量创建配置，未设置的字段使用安全默认值。"""
        load_dotenv()

        def env_int(name: str, default: int) -> int:
            value = os.getenv(name)
            return int(value) if value not in (None, "") else default

        def env_float(name: str, default: float) -> float:
            value = os.getenv(name)
            return float(value) if value not in (None, "") else default

        return cls(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "all-in-rag"),
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
            milvus_host=os.getenv("MILVUS_HOST", "localhost"),
            milvus_port=env_int("MILVUS_PORT", 19530),
            milvus_collection_name=os.getenv("MILVUS_COLLECTION_NAME", "cooking_knowledge"),
            milvus_dimension=env_int("MILVUS_DIMENSION", 512),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
            llm_model=os.getenv("LLM_MODEL", "deepseek-v4-flash"),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            top_k=env_int("TOP_K", 5),
            rrf_k=env_int("RRF_K", 60),
            retrieval_candidate_multiplier=env_int("RETRIEVAL_CANDIDATE_MULTIPLIER", 3),
            rrf_bm25_weight=env_float("RRF_BM25_WEIGHT", 1.0),
            rrf_vector_weight=env_float("RRF_VECTOR_WEIGHT", 1.0),
            rrf_entity_topic_weight=env_float("RRF_ENTITY_TOPIC_WEIGHT", 0.05),
            rrf_graph_weight=env_float("RRF_GRAPH_WEIGHT", 0.75),
            temperature=env_float("TEMPERATURE", 0.1),
            max_tokens=env_int("MAX_TOKENS", 2048),
            chunk_size=env_int("CHUNK_SIZE", 500),
            chunk_overlap=env_int("CHUNK_OVERLAP", 50),
            max_graph_depth=env_int("MAX_GRAPH_DEPTH", 2),
            max_graph_nodes=env_int("MAX_GRAPH_NODES", 50),
            agent_max_rounds=env_int("AGENT_MAX_ROUNDS", 6),
            agent_max_retries=env_int("AGENT_MAX_RETRIES", 3),
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GraphRAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'neo4j_uri': self.neo4j_uri,
            'neo4j_user': self.neo4j_user,
            'neo4j_password': self.neo4j_password,
            'neo4j_database': self.neo4j_database,
            'milvus_host': self.milvus_host,
            'milvus_port': self.milvus_port,
            'milvus_collection_name': self.milvus_collection_name,
            'milvus_dimension': self.milvus_dimension,
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'deepseek_base_url': self.deepseek_base_url,
            'top_k': self.top_k,
            'rrf_k': self.rrf_k,
            'retrieval_candidate_multiplier': self.retrieval_candidate_multiplier,
            'rrf_bm25_weight': self.rrf_bm25_weight,
            'rrf_vector_weight': self.rrf_vector_weight,
            'rrf_entity_topic_weight': self.rrf_entity_topic_weight,
            'rrf_graph_weight': self.rrf_graph_weight,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'max_graph_depth': self.max_graph_depth,
            'max_graph_nodes': self.max_graph_nodes,
            'agent_max_rounds': self.agent_max_rounds,
            'agent_max_retries': self.agent_max_retries
        }

    def rrf_channel_weights(self) -> Dict[str, float]:
        """返回各召回通道的加权 RRF 权重。"""
        return {
            "bm25": self.rrf_bm25_weight,
            "vector": self.rrf_vector_weight,
            "entity_topic": self.rrf_entity_topic_weight,
            "graph": self.rrf_graph_weight,
        }

# 默认配置实例
DEFAULT_CONFIG = GraphRAGConfig.from_env()
