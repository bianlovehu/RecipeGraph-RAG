# RecipeGraph-RAG

一个面向中文菜谱场景的知识图谱增强检索（GraphRAG）智能烹饪助手。项目将 Neo4j 图关系、Milvus 向量检索、BM25 关键词检索和大语言模型结合起来，为菜谱推荐、食材查询、烹饪问答和多跳关系推理提供可追溯的回答。

## 项目亮点

- **多路混合检索**：融合 BM25、向量、实体主题和图关系通道，并通过加权 RRF 完成结果排序。
- **智能查询路由**：根据问题类型与复杂度，在传统混合检索、GraphRAG 和组合策略之间自动选择。
- **图谱关系推理**：支持食材、菜谱、烹饪步骤等实体之间的多跳遍历和证据提取。
- **菜谱解析 Agent**：将 Markdown 菜谱解析为结构化数据，完成用量标准化、证据校验并生成可审查产物。
- **可视化 Web 应用**：FastAPI 同时提供聊天 API、推荐接口和静态前端页面。
- **检索效果评测**：内置样例集，可比较纯向量、BM25 + 向量和完整多路检索的 Recall@5、MRR@5 与延迟。

## 系统架构

```text
用户问题
   │
   ▼
智能查询路由器
   ├── BM25 关键词检索 ─┐
   ├── Milvus 向量检索 ─┼── 加权 RRF ── 候选菜谱
   ├── 实体/主题召回 ───┤
   └── Neo4j 图谱检索 ──┘
                              │
                              ▼
                       大语言模型生成回答
                              │
                              ▼
                    FastAPI + Web 可视化界面
```

## 技术栈

- Python 3.10+
- FastAPI / Uvicorn
- Neo4j
- Milvus / etcd / MinIO
- Sentence Transformers（`BAAI/bge-small-zh-v1.5`）
- DeepSeek API（兼容 OpenAI SDK）
- LangChain、jieba、rank-bm25
- HTML / CSS / JavaScript

## 目录结构

```text
.
├── api/                 # FastAPI Web 服务
├── frontend/            # 可视化聊天前端
├── rag_modules/         # 检索、路由、图谱与生成模块
├── recipe_agent/        # 菜谱解析 Agent
├── scripts/             # 数据解析、图片恢复与检索评测脚本
├── tests/               # 自动化测试
├── evaluation/          # 检索评测样例
├── data/                # 菜谱、技巧和 Neo4j 导入数据
├── docker-compose.yml   # Milvus 服务编排
├── requirements.txt     # Python 依赖
└── start_all.bat        # Windows 一键启动脚本
```

## 快速开始

### 1. 准备环境

请先安装：

- Python 3.10 或更高版本
- Docker Desktop（支持 Docker Compose）

克隆项目并进入目录：

```bash
git clone https://github.com/bianlovehu/RecipeGraph-RAG.git
cd RecipeGraph-RAG
```

创建虚拟环境并安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 配置环境变量

复制配置模板：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`，至少填写自己的 `DEEPSEEK_API_KEY`。不要将 `.env` 提交到版本库。

### 3. 启动数据库

在项目根目录启动 Milvus：

```powershell
docker compose up -d
```

再启动 Neo4j：

```powershell
docker compose -f data/docker-compose.yml up -d
```

### 4. 启动应用

Windows 用户可运行：

```powershell
.\start_all.bat
```

也可以手动启动 Web 服务：

```powershell
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

启动完成后访问：

- Web 应用：<http://127.0.0.1:8000>
- 健康检查：<http://127.0.0.1:8000/api/health>
- Neo4j Browser：<http://127.0.0.1:7474>
- MinIO Console：<http://127.0.0.1:9001>

## 主要 API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/home` | 获取首页内容 |
| `GET` | `/api/recommendations` | 分页获取菜谱推荐 |
| `POST` | `/api/chat` | 提交烹饪问题并获取 GraphRAG 回答 |
| `GET` | `/api/recipe-image/{recipe_key}` | 获取菜谱图片 |
| `GET` | `/api/health` | 查看服务及依赖状态 |

## 菜谱解析 Agent

默认只生成审查产物，不写入 Neo4j：

```powershell
python scripts/parse_recipes.py --input data/dishes --output artifacts/recipe_agent --limit 3
```

确认产物后使用 `--apply` 增量写入图数据库：

```powershell
python scripts/parse_recipes.py --file "data/dishes/aquatic/红烧鱼.md" --resume --apply
```

## 检索评测

确保 Neo4j、Milvus 和 DeepSeek API 可用后运行：

```powershell
python scripts/evaluate_retrieval.py
```

评测报告生成在 `artifacts/evaluation/`。项目当前的 12 个样例中，完整多路检索的 Recall@5 为 `0.9514`；该数值仅代表随仓库附带样例的本地测试结果。

## 运行测试

```powershell
python -m unittest discover -s tests -v
```

## 数据与安全说明

- 仓库包含中文菜谱 Markdown 和配套图片；使用数据前请确认其来源与授权满足你的场景。
- `.env`、数据库运行卷、缓存和日志已通过 `.gitignore` 排除。
- 示例数据库密码只适合本地开发，部署前请修改密码并限制端口访问。

## 后续方向

- 增加更大规模、分层覆盖的离线评测集
- 为回答补充更细粒度的图谱证据展示
- 增加容器化 API 服务与生产环境部署配置
- 完善食材替换、忌口过滤和营养信息推理

