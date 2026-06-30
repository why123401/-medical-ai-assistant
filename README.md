# 医疗设备知识助手 — RAG 智能客服

> 基于 RAG（检索增强生成）+ 通义千问的医疗设备技术支持系统，支持故障诊断、规格查询、维护计划等专业问答及日常交流。

## 功能特性

- **混合检索（Hybrid Retrieval）**：BM25 关键词检索 + 向量语义检索 + RRF 融合 + Cross-Encoder 重排，兼顾精确匹配与语义理解
- **多模型路由 + 熔断器**：通义千问 qwen-plus / qwen-max / qwen-turbo 三级降级，自动故障切换
- **对话记忆**：基于 SQLite 持久化的多轮对话，支持历史对话列表切换、自动命名
- **闲聊支持**：智能识别问候、闲聊等通用问题，直接由 LLM 自由回答，不强行走知识库检索
- **知识库管理**：支持 TXT / PDF 文档上传，自动分片、向量化、去重入库
- **评测框架**：内置 faithfulness / relevancy / precision / recall 等 RAG 指标评估

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│              Streamlit Frontend (:8501)                      │
│        对话界面 + 历史对话列表 + 知识库上传 + 参考资料展示      │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP REST
┌──────────────────────────▼──────────────────────────────────┐
│                  FastAPI Backend (:8000)                     │
│  ┌────────────────┐  ┌──────────────────┐  ┌─────────────┐  │
│  │ Conversations  │  │  Knowledge Base   │  │ Health Check│  │
│  │ CRUD + Chat    │  │  Upload / List    │  │             │  │
│  └───────┬────────┘  └───────┬──────────┘  └──────┬──────┘  │
│          │                   │                     │          │
│  ┌───────▼───────────────────▼─────────────────────▼──────┐  │
│  │              SQLAlchemy (SQLite)                        │  │
│  │         conversations / messages / knowledge_docs       │  │
│  └────────────────────────┬───────────────────────────────┘  │
└───────────────────────────┼──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│                     AI Core Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ RAG Pipeline │  │ Model Router │  │ Medical Agent    │   │
│  │ BM25+Vector  │  │ Circuit      │  │ Intent Routing   │   │
│  │ RRF Fusion   │  │ Breaker      │  │ KG + RAG Tools   │   │
│  │ Reranker     │  │ Fallback     │  │ Chit-chat Detect │   │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘   │
│         │                 │                     │             │
│  ┌──────▼────────────────▼─────────────────────▼──────────┐  │
│  │              ChromaDB Vector Store                      │  │
│  │         medical_devices collection                      │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 前置条件

- Python 3.11+
- 阿里云 DashScope API Key（[注册入口](https://dashscope.console.aliyun.com/)）

### 1. 安装依赖

```bash
pip install -e ".[dev]"
```

或手动安装：

```bash
pip install fastapi uvicorn streamlit langchain langchain-community \
    langchain-chroma chromadb sqlalchemy pydantic-settings dashscope \
    pymupdf pyyaml pytest httpx
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 DASHSCOPE_API_KEY
```

### 3. 启动后端

```bash
uvicorn src.api.app:app --reload --port 8000
```

### 4. 启动前端

```bash
streamlit run app.py --server.port 8501
```

浏览器打开 http://localhost:8501

### 5. 上传知识库文档

在 Streamlit 界面或通过 API 上传 TXT / PDF 文档：

```bash
curl -X POST http://localhost:8000/api/kb/upload \
  -F "file=@data/knowledge/equipment_specs/ventilator_med-vent-x200.txt"
```

### 6. 开始对话

```bash
# 创建对话
curl -X POST http://localhost:8000/api/conversations \
  -H "Content-Type: application/json" \
  -d '{"title": "呼吸机咨询"}'

# 发送消息
curl -X POST http://localhost:8000/api/conversations/{id}/messages \
  -H "Content-Type: application/json" \
  -d '{"message": "MED-VENT-X200 报警 E104 如何处理？"}'
```

## 核心架构详解

### 混合检索（Hybrid Retrieval）

```
查询 → [BM25 关键词] ──┐
                       ├──→ RRF 融合 → Top-20 → 重排 → Top-5 → 生成
查询 → [向量相似度] ───┘
```

| 组件 | 作用 |
|------|------|
| **BM25** | 精确匹配设备型号、故障码（如 `MED-VENT-X200`、`E104`） |
| **向量检索** | 语义理解模糊查询（如 "呼吸机报警怎么解决"） |
| **RRF 融合** | `score(d) = Σ 1/(k + rank_i)`，k=60，无需调参 |
| **Cross-Encoder 重排** | DashScope text-rerank-v2，从 Top-20 精选 Top-5 |

### 模型路由 + 熔断器

```
qwen-plus (主) → 失败 → qwen-max (强降级) → 失败 → qwen-turbo (弱降级)
  │
  └── 连续 5 次失败 → 熔断器打开 → 30s 后半开测试
```

### Agent 意图路由

| 意图类型 | 识别方式 | 处理方式 |
|----------|----------|----------|
| 闲聊问候 | 正则匹配（你好/谢谢/再见/你是谁） | 直接回复，跳过 RAG |
| 故障码查询 | 模式 `E\d+` / `F\d+` / `A\d+` | 知识图谱精确查找 |
| 设备规格查询 | 模式 `MED-XXX` / `PHIL-XXX` | 知识图谱精确查找 |
| 维护计划查询 | 关键词（维护/保养/校准/定期） | 知识图谱精确查找 |
| 通用问题 | 以上都不匹配 | RAG 知识库检索 + LLM 生成 |

### 对话记忆

- 每轮对话持久化到 SQLite（`conversations` + `messages` 表）
- 前端侧边栏展示历史对话列表，点击即可切换恢复
- 首条消息自动提取中文关键词作为对话标题
- 超过 16 轮触发摘要机制（placeholder）

## 项目结构

```
medical-ai-assistant/
├── src/
│   ├── api/              # FastAPI 后端
│   │   ├── app.py        # 应用入口 + CORS + 生命周期
│   │   ├── routes/       # 路由（对话、知识库）
│   │   └── schemas/      # Pydantic 数据模型
│   ├── ai/               # AI 核心
│   │   ├── agents/       # 医疗 Agent + 工具集
│   │   ├── rag/          # RAG 管道（检索、重排、生成）
│   │   └── routing/      # 模型路由 + 熔断器
│   ├── kb/               # 知识库管理（分片、索引）
│   ├── memory/           # 对话记忆管理
│   ├── eval/             # RAG 评测框架
│   ├── shared/           # 配置、日志、异常
│   └── infra/            # 数据库、ORM 模型
├── tests/
│   ├── unit/             # 单元测试
│   └── integration/      # 集成测试
├── data/
│   ├── knowledge/        # 医疗设备文档（TXT/PDF）
│   │   ├── equipment_specs/
│   │   ├── fault_trees/
│   │   └── maintenance/
│   ├── kg/               # 知识图谱（YAML）
│   └── eval/             # 评测数据集
├── prompts/v1/           # Prompt 模板
├── scripts/              # 数据处理脚本
├── app.py                # Streamlit 前端
├── pyproject.toml        # 项目依赖
├── docker-compose.yml    # Docker 一键部署
├── Dockerfile
└── .env.example          # 环境变量模板
```

## 技术栈

| 组件 | 选型 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| 前端框架 | Streamlit |
| 向量数据库 | ChromaDB |
| LLM | 通义千问 (DashScope) |
| 嵌入模型 | text-embedding-v4 (DashScope) |
| 重排模型 | text-rerank-v2 (DashScope) |
| 数据库 | SQLite (开发) / PostgreSQL (生产) |
| 配置管理 | Pydantic Settings |
| 日志 | 结构化 JSON 日志 |
| 测试 | pytest |
| 部署 | Docker Compose |

## API 文档

启动后端后访问：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 主要端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/conversations` | 创建新对话 |
| GET | `/api/conversations` | 获取对话列表 |
| GET | `/api/conversations/{id}/messages` | 获取对话历史 |
| POST | `/api/conversations/{id}/messages` | 发送消息（核心接口） |
| DELETE | `/api/conversations/{id}` | 删除对话 |
| POST | `/api/kb/upload` | 上传知识库文档 |
| GET | `/api/kb/documents` | 列出知识库文档 |
| DELETE | `/api/kb/documents/{filename}` | 删除文档 |
| GET | `/health` | 健康检查 |

## 评测

```bash
# 运行评测
python scripts/run_eval.py

# 生成黄金数据集
python scripts/generate_golden_set.py
```

## Docker 部署

```bash
docker-compose up --build
```

将启动三个服务：
- `api` — FastAPI 后端 (:8000)
- `streamlit` — 前端界面 (:8501)
- `postgres` — PostgreSQL 数据库 (:5432)

## 已知知识库设备

| 设备代码 | 设备名称 | 类别 |
|----------|----------|------|
| MED-VENT-X200 | Medtronic Ventilator X200 | 呼吸机 |
| PHIL-SV-COMPACT | Philips SV Compact | 呼吸机 |
| MINDRAY-T12 | Mindray BeneView T12 | 监护仪 |
| GE-REV-APEX | GE Revolution Apex CT | CT 扫描仪 |
| PHIL-IE33 | Philips IE33 | 超声诊断仪 |
