# Sale Agent

An AI-powered sales assistant that handles customer objections in real time. The system combines a **Hybrid RAG pipeline** for product and policy retrieval with a **LangGraph verification workflow** that automatically checks, corrects, and approves agent responses before they reach the customer.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Running Tests](#running-tests)
- [Docker](#docker)
- [Contributing](#contributing)

---

## Overview

| Component | Description |
|---|---|
| **RAG Pipeline** | Routes customer queries through a relevance checker, then retrieves context via hybrid BM25 + vector search |
| **Sales Research Agent** | LlamaIndex ReAct agent that uses internal product DB and Tavily web search to draft responses |
| **Verification Agent** | LangGraph workflow that validates price accuracy, policy authenticity, and response relevance |
| **Self-Correction Loop** | Automatically retries with structured feedback when verification fails |

---

## Architecture

```
Customer Objection
       │
       ▼
┌─────────────────┐
│  Relevance      │  NO_MATCH → default response
│  Checker        │
└────────┬────────┘
         │ CAN_ANSWER / PARTIAL
         ▼
┌─────────────────┐
│  Hybrid         │  BM25 + ChromaDB vector search
│  Retriever      │
└────────┬────────┘
         │ retrieved context
         ▼
┌─────────────────┐
│  Sales Research │  ReAct agent (internal DB + Tavily)
│  Agent          │
└────────┬────────┘
         │ draft response
         ▼
┌─────────────────────────────────────┐
│  Verification Workflow (LangGraph)  │
│                                     │
│  ┌──────────┐   ┌────────────────┐  │
│  │ Checkers │──▶│ Binary Decision│  │
│  │ - Price  │   │ PASS / FAIL    │  │
│  │ - Policy │   └───────┬────────┘  │
│  │ - Relevance          │           │
│  └──────────┘    PASS ──┤── FAIL    │
│                         │      │    │
│                    APPROVED  Self-  │
│                           Correction│
│                           Loop (max │
│                           retries)  │
└─────────────────────────────────────┘
         │
         ▼
  Final Response
```

---

## Project Structure

```
sale-agent/
├── agent/                          # Sales Research Agent
│   ├── sales_research_agent.py     # ReAct agent + AgentResult dataclass
│   ├── tools.py                    # Internal DB tool + Tavily tool builders
│   ├── prompts.py                  # System prompt + correction context builder
│   └── cache.py                    # Response caching
│
├── retriever/                      # Hybrid Retrieval
│   ├── hybrid_retriever.py         # BM25 + ChromaDB fusion retriever
│   └── relevance_checker.py        # Query relevance classifier
│
├── verification/                   # Verification Workflow
│   ├── agent/
│   │   ├── verification_agent.py   # LangGraph verification agent
│   │   └── checkers.py             # Price, policy, relevance checkers
│   ├── workflow/
│   │   ├── workflow.py             # LangGraph StateGraph definition
│   │   ├── routing.py              # Conditional edge routing logic
│   │   ├── correction.py           # Self-correction node
│   │   └── persistence.py          # Workflow state persistence
│   ├── models/
│   │   ├── state.py                # WorkflowState TypedDict
│   │   ├── verification.py         # VerificationResult + issue models
│   │   └── execution.py            # Execution tracking models
│   ├── config/                     # YAML configs + loaders
│   ├── utils/                      # Shared utilities (cache, logging, metrics, etc.)
│   └── api.py                      # FastAPI health + query endpoints
│
├── tests/                          # All tests (mirrors source structure)
│   ├── agent/                      # Agent unit tests
│   ├── verification/               # Verification module tests
│   ├── test_unit.py                # Core unit tests
│   ├── test_integration.py         # Integration tests
│   ├── test_e2e_workflow.py        # End-to-end workflow tests
│   └── test_pbt.py                 # Property-based tests
│
├── scripts/                        # One-off data & maintenance scripts
│   ├── data_cleaning.py            # Raw catalog cleaning
│   └── ingestion_pipeline.py       # Data ingestion into ChromaDB
│
├── examples/                       # Standalone usage examples
│   ├── execution_tracking_example.py
│   ├── logging_example.py
│   ├── config_example_usage.py
│   └── thresholds_example.py
│
├── data/
│   ├── product_catalog_clean.csv   # Cleaned product catalog
│   └── Policies/                   # Warranty & return policy PDFs
│
├── docker/                         # Container configuration
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── start.sh
│   └── healthcheck.sh
│
├── rag_pipeline.py                 # RAGPipeline orchestrator
├── pytest.ini
└── requirements.txt
```

---

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) or pip
- Docker & Docker Compose (for containerized deployment)
- API keys for:
  - Google Gemini (`GOOGLE_API_KEY`)
  - Tavily Search (`TAVILY_API_KEY`)
  - LlamaCloud (`LLAMA_CLOUD_API_KEY`)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/mindu2kk/Sale_Agent.git
cd Sale_Agent

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Configuration

Copy the example env file and fill in your API keys:

```bash
cp .env.example .env
```

```dotenv
# .env
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key
```

Verification thresholds and workflow behaviour are controlled via YAML files in `verification/config/`:

| File | Purpose |
|---|---|
| `thresholds.yaml` | Confidence thresholds for price / policy / relevance checks |
| `verification_config.yaml` | Max retries, timeout, escalation rules |
| `workflow_config.yaml` | LangGraph node configuration |
| `logging_config.yaml` | Log levels and output format |
| `environments/` | Per-environment overrides (development / test / production) |

---

## Usage

### 1. Ingest data

Load the product catalog and policy PDFs into ChromaDB:

```bash
python ingestion_pipeline.py
```

### 2. Run the API server

```bash
uvicorn verification.api:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/query` | Submit a customer objection |

### 3. Direct Python usage

```python
from retriever.hybrid_retriever import HybridRetriever
from retriever.relevance_checker import RelevanceChecker
from rag_pipeline import RAGPipeline
from agent.sales_research_agent import SalesResearchAgent

retriever = HybridRetriever(...)
checker = RelevanceChecker(...)
rag = RAGPipeline(retriever=retriever, checker=checker)

agent = SalesResearchAgent(llm=llm, rag_pipeline=rag, tavily_api_key="...")
result = agent.run("Sản phẩm này có bảo hành không?")
print(result.draft_response)
```

---

## Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/test_unit.py

# Integration tests
pytest tests/test_integration.py

# Verification module tests
pytest verification/tests/

# With coverage
pytest --cov=. --cov-report=term-missing
```

---

## Docker

```bash
# Build and start all services
docker-compose -f docker/docker-compose.yml up --build

# Run in detached mode
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f
```

The API will be available at `http://localhost:8000`.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

Please follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.
