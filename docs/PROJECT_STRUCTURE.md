# Project Structure

This document defines the current architecture boundaries so new files do not
drift back into a flat or mixed layout.

## Runtime Layout

```text
backend/
  api/
    main.py                 FastAPI app, endpoint schemas, route handlers
  services/
    catalog.py              Product catalog loading/search/serialization
    ai_service.py           AI answer orchestration
    advisor.py              Deterministic catalog advisor
    conversation.py         Conversation state and planning
    decision_engine.py      Decision packets and sales reasoning helpers
    grounded_rag.py         RAG pipeline glue
    observability.py        Metrics, timing, shadow routing
    policy_service.py       Policy knowledge base loading
    value_engine.py         Ranking and value scoring
  harness/
    runtime.py              Harness run state, evidence, budget enforcement
    types.py                Shared harness dataclasses/types
    advisor.py              Advisor harness session orchestration
    context.py              Context lifecycle and compacting
    governance.py           Recovery/governance policy
    preflight.py            Pre-execution checks
    postflight.py           Post-execution verification
    trace.py                Trace collection and budget tracking
    skills.py               Skill registry
    *_*.py                  Focused harness controls
  agent/
    *.py                    Domain contract, query frame, verifier, tools
```

The frontend runtime remains under `frontend/`, data under `data/`, ingestion
and maintenance scripts under `scripts/`, tests under `tests/`, evaluation
scripts under `evals/`, and generated documents/output under `output/`.

## Root Folder Ownership

```text
archive/                Legacy prototypes and historical docs, not runtime
backend/                FastAPI runtime and advisor/harness backend
chroma_db/              Generated vector store, kept at root for retriever defaults
data/                   Product catalog, images, policies, benchmarks
docs/                   Current project documentation
evals/                  Agent/harness evaluation scripts
frontend/               Current React/Vite storefront
logs/                   Local logs/debug output
output/                 Generated reports and deliverables
scripts/                Data ingestion, crawling, OCR, validation utilities
tests/                  Automated test suite
tools/                  Developer-only utilities, scratch scripts, examples
```

Legacy or exploratory code should go under `archive/legacy/` or `tools/`, not
beside production folders at the project root.

## Workflow Packages

The older root-level workflow packages have been folded into `backend/`:

```text
backend/workflows/research_agent/   Former sales research agent package
backend/retrieval/                  Former retriever package plus RAG pipeline
backend/verification/               Former verification workflow package
```

New code should import these packages through the backend namespace:

```python
from backend.workflows.research_agent.sales_research_agent import SalesResearchAgent
from backend.retrieval.pipeline import RAGPipeline
from backend.verification.workflow.workflow import VerificationWorkflow
```

## Import Rules

- API code imports business logic from `backend.services.*` and harness logic
  from `backend.harness.*`.
- Harness code may import services and shared agent/domain types, but route
  handlers should stay in `backend.api`.
- Agent contract code stays in `backend.agent.*`; do not place agent verifier,
  query-frame, or response-composer code in `backend.api`.
- New runtime modules should not be added directly under `backend/`.
- Files directly under `backend/` are compatibility shims for older import
  paths such as `backend.catalog` or `backend.main`.

## Entrypoints

Use the new backend entrypoint for development and deployment:

```bash
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The old entrypoint still works:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

It is kept only for compatibility with older scripts and tests.

## Data Path Rule

Modules inside `backend/services/` resolve the project root using
`Path(__file__).resolve().parents[2]`. This keeps catalog and policy files
pointing to:

```text
data/product_catalog_real.csv
data/product_catalog_clean.csv
data/product_images/
data/Policies/
```

When moving a module deeper or shallower, re-check any `Path(__file__)` based
path calculation before running tests.

## Verification Checklist After Moving Files

Run these checks after architecture changes:

```bash
python -m pytest tests/test_harness_runtime.py tests/test_agent_verifier.py tests/test_query_frame_display_specs.py tests/test_product_reference_resolution.py -q
cd frontend
npm run build
```

For backend smoke testing:

```bash
python - <<'PY'
from backend.api.main import app
from backend.services.catalog import get_catalog
print(app.title)
print(len(get_catalog().products))
PY
```
