# AURA AI Sales Advisor

A catalog-grounded sales advisor for laptops and phones. The React storefront
talks to a FastAPI gateway that provides product search, grounded chat,
comparison, decision traces for development, and deterministic fallbacks when
external AI services are unavailable.

## Local development

Requirements: Python 3.11+, Node.js 20+, and optionally Docker Desktop.

```bash
Copy-Item .env.example .env       # Windows PowerShell
python -m pip install -r requirements.txt
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

In a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open <http://localhost:5173>. The local Vite server proxies API requests to
`http://127.0.0.1:8000` by default. `start.bat` and `start.sh` provide the
same local startup flow.

The default deterministic catalog advisor works without external API keys.
Enable external research or LLM wording only after setting the corresponding
variables in `.env`; see [`.env.example`](.env.example).

## Deployment

The frontend can be deployed to Vercel and the FastAPI service to Render (or
another Python host). Set this Vercel environment variable for every deployed
frontend environment:

```text
VITE_API_BASE_URL=https://internship-eewx.onrender.com
```

Vite exposes only variables with the `VITE_` prefix to browser code. Do not
put API keys or bearer tokens in Vercel frontend variables.

For a local Docker development stack:

```bash
docker compose up --build
```

The current Compose file runs the Vite development server and FastAPI gateway;
it is not a hardened production image. See
[docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) for the deployment
checklist.

## Quality checks

```bash
python -m compileall -q backend
cd frontend && npm ci && npm run lint && npm run build
```

GitHub Actions runs the backend syntax check and the frontend lint/build on
pull requests and pushes to `main`. The full pytest suite remains an explicit
local verification command because it contains long-running integration and
runtime-contract coverage:

```bash
python -m pytest -q
```

## Architecture

This is a modular monolith, intentionally organized by runtime boundary rather
than deployed as independent microservices:

```text
backend/
  api/             FastAPI schemas and route handlers
  services/        catalog, AI orchestration, policy and ranking logic
  harness/         execution budgets, trace, preflight and postflight guards
  agent/           domain contract, query framing and response verification
  retrieval/       RAG and hybrid retrieval components
  verification/    verification workflow and support utilities
  workflows/       research-agent workflow
frontend/          React/Vite storefront
data/              catalog, policies and product images
scripts/           ingestion and catalog maintenance utilities
tests/             automated tests
```

API routes stay in `backend.api`; business logic stays in `backend.services`;
new runtime modules must not be added directly under `backend/`. Compatibility
shims at that level exist only for legacy imports.

More detail: [project structure](docs/PROJECT_STRUCTURE.md).

## API endpoints

- `GET /health` — service and catalog status
- `GET /api/products` — catalog search and filters
- `GET /api/products/featured` — storefront cards
- `POST /api/chat` — grounded advisor interaction
- `GET /metrics` — development metrics (do not expose publicly without an
  access-control layer)

## Repository hygiene

- `.env`, local databases, logs, generated frontend output, dependency caches,
  and scratch clones are ignored.
- Source, tests, Docker configuration, and Markdown documentation remain
  trackable.
- The `archive/` and `tools/scratch/` directories are deliberately excluded
  from the production source tree.
