# Deployment guide

This project is a React/Vite frontend plus a FastAPI backend. Deploy each
runtime independently; do not send Python secrets to the browser build.

## Backend

Deploy the repository with Python 3.11 or newer. The runtime command is:

```bash
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port $PORT
```

On Render, use `$PORT` supplied by the platform. The health-check path is
`/health`.

Set only the backend variables that the selected behavior needs. The safe
default keeps `ENABLE_EXTERNAL_AI_WORKFLOW=false`; then the catalog-backed
deterministic advisor does not need third-party API keys. Configure
`GOOGLE_API_KEY`, `TAVILY_API_KEY`, and `LLAMA_CLOUD_API_KEY` only when the
matching external workflow is enabled. Never commit `.env`.

Before deploying, run:

```bash
python -m compileall -q backend
python -B -c "from backend.api.main import app; print(len(app.routes))"
```

## Frontend on Vercel

Set the following environment variable in each Vercel environment that should
reach the production backend:

```text
VITE_API_BASE_URL=https://internship-eewx.onrender.com
```

The value must be the absolute backend origin, without a trailing slash. It is
embedded in the generated client bundle, so it must never contain a secret.

Vercel build settings:

```text
Root Directory: frontend
Install Command: npm ci
Build Command: npm run build
Output Directory: dist
```

Validate the frontend before deployment:

```bash
cd frontend
npm ci
npm run lint
npm run build
```

## Cross-origin access

The FastAPI application currently permits local Vite development origins only.
Before making a public Vercel deployment, add the exact Vercel production and
preview origins to the backend CORS allow-list, then verify browser requests
from those origins. Avoid wildcard origins when credentials or bearer-token
authentication is introduced.

## Docker

`docker-compose.yml` is a local development convenience. It starts the Vite
development server and mounts source code; it is not the production deployment
path. Validate its syntax with:

```bash
docker compose config --quiet
```

## Release checklist

1. Confirm the backend entrypoint is `backend.api.main:app`.
2. Run backend compile and frontend lint/build checks.
3. Set `VITE_API_BASE_URL` in Vercel; do not add backend secrets there.
4. Configure exact production CORS origins on the backend.
5. Check `GET /health` and a browser request to `/api/products` after release.
6. Restrict or remove public `/metrics` before exposing the backend publicly.
