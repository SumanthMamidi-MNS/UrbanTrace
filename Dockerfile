# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1: build the web UI against the real API (VITE_USE_MOCK=false).
# ---------------------------------------------------------------------------
FROM node:22-slim AS web-build
WORKDIR /web

COPY web/package.json web/package-lock.json ./
RUN npm ci

COPY web/ ./
ENV VITE_USE_MOCK=false
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2: python runtime. Serves the API and the built UI (web/dist) from a
# single uvicorn process on :8000 (api/main.py mounts web/dist at "/" when it
# exists).
#
# Dependencies are installed directly (not via `pip install .`) so the engine
# runs from /app as plain source, exactly like the documented dev workflow
# (`uvicorn api.main:app`, `python -m sim.generate`, ... all run from the repo
# root) -- this avoids relocating api/main.py into site-packages, which would
# break its `Path(__file__).resolve().parents[1] / "web" / "dist"` lookup.
#
# Runtime-only dependency set (see pyproject.toml's [project.dependencies]
# comment): scipy, scikit-learn and httpx are eval-/test-only and never
# imported by `uvicorn api.main:app`, `python -m sim.generate`, or
# `python -m api.ingest` (including --build-demo) -- they stay out of the
# image. `uvicorn` + `websockets` (not `uvicorn[standard]`, which pulls in
# uvloop/httptools/watchfiles/python-dotenv) is enough for the /ws/live
# websocket endpoint.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Strip bundled test suites and bytecode caches from site-packages in the
# SAME layer as the install, so they never end up in an image layer at all.
RUN pip install --no-cache-dir \
    "pydantic>=2" \
    numpy \
    networkx \
    rapidfuzz \
    pyyaml \
    typer \
    rich \
    fastapi \
    uvicorn \
    websockets \
    sqlalchemy \
    && find /usr/local/lib/python3.12/site-packages -type d -name "tests" -prune -exec rm -rf {} + \
    && find /usr/local/lib/python3.12/site-packages -type d -name "test" -prune -exec rm -rf {} + \
    && find /usr/local/lib/python3.12/site-packages -type d -name "__pycache__" -prune -exec rm -rf {} +

# Runtime source only: eval/ is NOT copied as a package -- api/routers/eval.py
# reads eval/reports/*.json as plain files at request time, it never imports
# the eval/ Python package (confirmed by grep: no api/, engine/, or sim/
# module used at runtime imports from eval/).
COPY engine/ ./engine/
COPY api/ ./api/
COPY sim/ ./sim/
COPY eval/reports/ ./eval/reports/

COPY --from=web-build /web/dist ./web/dist

RUN useradd --create-home --uid 1000 urbantrace \
    && mkdir -p /data \
    && chown -R urbantrace:urbantrace /app /data

USER urbantrace

# URBANTRACE_DB_PATH must point at a volume, never a bind mount into the
# (OneDrive) project folder -- see docker-compose.yml's `urbantrace-db` named
# volume.
ENV URBANTRACE_DB_PATH=/data/urbantrace.db
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
