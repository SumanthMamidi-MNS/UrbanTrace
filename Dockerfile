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
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime
WORKDIR /app

RUN pip install --no-cache-dir \
    "pydantic>=2" \
    numpy \
    scipy \
    scikit-learn \
    networkx \
    rapidfuzz \
    pyyaml \
    typer \
    rich \
    fastapi \
    "uvicorn[standard]" \
    sqlalchemy \
    httpx

COPY engine/ ./engine/
COPY api/ ./api/
COPY sim/ ./sim/
COPY eval/ ./eval/

COPY --from=web-build /web/dist ./web/dist

RUN useradd --create-home --uid 1000 sutra \
    && mkdir -p /data \
    && chown -R sutra:sutra /app /data

USER sutra

# SUTRA_DB_PATH must point at a volume, never a bind mount into the (OneDrive)
# project folder -- see docker-compose.yml's `sutra-db` named volume.
ENV SUTRA_DB_PATH=/data/sutra.db
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
