# SUTRA dev tasks. Windows teammates: use `make.ps1` instead (same target
# names) if you don't have `make` on PATH -- e.g. `./make.ps1 setup`.
#
# SUTRA_DB_PATH should point OUTSIDE this (OneDrive-synced) folder -- see
# README's troubleshooting section. Example (bash): export
# SUTRA_DB_PATH=~/.local/share/sutra/sutra.db

ifeq ($(OS),Windows_NT)
    PY := .venv/Scripts/python.exe
else
    PY := .venv/bin/python
endif

DB ?= data/sutra.db

.PHONY: setup test lint sim pipeline ingest serve docker-up docker-seed

setup:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"
	cd web && npm install

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	cd web && npm run lint

# Small demo dataset: 25 cameras, 4,000 vehicles, 6 hours (~few minutes).
# For the full target-scale run (architecture.md §8): make sim CAMERAS=50
# VEHICLES=20000 HOURS=24
CAMERAS ?= 25
VEHICLES ?= 4000
HOURS ?= 6
SEED ?= 42

sim:
	$(PY) -m sim.generate --cameras $(CAMERAS) --vehicles $(VEHICLES) --hours $(HOURS) --seed $(SEED) --out data/run1

pipeline:
	$(PY) -m eval.run_pipeline --data data/run1 --out data/run1/pipeline

ingest:
	$(PY) -m api.ingest --data data/run1 --trajectories data/run1/pipeline/trajectories.jsonl --db $(DB)

serve:
	$(PY) -m uvicorn api.main:app --port 8000

docker-up:
	docker compose up --build sutra

docker-seed:
	docker compose --profile seed run --rm seed
