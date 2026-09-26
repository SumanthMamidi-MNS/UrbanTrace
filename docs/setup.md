# UrbanTrace — setup, tests and troubleshooting

## Quickstart — Windows, no Docker

Requires Python 3.12+ and Node 22+.

```powershell
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"     # or just -e . to only serve/simulate/ingest (no scipy/sklearn/pytest)

# Keep the SQLite DB OFF the OneDrive-synced project folder -- see Troubleshooting.
$env:URBANTRACE_DB_PATH = "$env:LOCALAPPDATA\urbantrace\urbantrace.db"
New-Item -ItemType Directory -Force -Path (Split-Path $env:URBANTRACE_DB_PATH) | Out-Null

# 1. Generate a small demo dataset (25 cameras, 4,000 vehicles, 6 simulated hours -- a few minutes).
python -m sim.generate --cameras 25 --vehicles 4000 --hours 6 --seed 42 --out data/run1

# 2. Run the real linking pipeline (gate -> fuse -> min-cost flow) and get the UrbanTrace-vs-baseline scoreboard.
python -m eval.run_pipeline --data data/run1 --out data/run1/pipeline

# 3. Load events + trajectories into SQLite.
python -m api.ingest --data data/run1 --trajectories data/run1/pipeline/trajectories.jsonl --db $env:URBANTRACE_DB_PATH

# 4. Build the UI against the real API, then serve everything from one process.
cd web
npm install
npm run build
cd ..
python -m uvicorn api.main:app --port 8000
```

Open http://localhost:8000.

Skipping step 2 (and passing `--build-demo` instead of `--trajectories ...` in step 3) links events
in-process instead of via `eval/run_pipeline.py` — faster, but you won't get the baseline-comparison
report. See `api/ingest.py`'s module docstring.

An equivalent task runner exists for both shells:

```powershell
./make.ps1 setup
./make.ps1 sim         # -Cameras / -Vehicles / -Hours / -Seed to override
./make.ps1 pipeline
./make.ps1 ingest
./make.ps1 serve
```

```bash
make setup
make sim          # CAMERAS=50 VEHICLES=20000 HOURS=24 for the full target-scale run
make pipeline
make ingest
make serve
```

## Quickstart — Docker

Requires Docker Desktop (with Compose v2) running.

```bash
# One-time: generate a small demo dataset (25 cameras, 4,000 vehicles, 6h -- a few minutes)
# and ingest it into the named volume the app container reads from.
docker compose --profile seed run --rm seed

# Build and start the app (API + UI in one container, port 8000).
docker compose up --build urbantrace
```

or with the task runner: `make docker-seed && make docker-up` / `./make.ps1 docker-seed` then
`./make.ps1 docker-up`.

Open http://localhost:8000. The SQLite DB lives on the named Docker volume `urbantrace-db`, never on a bind
mount into this folder — see **Troubleshooting**.

To reseed from scratch: `docker compose down -v` (removes the volume) then repeat the two commands
above.

## Tests

```bash
./.venv/Scripts/python.exe -m pytest      # or `make test` / `./make.ps1 test`
```

Frontend checks (from `web/`):

```bash
npx tsc --noEmit -p tsconfig.app.json     # type check
npm run lint                              # oxlint
npm run build                             # full build
```

Lint (Python): `./.venv/Scripts/python.exe -m ruff check .` (or `make lint` / `./make.ps1 lint`, which
also runs the web linter).

## Troubleshooting

**Everything (generate/ingest/serve) is painfully slow, especially SQLite writes.**
This project folder lives inside OneDrive. OneDrive's background sync makes disk I/O on files inside it
much slower than a local disk — SQLite, which does frequent small writes, feels this badly. Fix: point
`URBANTRACE_DB_PATH` at a location *outside* OneDrive, e.g. `%LOCALAPPDATA%\urbantrace\urbantrace.db` on Windows
(`make.ps1` does this for you automatically; the plain `Makefile`/manual commands need you to set the
env var yourself, as shown in the quickstart above). In Docker this is a non-issue: the DB lives on the
named volume `urbantrace-db`, which Docker manages outside any bind-mounted, OneDrive-synced folder — never
change `docker-compose.yml` to bind-mount `/data` into this repo.

**Where OCR training/raw data lives (`URBANTRACE_DATA_DIR`).**
Every default path in `engine/`, `eval/`, and `sim/` that points at raw datasets, detector/OCR weights, or
training runs (see `engine/paths.py`) is resolved from one setting: `URBANTRACE_DATA_DIR`, which defaults to
a repo-relative `./data` (gitignored). If your OCR/detector data lives somewhere else — e.g. an existing
`ocr/` tree outside the repo, alongside a separate OCR venv — set it before running any `engine.perception.*`
or `eval.*` script:

```powershell
$env:URBANTRACE_DATA_DIR = "C:\path\to\your\data"
```

This is independent of `URBANTRACE_DB_PATH` above (that's just the API's SQLite file); most people only
ever need to set `URBANTRACE_DATA_DIR` if they're re-running the OCR/detector training scripts against a
pre-existing dataset outside the repo.

**`docker compose up` fails to connect / hangs.**
Docker Desktop needs to be running (and fully started, not just launching) before `docker compose`
commands work. On Windows this can take a minute or two after starting the app.

**Port 8000 (or 5173) already in use.**
Something else is bound to it — stop it, or run `uvicorn api.main:app --port 8001` / edit the port
mapping in `docker-compose.yml`.

**The web build fails / behaves oddly.**
Needs Node 22+. If you see stale UI behavior after switching between mock and real API, check
`VITE_USE_MOCK` — the Docker image is always built with `VITE_USE_MOCK=false`; `npm run dev` defaults to
mock mode (see `web/README.md`).

**A full pipeline run is taking a long time.**
`eval/run_pipeline.py --max-hours N` evaluates only the first N simulated hours, for a quick timing
check before committing to a full run — see `trajectory_metrics_prefix.json` above. The full
target-scale run (50 cameras / 20,000 vehicles / 24h, architecture.md §8) is expected to take a while on
CPU; the demo dataset this README's quickstarts use (25 cameras / 4,000 vehicles / 6h) is deliberately
much smaller.
