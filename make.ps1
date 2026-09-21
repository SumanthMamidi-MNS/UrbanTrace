<#
    SUTRA dev tasks (Windows PowerShell equivalent of the Makefile).

    Usage:  ./make.ps1 <target>
    Targets: setup, test, lint, sim, pipeline, ingest, serve, docker-up, docker-seed

    SUTRA_DB_PATH is set here to $env:LOCALAPPDATA\sutra\sutra.db by default --
    OUTSIDE this OneDrive-synced project folder, where SQLite write latency is
    much lower (see README's troubleshooting section). Override it yourself
    before calling this script if you want a different location.
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("setup", "test", "lint", "sim", "pipeline", "ingest", "serve", "docker-up", "docker-seed")]
    [string]$Target,

    [int]$Cameras = 25,
    [int]$Vehicles = 4000,
    [int]$Hours = 6,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
$PY = ".\.venv\Scripts\python.exe"

if (-not $env:SUTRA_DB_PATH) {
    $env:SUTRA_DB_PATH = Join-Path $env:LOCALAPPDATA "sutra\sutra.db"
}
New-Item -ItemType Directory -Force -Path (Split-Path $env:SUTRA_DB_PATH) | Out-Null

switch ($Target) {
    "setup" {
        python -m venv .venv
        & $PY -m pip install --upgrade pip
        & $PY -m pip install -e ".[dev]"
        Push-Location web
        npm install
        Pop-Location
    }
    "test" {
        & $PY -m pytest
    }
    "lint" {
        & $PY -m ruff check .
        Push-Location web
        npm run lint
        Pop-Location
    }
    "sim" {
        # Small demo dataset by default (~few minutes). For the full
        # target-scale run (architecture.md §8):
        #   ./make.ps1 sim -Cameras 50 -Vehicles 20000 -Hours 24
        & $PY -m sim.generate --cameras $Cameras --vehicles $Vehicles --hours $Hours --seed $Seed --out data/run1
    }
    "pipeline" {
        & $PY -m eval.run_pipeline --data data/run1 --out data/run1/pipeline
    }
    "ingest" {
        & $PY -m api.ingest --data data/run1 --trajectories data/run1/pipeline/trajectories.jsonl --db $env:SUTRA_DB_PATH
    }
    "serve" {
        & $PY -m uvicorn api.main:app --port 8000
    }
    "docker-up" {
        docker compose up --build sutra
    }
    "docker-seed" {
        docker compose --profile seed run --rm seed
    }
}
