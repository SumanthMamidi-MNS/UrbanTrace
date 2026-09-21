# SUTRA web console

React + Vite + TypeScript + Tailwind v4, MapLibre GL (v5), Recharts, TanStack Query.
Built against the frozen contract in `../docs/api-contract.md`.

## Run

```bash
cd web
npm install
npm run dev          # http://localhost:5173 — mock mode by default, no backend needed
```

### Mock vs real API

| Variable | Default | Meaning |
|---|---|---|
| `VITE_USE_MOCK` | `true` | `true`: in-browser fixture city + simulated `/ws/live`. `false`: real API. |
| `VITE_API_BASE` | `http://localhost:8000` | REST base; WebSocket is derived as `ws://…/ws/live`. |

Switch to the real engine:

```bash
VITE_USE_MOCK=false npm run dev            # bash
$env:VITE_USE_MOCK='false'; npm run dev    # PowerShell
```

or put `VITE_USE_MOCK=false` in `web/.env.local` (see `.env.example`). Nothing else changes: the mock implements the same `SutraApi` interface (`src/api/client.ts`) with contract-exact types (`src/api/types.ts`).

## Checks

```bash
npx tsc --noEmit -p tsconfig.app.json   # type check (root tsconfig only holds project references)
npm run lint                            # oxlint
npm run build                           # tsc -b && vite build -> dist/
```

## Pages

- **Live** — road network from `/api/city` (no tile server, works offline), cameras sized by last-hour volume and pulsing on reads, trajectories drawn along roads as they extend, alert feed + read ticker. Replay controls (start/pause/reset, 1×/10×/60×/300×) live in the top bar on every page.
- **Tracks** — filterable trajectory list; detail has animated path replay, camera-hit timeline, the **WHY panel** (per-link waterfall: prior → plate → appearance → travel time = total log-odds, plain-English verdict, Δt vs expected; `null` channels shown as non-finite and excluded from the sum) and the **plate consensus panel** (single reads with wrong characters highlighted over the fused plate, per-slot confidence on a log "nines" scale).
- **Search** — `?` wildcards (`MH12??1234`), colour/type/time filters, probability-ranked hits.
- **Analytics** — KPI cards, volume over time, 5-zone OD heatmap, corridor congestion table.
- **Alerts** — clone / impossible-travel / anomaly list; detail shows both sightings on the map, distance, observed gap vs minimum physically required time, implied speed, appearance distance.
- **Results** — renders `/api/eval`. Known reports get dedicated views (stratified AUC matrix, clone disjoint vs overlap, appearance scaling curve, gating/blocking); unknown reports fall back to a generic renderer; missing ones show empty states.

## Mock data

`src/api/mock/` builds a seeded 9×9 grid city (50 cameras around a Pune-like centre), ~330 historical trajectories (06:00–12:00 sim time) with OCR misreads, 5 planted clone/impossible-travel alerts + 2 anomalies, and a live simulator that spawns vehicles and occasional clones from 12:00 onward. Showcase vehicle: `MH 12 AB 1234`, misread at 3 of 6 cameras with one missed camera. `src/api/mock/eval/*.json` are verbatim copies of `eval/reports/*.json` — refresh them when the engine regenerates reports.

## Deployment note

Routes use `BrowserRouter`, so a static host must fall back to `index.html` (nginx: `try_files $uri /index.html;`).
