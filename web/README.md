# UrbanTrace web console

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

or put `VITE_USE_MOCK=false` in `web/.env.local` (see `.env.example`). Nothing else changes: the mock implements the same `UrbanTraceApi` interface (`src/api/client.ts`) with contract-exact types (`src/api/types.ts`).

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

## v2 (contract "v2 additions", 2026-09-25)

Types, client and mock cover every v2 endpoint: `/api/watchlist` (GET/POST/DELETE), `/api/watchlist/hits`, `/api/analytics/heatmap`, `/api/analytics/flow_trend`, plus the new fields on `Alert`, `PathPoint`, `TrajectoryDetail` and `Corridor`.

- **Watchlist** (`/watchlist`): add a plate or `?` pattern with a reason. The form previews the canonical layouts using the same grammar as `api/plate_grammar.py` (`src/lib/plate.ts`) and blocks duplicates. It lists entries with hit counts, removes entries after a confirm step, and shows recent hits. A hit marked *Consensus* matched on the fused plate, even though that camera's own read may be wrong (the wrong slots are highlighted).
- **Watchlist alerts**: they carry their own badge in the Live rail and on the Alerts page (with a *Watchlist* filter). The detail says it plainly: "This camera read X, but the vehicle's fused plate across N cameras matches your watchlist entry P at 98%". It also shows a slot-by-slot comparison, the probability against the 50% threshold, and the matched-on value.
- **Heatmap** (Live map, top right): toggle it, then pick *Density* or *Speed*, and *Live* or *Snapshot*.
  - *Live* accumulates `/ws/live` events client-side over the last 15 min of sim time (`src/hooks/useHeat.ts`), so it moves during replay.
  - *Snapshot* calls `/api/analytics/heatmap`. Live falls back to the snapshot until the feed has data.
  - Both use the contract definitions: density = reads per camera normalised to [0, 1]; speed = road-graph distance / travel time, excluding links above v_max (`src/lib/speed.ts`, 120 km/h as in the engine).
  - Speed is drawn as "slowness", so congestion glows. The legend lists the busiest or slowest cameras as numbers.
- **Analytics**:
  - The corridor table gains road distance, avg and P85 km/h, a speed-vs-free-flow bar, and congestion chips on a sequential amber scale, always with the number.
  - A flow-trend chart shows reads and mean speed as two synced charts (no dual axis).
  - A bottleneck ranking lists corridors by lowest avg speed ÷ free-flow speed.
- **Direction**: the trajectory map draws an arrow from each camera along `heading_deg`. The header shows "Heading NE" with the bearing, and each link row shows its heading.
- **Results**: sections for real-plate OCR, the OCR progression, synthetic OCR (labelled), the detector, linking vs baselines, the stress sweep and calibration, then the older AUC sections. Every other report sits in a collapsed raw view.
  - The OCR target verdict is computed from the report, not hard-coded.
  - `src/api/mock/eval/*.json` are byte-identical copies of `../eval/reports/*.json`, loaded with `import.meta.glob`. Refresh them with `cp ../eval/reports/*.json src/api/mock/eval/`.
- **Mock behaviour** (matches the API): watchlist matching is probabilistic with P ≥ 0.5, checked on each single read and on the trajectory consensus. Each (entry, trajectory) pair alerts once, as severity `high`, and later matches become hits. DELETE hard-deletes. Replay reset keeps entries and drops that replay's hits.
  - So the demo always fires, adding an entry plants two target vehicles: a dirty plate that every camera misreads (caught only by consensus), and a clean one (caught on a single read).
- **Bundles**: every page except Live is lazy-loaded (`React.lazy`), so Recharts is only fetched for Analytics and Results.

## Deployment note

Routes use `BrowserRouter`, so a static host must fall back to `index.html` (nginx: `try_files $uri /index.html;`).
