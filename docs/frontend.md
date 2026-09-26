# UrbanTrace — Frontend Reference

**At a glance**
- A single-page React console (`web/`) for a city-scale multi-camera ANPR tracking engine: a live map, per-vehicle trajectory replay with an evidence explainer, partial-plate search, analytics, alert triage, a watchlist, and an evaluation-results viewer.
- Built with React 19, TypeScript, Vite 8 and Tailwind v4; MapLibre GL v5 draws the road network with no tile server; TanStack Query handles all server state; Recharts draws the charts.
- It runs in two modes controlled by one env flag, `VITE_USE_MOCK`: against an in-browser fixture city (default, no backend needed) or against the real FastAPI engine, through the same typed client interface either way.
- In production it ships as static files (`web/dist`) served directly by the FastAPI process — one `uvicorn` process is the whole demo.
- Every type in `src/api/types.ts` mirrors `docs/api-contract.md` field-for-field; the contract, not this document, is the source of truth if the two ever disagree.

## Table of contents

1. [Purpose](#purpose)
2. [Tech stack](#tech-stack)
3. [Folder structure](#folder-structure)
4. [Routing table](#routing-table)
5. [Pages](#pages)
6. [API client layer](#api-client-layer)
7. [Mock mode](#mock-mode)
8. [Map rendering](#map-rendering)
9. [Styling system](#styling-system)
10. [Persisted state](#persisted-state)
11. [Commands](#commands)
12. [Serving the built UI](#serving-the-built-ui)
13. [Known warnings and limitations](#known-warnings-and-limitations)
14. [How to add a page](#how-to-add-a-page)

## Purpose

The console is the operator-facing surface of UrbanTrace: it turns per-camera ANPR reads and the linking engine's trajectories, evidence, and alerts into something a traffic-police operator or a hackathon judge can read at a glance. It never runs any tracking logic itself — everything it shows (trajectories, likelihood ratios, clone/impossible-travel alerts, watchlist matches, analytics) is computed by the Python engine and served over REST/WebSocket; the frontend's job is presentation, filtering, and light client-side derivation (road-following polylines, a rolling live heatmap, plate-pattern preview) that does not duplicate the engine's probabilistic scoring.

## Tech stack

| Library | Version (`web/package.json`) | Why |
|---|---|---|
| React | ^19.2.8 | UI framework. |
| React DOM | ^19.2.8 | React's browser renderer. |
| TypeScript | ~6.0.2 | Static typing across the app; `src/api/types.ts` is typed to match the frozen API contract exactly. |
| Vite | ^8.3.0 | Dev server and bundler. |
| `@vitejs/plugin-react` | ^6.1.1 | React fast-refresh/JSX support for Vite. |
| Tailwind CSS | ^4.3.3 | Utility-first styling; theme tokens declared in `src/index.css`. |
| `@tailwindcss/vite` | ^4.3.3 | Tailwind's Vite plugin (v4's CSS-first config, no `tailwind.config.js`). |
| MapLibre GL | ^5.24.0 | Map rendering. Per `docs/decisions.md`: chosen over Mapbox/Google because it needs no API token and works offline at a demo table; **pinned to v5, not v6**, because v6's worker never initialised under Vite 8 in dev or production build, while v5 renders offline with the worker in the main bundle. |
| Recharts | ^3.10.1 | Charts on the Analytics and Results pages. Per `docs/architecture.md`: "quick, good enough". |
| `@tanstack/react-query` | ^5.103.0 | Server-state fetching, caching, background refetch, and mutations for every REST call. |
| `react-router-dom` | ^7.18.4 | Client-side routing (`BrowserRouter`). |
| `@types/geojson` | ^7946.0.16 | Types for the GeoJSON features built for MapLibre sources. |
| `@types/node` | ^24.13.3 | Node types (build tooling only). |
| `@types/react`, `@types/react-dom` | ^19.2.18 / ^19.2.7 | React type definitions. |
| oxlint | ^1.81.0 | Linting (`npm run lint`); configured in `.oxlintrc.json` with `react`, `typescript`, `oxc` plugins. |

`docs/decisions.md` also records: MapLibre labels are drawn as HTML markers rather than text layers, because MapLibre text layers need a glyph/font server, which would break the offline demo; and the UI's mock mode is deliberately lazy-loaded behind `VITE_USE_MOCK` and implements the same typed client, so switching to the live API is a one-flag change, and the mock's `/api/eval` payload is a byte-identical copy of `eval/reports/*.json` so the Results page never shows invented numbers.

## Folder structure

```
web/
├── index.html                  HTML shell; mounts #root, loads src/main.tsx
├── package.json                scripts + dependencies
├── vite.config.ts              Vite config: React + Tailwind plugins, dev port 5173, 1500 kB chunk-size warning limit
├── tsconfig.json                project-reference root (no compiler options of its own)
├── tsconfig.app.json            app compiler options (bundler resolution, ES2023, strict unused-var/param checks)
├── tsconfig.node.json           compiler options for Vite config itself
├── .oxlintrc.json               lint rule config
├── .env.example / .env.local    VITE_USE_MOCK / VITE_API_BASE
├── public/favicon.svg
├── dist/                        production build output (served by FastAPI)
└── src/
    ├── main.tsx                 React root render
    ├── App.tsx                  route table, QueryClient, lazy page imports
    ├── config.ts                USE_MOCK / API_BASE / WS_URL, derived from env vars
    ├── index.css                Tailwind import + design tokens (dark theme) + a few global rules
    ├── vite-env.d.ts             Vite client type reference
    ├── api/
    │   ├── client.ts             UrbanTraceApi interface, real fetch/WebSocket implementation, mock wiring, ApiError
    │   ├── hooks.ts               one TanStack Query hook per endpoint + query-key registry (qk)
    │   ├── types.ts               types mirroring docs/api-contract.md exactly
    │   └── mock/
    │       ├── index.ts           builds the fixture world once, exposes mockApi + subscribeMockLive
    │       ├── world.ts           builds the historical city, cameras, trajectories, events, posteriors
    │       ├── server.ts          implements UrbanTraceApi entirely from the in-memory world (createMockApi)
    │       ├── live.ts            LiveSim: simulated clock + spawns live events/trajectories/alerts
    │       ├── watchlist.ts       mock watchlist matching (probabilistic, mirrors the API's rules)
    │       ├── rng.ts             seeded PRNG so the fixture world is deterministic
    │       ├── evalFixtures.ts    loads api/mock/eval/*.json via import.meta.glob into the /api/eval shape
    │       └── eval/*.json        byte-identical copies of eval/reports/*.json
    ├── components/
    │   ├── Layout.tsx             app shell: side nav, header (wordmark + tagline), replay transport bar, alert nav badge
    │   ├── Wordmark.tsx           BrandMark (a path through three camera points, also the favicon) and the two-tone UrbanTrace wordmark
    │   ├── CityMap.tsx             MapLibre wrapper: roads, cameras, trajectory lines/points/labels/arrows, heat layer, live pulses
    │   ├── HeatOverlay.tsx         heatmap on/off + metric/source controls, and the ranked legend
    │   ├── WhyPanel.tsx            per-link evidence waterfall + plain-English verdict (Tracks detail)
    │   ├── ConsensusPanel.tsx      plate-consensus visualisation (single reads vs fused plate)
    │   ├── Watchlist.tsx           shared watchlist widgets (pattern chars, matched-on tag, evidence block) used by Alerts + Watchlist pages
    │   ├── icons.tsx               inline SVG icon set used by nav/buttons
    │   └── ui.tsx                  generic building blocks: Panel, Plate, Button, Field, Swatch, badges, EmptyState/ErrorState/Loading, Segmented, Stat, Bar
    ├── hooks/
    │   ├── liveStore.ts            LiveStore: single WebSocket connection shared app-wide, buffers events/alerts/trajectories/clock, exposes useLive()
    │   └── useHeat.ts               useHeat(): merges the live client-side heat buffer with the /api/analytics/heatmap snapshot
    ├── lib/
    │   ├── geo.ts                   RoadRouter (Dijkstra over /api/city graph), polyline slicing/length helpers
    │   ├── plate.ts                 plate formatting, slot helpers, and a TS port of api/plate_grammar.py for pattern preview
    │   ├── speed.ts                 shared link-speed formula (distance / dt, v_max-gated) used by mock + live heatmap
    │   ├── direction.ts              bearing / compass-label helpers for heading_deg
    │   ├── format.ts                 number/time/duration/percent/log-odds formatters
    │   └── heat.ts                   the heatmap colour ramp
    └── pages/
        ├── LivePage.tsx              Live map + read ticker + alert feed
        ├── TrajectoriesPage.tsx      Tracks list + routes to detail
        ├── TrajectoryDetail.tsx      Tracks detail: path replay, WHY panel, consensus panel
        ├── SearchPage.tsx            partial-plate search
        ├── AnalyticsPage.tsx         KPIs, volumes, OD matrix, corridors, flow trend
        ├── AlertsPage.tsx            alert list + detail (clone/impossible-travel/anomaly/watchlist)
        ├── WatchlistPage.tsx         watchlist entries + hits
        ├── ResultsPage.tsx           /api/eval report viewer (dispatches to results/*)
        └── results/
            ├── EngineSections.tsx    detector / linking-vs-baselines / gating-blocking / stress-calibration sections
            ├── OcrSection.tsx        real-plate OCR progression + synthetic OCR sections
            ├── bits.tsx              small shared report-rendering widgets (Tile, Methodology, Description, Missing, PctBar)
            └── shared.ts             shared parsing helpers for arbitrary report JSON (num, pct, str, objs, CHART colours)
```

## Routing table

Defined in `src/App.tsx` (`BrowserRouter`, so a static host needs an `index.html` fallback — see `web/README.md`'s deployment note).

| Route | Component | Shows | Endpoints/hooks used |
|---|---|---|---|
| `/` | — | redirects to `/live` | — |
| `/live` | `LivePage` (main bundle) | Road map, live trajectories/reads, alert ticker, heatmap toggle | `useCity`, `useCameras`, `useAlerts`, `useHeat`/`useHeatmap`, live WebSocket via `useLive` |
| `/trajectories`, `/trajectories/:id` | `TrajectoriesPage` (lazy) | Filterable trajectory list; `:id` opens `TrajectoryDetail` inline | `useCity`, `useTrajectories`; detail: `useTrajectory`, `useAlerts` |
| `/search` | `SearchPage` (lazy) | Partial-plate + filter search | `useSearch` |
| `/analytics` | `AnalyticsPage` (lazy) | KPI cards, volumes, OD matrix, corridors, flow trend | `useSummary`, `useVolumes`, `useOdMatrix`, `useCorridors`, `useFlowTrend`, `useCity` |
| `/alerts`, `/alerts/:id` | `AlertsPage` (lazy) | Alert list + detail with map and physics check | `useAlerts`, `useCity`, `api.trajectory` (via `useQueries`) |
| `/watchlist` | `WatchlistPage` (lazy) | Watchlist entries, add/remove, hits | `useWatchlist`, `useAddWatchlist`, `useDeleteWatchlist`, `useWatchlistHits` |
| `/results` | `ResultsPage` (lazy) | `/api/eval` report viewer | `useEval` |
| `*` | — | redirects to `/live` | — |

`LivePage` ships in the main bundle since it is the landing page; every other page is `React.lazy`-loaded (`App.tsx` comment: "Live is the landing page and ships in the main chunk; everything else (Recharts-heavy Analytics and Results especially) loads on first visit").

## Pages

### Live (`src/pages/LivePage.tsx`)

The default landing page. Renders the road network from `/api/city` on `CityMap` (no tile server, so it works fully offline), sizes camera markers by `volume_last_hour` (from `useCameras`), and pulses a camera marker on each live read via `liveStore`'s pulse listeners. Live trajectories extend along the road-graph polyline as new events arrive. A read ticker and an alert feed run alongside the map. The replay transport (start/pause/reset, 1×/10×/60×/300×) is not on this page — it lives in `Layout.tsx`'s header and is visible on every page. The heatmap toggle (`HeatControls`/`HeatLegend`) lets an operator switch between *Density* and *Speed*, and between *Live* (client-accumulated over the last 15 minutes of sim time) and *Snapshot* (`/api/analytics/heatmap`).

### Tracks (`src/pages/TrajectoriesPage.tsx` + `TrajectoryDetail.tsx`)

A filterable list of trajectories (by time range, camera, minimum length, has-alert). Selecting one opens `TrajectoryDetailView`, which shows:
- An animated path replay along the road-graph polyline (`getRouter`/`sliceLine` from `lib/geo.ts`), with directional arrows drawn from each camera's `heading_deg`.
- A camera-hit timeline.
- The **WHY panel** (`components/WhyPanel.tsx`): a per-link evidence waterfall — prior → plate → appearance → travel-time log-odds → total — with a plain-English verdict sentence (`verdict()` in the same file) and Δt vs. expected travel time. Non-finite channels (`null` on the wire) are shown as "non-finite" and excluded from the total, per `lib/format.ts`'s `fmtLogOdds`.
- The **plate consensus panel** (`components/ConsensusPanel.tsx`): every constituent single read is shown with wrong characters highlighted against the fused plate, and per-slot confidence is drawn on a log "nines" scale (`nines()`: 90% → 1/4 of the bar, 99% → 2/4, 99.9% → 3/4, 99.99% → full), so a jump from 99% to 99.99% confidence is visually distinguishable.

### Search (`src/pages/SearchPage.tsx`)

Free-text plate search against `/api/search`. Accepts `?` as a single-character wildcard (e.g. `MH12??1234`), plus colour, vehicle-type, and time-range filters. Results are `SearchHit`s ranked by `probability` (the probability the fused plate matches the query given the evidence, computed engine-side — the mock replicates the same idea locally in `api/mock/server.ts`'s `search()`).

### Analytics (`src/pages/AnalyticsPage.tsx`)

KPI cards (`useSummary`), a volume-over-time chart (`useVolumes`, Recharts `AreaChart`), a 5-zone OD heatmap (`useOdMatrix`), and a corridor table (`useCorridors`) with road distance, average/P85 km/h, a speed-vs-free-flow bar, and congestion chips. A flow-trend chart (`useFlowTrend`) shows reads and mean speed as two synced charts rather than one dual-axis chart. Speeds use `SPEED_V_MAX_KMH` (120, from `lib/speed.ts`) to exclude physically-impossible links from corridor speed stats, consistent with the engine's hard kinematic gate.

### Alerts (`src/pages/AlertsPage.tsx`)

Lists `clone`, `impossible_travel`, `anomaly`, and `watchlist` alerts (`useAlerts`, filterable by type). The clone/impossible-travel detail shows both sightings on the map, the distance between them, the observed time gap vs. the minimum time physically required to cover that distance, the implied speed, and the appearance distance between the two sightings — the "physics check" that makes an impossible-travel alert legible without reading log-odds. Watchlist alert detail uses the shared `components/Watchlist.tsx` widgets to show the slot-by-slot pattern match, the match probability against the watchlist's 50% threshold, and whether the match was on a single camera read or the fused trajectory consensus.

### Watchlist (`src/pages/WatchlistPage.tsx`)

Add a plate or `?`-pattern with a reason; the form previews every canonical 10-slot layout the pattern expands to, using `lib/plate.ts`'s `expandPlatePattern()` — a direct TypeScript port of the same grammar as `api/plate_grammar.py` (state letters, 1–2 digit RTO code, 0–2 letter series, 1–4 digit number) — and blocks duplicate patterns. Lists entries with hit counts, supports removal behind a confirm step, and shows recent hits (`useWatchlistHits`). A hit's `matched_on` field distinguishes a match on a single camera read from a match on the fused trajectory consensus (i.e. that camera's own read may have been wrong).

### Results (`src/pages/ResultsPage.tsx` + `src/pages/results/*`)

Renders `/api/eval`, whose payload is `{ reports: Record<string, unknown> }` — one entry per file under `eval/reports/`. Known report keys get a dedicated view: `OcrSection.tsx` renders the real-plate OCR progression (CRNN synthetic-only → CRNN + real fine-tune → fast-plate-ocr zero-shot → fast-plate-ocr fine-tuned, in that documented order) and computes its "PRD target > 90%" verdict directly from the report numbers (`TargetVerdict`, not hard-coded), plus a labelled synthetic-OCR section. `EngineSections.tsx` renders the detector metrics, linking-vs-baselines, gating/blocking, and the stress sweep/calibration sections. Any other report key falls back to a generic renderer built from `results/shared.ts`'s type-narrowing helpers (`num`, `pct`, `str`, `objs`, `isObj`); a report that is entirely missing from the payload shows an empty state instead of a broken chart.

## API client layer

`src/api/client.ts` defines the `UrbanTraceApi` interface — one method per REST endpoint in `docs/api-contract.md` — plus a `LiveMessage`/WebSocket subscription contract. Two implementations satisfy it:

- **`httpApi`**: a thin `fetch` wrapper (`request<T>()`) that builds query strings, sets `Accept`/`Content-Type` headers, and throws `ApiError` (carrying the contract's `{ detail }` body and HTTP status, or status `0` with a "Cannot reach API" message on a network failure).
- **`mockApi`**: a `Proxy` that lazily `import()`s `./mock` on first call (so the fixture-generation code is not on the real build's hot path) and forwards every call to the real mock implementation.

`export const api: UrbanTraceApi = USE_MOCK ? mockApi : httpApi` is the single switch point; every page imports `api` (directly, or through `src/api/hooks.ts`) and never branches on `USE_MOCK` itself.

**Live stream**: `subscribeLive(onMessage, onStatus)` is likewise swapped between `httpSubscribe` (a real `WebSocket` to `WS_URL` with exponential backoff reconnect, capped at 10 s) and `mockSubscribe` (subscribes to the in-memory `LiveSim`). `src/hooks/liveStore.ts` wraps this in a single app-wide `LiveStore` (one socket, shared via `useSyncExternalStore`), which buffers the last 120 events, 50 alerts, and 60 trajectory updates, tracks a rolling one-minute arrivals window for the "reads/min" readout, and republishes to React at most every ~200 ms so a 300× replay does not force a render per frame. It handles four `LiveMessage` types (from `api/types.ts`): `event`, `trajectory`, `alert`, and `clock` (sim time, running flag, speed).

**TanStack Query** (`src/api/hooks.ts`): one hook per endpoint (e.g. `useCity`, `useTrajectories`, `useAlerts`, `useWatchlist`), keyed through a central `qk` registry so cache keys stay consistent across pages. Defaults set in `App.tsx`'s `QueryClient`: `staleTime: 5_000`, `retry: 1`, `refetchOnWindowFocus: false`. Individual hooks override this where it matters: `useCity` uses `staleTime: Infinity` (the city graph is static for a session); `useHealth`/`useSummary` poll every 15 s; `useCameras` every 10 s (default); `useWatchlistHits`/`useHeatmap` poll every 10 s / 20 s while enabled; `useFlowTrend` every 30 s; list-style queries (`useTrajectories`, `useVolumes`, `useSearch`, `useWatchlistHits`, `useFlowTrend`) use `placeholderData: keepPreviousData` so pagination/filter changes don't flash a loading state. `useReplay` and the watchlist add/delete hooks are mutations that invalidate the relevant query keys (`['watchlist']`, or everything on a replay `reset`).

## Mock mode

Controlled by `VITE_USE_MOCK` (`src/config.ts`, default `true`) and `VITE_API_BASE` (default `http://localhost:8000`, also used to derive `WS_URL` by swapping the `http`/`https` prefix for `ws`/`wss`). `web/.env.example` documents the flag; `web/.env.local` (present in this checkout) sets `VITE_USE_MOCK=false` to point the dev server at a real running API. Toggling this one variable is the only change needed to move between fixture data and the live engine, because both implementations satisfy the same `UrbanTraceApi` type.

The mock world (`src/api/mock/world.ts`) builds a seeded 9×9 grid city (50 cameras, seeded PRNG in `rng.ts` for determinism) with roughly 330 historical trajectories and OCR misreads baked in, plus a handful of planted clone/impossible-travel alerts and anomalies; `live.ts`'s `LiveSim` then spawns further vehicles (and occasional clones) forward from that point. `server.ts` (`createMockApi`) implements every `UrbanTraceApi` method purely from this in-memory world — including its own `/api/search`, corridor-speed, and watchlist-matching logic, deliberately mirroring the real engine's rules (see `docs/decisions.md` and the speed formula shared in `lib/speed.ts`) rather than returning canned data.

The `/api/eval` payload is the one place accuracy matters most for a judge, so it is not reconstructed: `evalFixtures.ts` loads `src/api/mock/eval/*.json` via `import.meta.glob(..., { eager: true })` and keys each by its filename stem — and those files are documented as byte-identical copies of `eval/reports/*.json`, refreshed with `cp ../eval/reports/*.json src/api/mock/eval/` (see `web/README.md` and `docs/decisions.md`, which notes this was verified by the lead so "the Results page never shows invented numbers").

## Map rendering

`components/CityMap.tsx` wraps a single MapLibre GL `Map` instance per page. The base style (`STYLE` constant) has no tile source at all — just a flat background colour — since the road network itself comes from `/api/city`'s `nodes`/`edges`/`cameras` and is converted to a `FeatureCollection` of `LineString`s (`roadsGeoJSON()`), deduplicating undirected edges. Everything else drawn on top — camera markers, trajectory `MapLine`s, `MapPoint`s, text `MapLabel`s (rendered as HTML markers, not MapLibre text layers, per the decisions-log reasoning above), directional `MapArrow`s (rotated to `heading_deg`), and the heat layer — is passed in as typed props by the page, so `CityMap` itself has no knowledge of trajectories, alerts, or the engine's data model.

The heatmap overlay (`components/HeatOverlay.tsx`, `hooks/useHeat.ts`, `lib/heat.ts`) supports two metrics and two sources:
- **Density**: reads per camera in the window, live-accumulated counts normalised to `[0, 1]` client-side, or the API's already-normalised `/api/analytics/heatmap` values.
- **Speed**: road-graph distance ÷ observed travel time between consecutive cameras on a trajectory, excluding any link implying a speed above `SPEED_V_MAX_KMH` (120 km/h, matching the engine's `DEFAULT_V_MAX_KMH` in `engine/decode/clone_detect.py` per `lib/speed.ts`'s own comment) — those links are treated as clone evidence, not real speed. It is drawn as "slowness" (`speedToWeight` in `useHeat.ts`) so congestion glows brighter, with the underlying km/h numbers still shown in the legend.
- **Live** accumulates `/ws/live` frames into a rolling buffer (`liveStore.getHeatBuffer()`, up to `HEAT_BUFFER_MIN = 60` minutes of sim time) and falls back to the **Snapshot** (`/api/analytics/heatmap`) until that buffer has data for the current window.

## Styling system

Tailwind v4's CSS-first configuration lives in `src/index.css`: a `@theme` block declares the full colour and font palette as CSS custom properties (`--color-ink-*` for a 5-step-plus neutral dark scale, one `--color-accent-*` triad for signal cyan, one `--color-caution-*` triad for amber used on misreads/medium severity/weak evidence, and one `--color-alert-*` triad reserved strictly for alerts — stated directly in the file's own comment), plus `--font-sans` (Inter, falling back to system fonts) and `--font-mono` (JetBrains Mono, falling back to system monospace stacks). `color-scheme: dark` is set on `:root` and mirrored in `index.html`'s `<meta name="color-scheme" content="dark">` — the console is dark-theme only, there is no light mode. A `.num` utility class applies `font-variant-numeric: tabular-nums` for aligned numeric columns. A handful of global rules restyle MapLibre's default popups/controls to match the dark palette, define two attention keyframe animations (`ticker-in`, `alert-in` — a brief colour flash for new live rows and new alerts, both disabled under `prefers-reduced-motion: reduce`), and a `.hatch` diagonal-stripe background used for a caution/muted fill. `src/components/ui.tsx` builds the shared component vocabulary (`Panel`, `Plate`, `Button`, `Field`, `Swatch`, `SeverityBadge`, `AlertTypeTag`, `EmptyState`/`ErrorState`/`Loading`, `Segmented`, `Stat`, `Bar`) on top of these tokens so pages compose from a small, consistent set of primitives rather than one-off styling.

## Persisted state

The only `localStorage` usage in the app is in `src/pages/LivePage.tsx`'s `usePersisted()` hook, which remembers the Live page's heatmap controls (on/off, metric, source) across reloads. It is explicitly documented in the code as "per-viewer convenience only" — reads/writes are wrapped in `try`/`catch` so a blocked or unavailable storage silently falls back to in-memory state instead of breaking the page.

## Commands

From `web/package.json` scripts, verified against `web/README.md`'s "Checks" section:

| Command | Does |
|---|---|
| `npm run dev` | Starts the Vite dev server on port 5173 (`vite.config.ts`: `strictPort: true`). |
| `npm run build` | `tsc -b && vite build` — type-checks via project references, then bundles to `web/dist`. |
| `npm run lint` | Runs `oxlint`. |
| `npm run preview` | Serves the production build locally. |
| `npx tsc --noEmit -p tsconfig.app.json` | Type-check only, against the app project (the root `tsconfig.json` holds only project references, so pointing `tsc` at it directly would not check application code). |

## Serving the built UI

`api/main.py` mounts the built frontend directly: if `web/dist` exists, its `assets/` directory is mounted at `/assets` via `StaticFiles`, and a catch-all route (`GET /{full_path:path}`) serves any other file that exists under `web/dist` verbatim, or falls back to `web/dist/index.html` for any path that isn't `/api/...` or `/ws...` — i.e. a single-page-app fallback for client-side routes, implemented in Python rather than relying on a separate static-file server. This is why `BrowserRouter` (not `HashRouter`) is safe to use in production without extra nginx-style config in this deployment.

The Dockerfile builds the console in a dedicated stage (`web-build`, `node:22-slim`): `npm ci` then `npm run build` with `VITE_USE_MOCK=false` baked in at build time (so the shipped image never falls back to fixture data), and only the resulting `web/dist` directory is copied into the Python runtime stage (`COPY --from=web-build /web/dist ./web/dist`) — none of `web/`'s `node_modules` or source ships in the final image. The runtime stage then runs `uvicorn api.main:app` on port 8000, serving both the API and the static console from one process.

## Known warnings and limitations

Measured directly by running the project's own build and lint commands in `web/`:

- **`npm run build`** succeeds (`713 modules transformed`) but the main JS chunk is **1,377.29 kB (382.72 kB gzip)** — well over Vite's default 500 kB warning threshold, which is why `vite.config.ts` explicitly raises `chunkSizeWarningLimit` to 1500 to silence that specific warning rather than splitting the bundle further. Lazy-loaded page chunks are much smaller (e.g. `SearchPage` 6.06 kB, `WhyPanel` 12.76 kB, `AnalyticsPage` 31.34 kB, `ResultsPage` 85.77 kB), and the Recharts-heavy `LineChart` chunk (354.95 kB) and the mock-data chunk (184.03 kB) only load when Analytics/Results or mock mode are actually used.
- **`npx oxlint`** reports 9 warnings (0 errors), all `react` plugin rules:
  - `react/only-export-components`: `src/components/ui.tsx` (2 occurrences), `src/components/WhyPanel.tsx` (2 occurrences) export non-component values alongside components, which breaks Vite's fast-refresh boundary for those files (does not affect production behavior).
  - `react/set-state-in-effect`: `src/components/Layout.tsx:31`, `src/pages/TrajectoryDetail.tsx:40` and `:126` call `setState` synchronously inside a `useEffect`, which the linter flags as a potential cascading-render pattern that could often be replaced by deriving state during render.
  - `react/purity`: `src/pages/LivePage.tsx:151` and `:168` call `Date.now()` during render, which the linter considers an impure operation that can produce inconsistent results across renders.
  These are pre-existing lint warnings in the current codebase, not something introduced by this document; none of them are `error`-level and the build is unaffected.

## How to add a page

1. Create `src/pages/YourPage.tsx` exporting a named component (existing pages export named, not default, components).
2. Add the corresponding types to `src/api/types.ts` if the page needs a response shape not already covered, keeping it an exact mirror of `docs/api-contract.md`.
3. Add an `UrbanTraceApi` method in `src/api/client.ts` (both the `httpApi.` entry and, if the mock should behave meaningfully rather than throwing `not implemented`, a matching implementation in `src/api/mock/server.ts`), then a query/mutation hook in `src/api/hooks.ts` with a new `qk` entry.
4. Register the route in `src/App.tsx`: for anything other than the landing page, add a `lazy(() => import('./pages/YourPage').then((m) => ({ default: m.YourPage })))` and wrap it with the shared `page()` Suspense helper, then add a `<Route path="..." element={page(<YourPage />)} />` inside the `Layout` route.
5. If it needs a nav entry, add it to the `NAV` array in `src/components/Layout.tsx` with an icon from `src/components/icons.tsx` (or a new one added there).
6. Reuse `src/components/ui.tsx` primitives (`Panel`, `Plate`, `Button`, `Field`, badges, `EmptyState`/`ErrorState`/`Loading`) rather than one-off markup, and reuse `src/lib/format.ts` formatters for any time/number/percent/log-odds display so formatting stays consistent across pages.
7. Run `npm run build`, `npx tsc --noEmit -p tsconfig.app.json`, and `npm run lint` before considering the page done.
