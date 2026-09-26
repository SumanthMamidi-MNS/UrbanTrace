# API contract — UrbanTrace (frozen)

The single source of truth between `api/` (engine side) and `web/` (UI side). Both sides build against this document. Changing it requires the lead's sign-off. `api/openapi.json` is generated from the implementation and must agree with this file.

## Conventions

- Base path: `/api`. WebSocket: `/ws/live`.
- All timestamps: ISO-8601 strings (`2026-01-01T08:15:02.113Z`).
- Log-likelihood values are floats. **Non-finite floats (`±inf`, `NaN`) are serialised as `null`** — JSON has no infinity.
- Paginated responses: `Page<T> = { items: T[], total: number, limit: number, offset: number }`. Defaults `limit=50`, max `500`.
- Errors: HTTP status + `{ "detail": string }`.
- CORS open for `http://localhost:5173` (Vite dev).

## Shared types

```ts
type RoadNode  = { node_id: string; lat: number; lon: number }
type RoadEdge  = { from_node: string; to_node: string; length_m: number; speed_limit_kmh: number }
type Camera    = { camera_id: string; name: string; lat: number; lon: number; node_id: string;
                   bearing_deg: number; is_border: boolean }
type CameraStats = Camera & { events_total: number; volume_last_hour: number }

type EventSummary = {
  event_id: string; camera_id: string; timestamp: string;
  plate_argmax: string;            // 10-slot canonical, "_" = blank
  plate_confidence: number;        // 0..1
  color: string; vehicle_type: string;
  trajectory_id: string | null;
}

type SlotRead = { char: string; prob: number }
type EventDetail = EventSummary & {
  plate_posterior: { top: SlotRead[]; unread: boolean }[];   // 10 entries, top <= 5 chars
}

type LinkEvidence = {
  from_event_id: string; to_event_id: string;
  plate_lr: number | null; appearance_lr: number | null; kinematic_lr: number | null;
  prior_log_odds: number; total_log_odds: number | null;
  delta_t_s: number; expected_t_s: number; skipped_cameras: string[];
}

type TrajectorySummary = {
  trajectory_id: string; decoded_plate: string; plate_confidence: number;
  start_time: string; end_time: string; n_events: number;
  camera_sequence: string[]; color: string; vehicle_type: string;
  has_alert: boolean;
}

type PathPoint = { event_id: string; camera_id: string; lat: number; lon: number; timestamp: string }

type PlateConsensus = {
  per_slot: SlotRead[][];          // 10 entries, fused posterior top <= 5 per slot
  single_read_plates: string[];    // plate_argmax of each constituent event, in time order
  entropy_bits: number;
}

type TrajectoryDetail = TrajectorySummary & {
  events: EventSummary[]; links: LinkEvidence[]; path: PathPoint[]; consensus: PlateConsensus;
}

type SearchHit = { trajectory: TrajectorySummary; probability: number; matched_plate: string }

type Alert = {
  alert_id: string;
  type: "clone" | "impossible_travel" | "anomaly";
  severity: "high" | "medium" | "low";
  created_at: string;
  plate: string;
  trajectory_ids: string[];
  summary: string;                                   // one human-readable sentence
  evidence: {
    distance_m?: number; delta_t_s?: number; min_required_s?: number;
    appearance_distance?: number; points?: PathPoint[];
  };
}
```

## Endpoints

| Method | Path | Query | Returns |
|---|---|---|---|
| GET | `/api/health` | — | `{ status: "ok", dataset: string, n_events: number, n_trajectories: number }` |
| GET | `/api/city` | — | `{ nodes: RoadNode[], edges: RoadEdge[], cameras: Camera[] }` |
| GET | `/api/cameras` | — | `CameraStats[]` |
| GET | `/api/events` | `camera_id?, from?, to?, limit?, offset?` | `Page<EventSummary>` (newest first) |
| GET | `/api/events/{event_id}` | — | `EventDetail` |
| GET | `/api/trajectories` | `from?, to?, plate?, camera_id?, min_len?, has_alert?, limit?, offset?` | `Page<TrajectorySummary>` (newest first) |
| GET | `/api/trajectories/{trajectory_id}` | — | `TrajectoryDetail` |
| GET | `/api/search` | `q` (plate, `?` = one unknown char), `color?, vehicle_type?, from?, to?, limit?` | `SearchHit[]` (probability desc) |
| GET | `/api/analytics/summary` | — | `{ n_events, n_trajectories, mean_trip_duration_s, mean_events_per_trajectory, plate_repair_rate, active_alerts }` |
| GET | `/api/analytics/volumes` | `bucket_minutes? (default 60), camera_id?` | `{ camera_id: string, bucket_start: string, count: number }[]` |
| GET | `/api/analytics/od_matrix` | — | `{ zones: string[], matrix: number[][] }` (zones = N, S, E, W, Central; row = origin) |
| GET | `/api/analytics/corridors` | `limit? (default 20)` | `{ from_camera, to_camera, n_trips, median_travel_s, p90_travel_s, free_flow_s, congestion_index }[]` |
| GET | `/api/alerts` | `type?, limit?` | `Alert[]` (newest first) |
| GET | `/api/eval` | — | `{ reports: Record<string, unknown> }` — raw contents of `eval/reports/*.json` keyed by file stem |
| POST | `/api/replay` | body `{ action: "start" \| "pause" \| "reset", speed?: number }` | `{ running: boolean, speed: number, sim_time: string }` |

`plate_repair_rate` = fraction of multi-event trajectories whose consensus plate differs from at least one of its single reads (i.e. OCR errors the linking corrected).

`congestion_index` = `median_travel_s / free_flow_s` (1.0 = free flow).

## WebSocket `/ws/live`

Server → client JSON messages, one per frame:

```ts
{ type: "event",      data: EventSummary }
{ type: "trajectory", data: TrajectorySummary }   // emitted when a trajectory is created or extended
{ type: "alert",      data: Alert }
{ type: "clock",      data: { sim_time: string; speed: number; running: boolean } }   // ~1 Hz
```

Replay is driven by `POST /api/replay`; the socket only streams.

---

## Contract v2 additions (2026-09-25, lead sign-off) — PRD components 2, 3, 4

Additive only: every v1 field and endpoint is unchanged. Clients built against v1 keep working.

### Changed types (new optional-safe fields)

```ts
// Alert.type gains "watchlist"; Alert.evidence gains watchlist fields.
type Alert = {
  ...v1 fields,
  type: "clone" | "impossible_travel" | "anomaly" | "watchlist";
  evidence: {
    ...v1 evidence fields,
    watchlist_entry_id?: string; pattern?: string; match_probability?: number;
    matched_on?: "single_read" | "trajectory_consensus";
  };
}

// Direction of travel (PRD: "direction ... and route").
type PathPoint = { ...v1 fields, heading_deg: number | null }   // bearing to the NEXT point, 0 = north, clockwise; null on the last point
type TrajectoryDetail = { ...v1 fields, overall_heading_deg: number | null; direction_label: string | null }  // first->last camera, label = N/NE/E/SE/S/SW/W/NW

// Corridor speeds (PRD: "average vehicle speeds").
type Corridor = { ...v1 fields (from_camera, to_camera, n_trips, median_travel_s, p90_travel_s, free_flow_s, congestion_index),
                  distance_m: number; avg_speed_kmh: number; p85_speed_kmh: number; free_flow_speed_kmh: number }
```

### New types

```ts
type WatchlistEntry = {
  entry_id: string;
  pattern: string;              // as typed by the operator, same grammar as /api/search ("MH12??1234", "DL3C?456")
  canonical_patterns: string[]; // grammar-consistent 10-slot forms it expands to
  reason: string;               // free text, e.g. "stolen vehicle FIR 1234/2026"
  created_at: string;
  active: boolean;
  hits: number;
}

type WatchlistHit = {
  hit_id: string; entry_id: string; pattern: string;
  event_id: string; trajectory_id: string | null; camera_id: string; timestamp: string;
  probability: number;                          // P(plate matches pattern | evidence)
  matched_on: "single_read" | "trajectory_consensus";
  plate_read: string;                           // what this camera actually read (plate_argmax)
}

type HeatPoint = { camera_id: string; lat: number; lon: number; weight: number }
type Heatmap = { at: string; window_minutes: number; metric: "density" | "speed"; points: HeatPoint[] }

type FlowBucket = { bucket_start: string; events: number; active_trajectories: number; mean_speed_kmh: number | null }
```

### New endpoints

| Method | Path | Query / body | Returns |
|---|---|---|---|
| GET | `/api/watchlist` | — | `WatchlistEntry[]` |
| POST | `/api/watchlist` | body `{ pattern: string, reason: string }` | `WatchlistEntry` (422 with a helpful detail if the pattern cannot be a plate) |
| DELETE | `/api/watchlist/{entry_id}` | — | `{ deleted: true }` |
| GET | `/api/watchlist/hits` | `entry_id?, limit?` | `WatchlistHit[]` (newest first) |
| GET | `/api/analytics/heatmap` | `at? (default: current sim time, or end of data), window_minutes? (default 15), metric? ("density" default, or "speed")` | `Heatmap` |
| GET | `/api/analytics/flow_trend` | `bucket_minutes? (default 15)` | `FlowBucket[]` |

### Semantics

- **Watchlist matching is probabilistic.** A read matches when P(plate ∈ pattern | evidence) ≥ 0.5 (server constant, documented). It is evaluated twice:
  1. on each **single read**, using that read's own plate posterior;
  2. on the **trajectory consensus** as the trajectory grows.
  The second is the differentiator: a watchlisted vehicle is caught **even when this camera misread its plate**, because the fused plate across cameras still matches. Each (entry, trajectory) pair alerts at most once; later matches become hits, not new alerts.
- Watchlist entries persist in the database. During replay, matches emit `alert` messages on `/ws/live` in real time, exactly like other alerts.
- **Heatmap `density`** = reads per camera in the window, normalised to [0, 1]. **`speed`** = mean speed of trajectory links arriving at that camera in the window (km/h). The UI's live heatmap may instead accumulate `/ws/live` events client-side; both must agree on the definition.
- **Speeds** use road-graph shortest-path distance between consecutive cameras divided by the observed travel time; links implying more than the kinematic `v_max` are excluded (they are clone evidence, not speed).
