# API contract — SUTRA (frozen)

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
