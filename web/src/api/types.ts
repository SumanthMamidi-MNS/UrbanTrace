/**
 * Types mirroring docs/api-contract.md EXACTLY (frozen contract).
 * Do not add, rename or re-type fields here without the lead changing the contract first.
 */

export type RoadNode = { node_id: string; lat: number; lon: number }
export type RoadEdge = { from_node: string; to_node: string; length_m: number; speed_limit_kmh: number }
export type Camera = {
  camera_id: string
  name: string
  lat: number
  lon: number
  node_id: string
  bearing_deg: number
  is_border: boolean
}
export type CameraStats = Camera & { events_total: number; volume_last_hour: number }

export type EventSummary = {
  event_id: string
  camera_id: string
  timestamp: string
  /** 10-slot canonical, "_" = blank */
  plate_argmax: string
  /** 0..1 */
  plate_confidence: number
  color: string
  vehicle_type: string
  trajectory_id: string | null
}

export type SlotRead = { char: string; prob: number }
export type EventDetail = EventSummary & {
  /** 10 entries, top <= 5 chars */
  plate_posterior: { top: SlotRead[]; unread: boolean }[]
}

/** Non-finite floats (±inf, NaN) arrive as null. */
export type LinkEvidence = {
  from_event_id: string
  to_event_id: string
  plate_lr: number | null
  appearance_lr: number | null
  kinematic_lr: number | null
  prior_log_odds: number
  total_log_odds: number | null
  delta_t_s: number
  expected_t_s: number
  skipped_cameras: string[]
}

export type TrajectorySummary = {
  trajectory_id: string
  decoded_plate: string
  plate_confidence: number
  start_time: string
  end_time: string
  n_events: number
  camera_sequence: string[]
  color: string
  vehicle_type: string
  has_alert: boolean
}

export type PathPoint = { event_id: string; camera_id: string; lat: number; lon: number; timestamp: string }

export type PlateConsensus = {
  /** 10 entries, fused posterior top <= 5 per slot */
  per_slot: SlotRead[][]
  /** plate_argmax of each constituent event, in time order */
  single_read_plates: string[]
  entropy_bits: number
}

export type TrajectoryDetail = TrajectorySummary & {
  events: EventSummary[]
  links: LinkEvidence[]
  path: PathPoint[]
  consensus: PlateConsensus
}

export type SearchHit = { trajectory: TrajectorySummary; probability: number; matched_plate: string }

export type AlertType = 'clone' | 'impossible_travel' | 'anomaly'
export type AlertSeverity = 'high' | 'medium' | 'low'

export type Alert = {
  alert_id: string
  type: AlertType
  severity: AlertSeverity
  created_at: string
  plate: string
  trajectory_ids: string[]
  /** one human-readable sentence */
  summary: string
  evidence: {
    distance_m?: number
    delta_t_s?: number
    min_required_s?: number
    appearance_distance?: number
    points?: PathPoint[]
  }
}

export type Page<T> = { items: T[]; total: number; limit: number; offset: number }

// ---- Endpoint response shapes ----

export type Health = { status: 'ok'; dataset: string; n_events: number; n_trajectories: number }

export type City = { nodes: RoadNode[]; edges: RoadEdge[]; cameras: Camera[] }

export type AnalyticsSummary = {
  n_events: number
  n_trajectories: number
  mean_trip_duration_s: number
  mean_events_per_trajectory: number
  plate_repair_rate: number
  active_alerts: number
}

export type VolumeBucket = { camera_id: string; bucket_start: string; count: number }

export type OdMatrix = { zones: string[]; matrix: number[][] }

export type Corridor = {
  from_camera: string
  to_camera: string
  n_trips: number
  median_travel_s: number
  p90_travel_s: number
  free_flow_s: number
  congestion_index: number
}

export type EvalReports = { reports: Record<string, unknown> }

export type ReplayAction = 'start' | 'pause' | 'reset'
export type ReplayRequest = { action: ReplayAction; speed?: number }
export type ReplayState = { running: boolean; speed: number; sim_time: string }

// ---- Query params ----

export type EventsQuery = { camera_id?: string; from?: string; to?: string; limit?: number; offset?: number }

export type TrajectoriesQuery = {
  from?: string
  to?: string
  plate?: string
  camera_id?: string
  min_len?: number
  has_alert?: boolean
  limit?: number
  offset?: number
}

export type SearchQuery = {
  q: string
  color?: string
  vehicle_type?: string
  from?: string
  to?: string
  limit?: number
}

export type VolumesQuery = { bucket_minutes?: number; camera_id?: string }
export type CorridorsQuery = { limit?: number }
export type AlertsQuery = { type?: AlertType; limit?: number }

// ---- WebSocket /ws/live ----

export type ClockData = { sim_time: string; speed: number; running: boolean }

export type LiveMessage =
  | { type: 'event'; data: EventSummary }
  | { type: 'trajectory'; data: TrajectorySummary }
  | { type: 'alert'; data: Alert }
  | { type: 'clock'; data: ClockData }
