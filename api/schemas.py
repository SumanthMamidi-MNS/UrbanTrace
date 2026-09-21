"""Pydantic response models mirroring `web/src/api/types.ts` /
docs/api-contract.md EXACTLY — field names, nullability, everything. This
module is the FastAPI-side half of the frozen contract; do not rename or
retype a field here without the lead changing the contract first.
"""

from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class RoadNode(BaseModel):
    node_id: str
    lat: float
    lon: float


class RoadEdge(BaseModel):
    from_node: str
    to_node: str
    length_m: float
    speed_limit_kmh: float


class Camera(BaseModel):
    camera_id: str
    name: str
    lat: float
    lon: float
    node_id: str
    bearing_deg: float
    is_border: bool


class CameraStats(Camera):
    events_total: int
    volume_last_hour: int


class EventSummary(BaseModel):
    event_id: str
    camera_id: str
    timestamp: str
    plate_argmax: str
    plate_confidence: float
    color: str
    vehicle_type: str
    trajectory_id: str | None


class SlotRead(BaseModel):
    char: str
    prob: float


class SlotPosteriorOut(BaseModel):
    top: list[SlotRead]
    unread: bool


class EventDetail(EventSummary):
    plate_posterior: list[SlotPosteriorOut]


class LinkEvidence(BaseModel):
    from_event_id: str
    to_event_id: str
    plate_lr: float | None
    appearance_lr: float | None
    kinematic_lr: float | None
    prior_log_odds: float
    total_log_odds: float | None
    delta_t_s: float
    expected_t_s: float
    skipped_cameras: list[str]


class TrajectorySummary(BaseModel):
    trajectory_id: str
    decoded_plate: str
    plate_confidence: float
    start_time: str
    end_time: str
    n_events: int
    camera_sequence: list[str]
    color: str
    vehicle_type: str
    has_alert: bool


class PathPoint(BaseModel):
    event_id: str
    camera_id: str
    lat: float
    lon: float
    timestamp: str


class PlateConsensus(BaseModel):
    per_slot: list[list[SlotRead]]
    single_read_plates: list[str]
    entropy_bits: float


class TrajectoryDetail(TrajectorySummary):
    events: list[EventSummary]
    links: list[LinkEvidence]
    path: list[PathPoint]
    consensus: PlateConsensus


class SearchHit(BaseModel):
    trajectory: TrajectorySummary
    probability: float
    matched_plate: str


AlertType = Literal["clone", "impossible_travel", "anomaly"]
AlertSeverity = Literal["high", "medium", "low"]


class AlertEvidence(BaseModel):
    distance_m: float | None = None
    delta_t_s: float | None = None
    min_required_s: float | None = None
    appearance_distance: float | None = None
    points: list[PathPoint] | None = None


class Alert(BaseModel):
    alert_id: str
    type: AlertType
    severity: AlertSeverity
    created_at: str
    plate: str
    trajectory_ids: list[str]
    summary: str
    evidence: AlertEvidence


class Page(BaseModel, Generic[T]):  # noqa: UP046 - PEP 695 type params not used: keep py3.12 pydantic v2 generics simple
    items: list[T]
    total: int
    limit: int
    offset: int


class Health(BaseModel):
    status: Literal["ok"]
    dataset: str
    n_events: int
    n_trajectories: int


class City(BaseModel):
    nodes: list[RoadNode]
    edges: list[RoadEdge]
    cameras: list[Camera]


class AnalyticsSummary(BaseModel):
    n_events: int
    n_trajectories: int
    mean_trip_duration_s: float
    mean_events_per_trajectory: float
    plate_repair_rate: float
    active_alerts: int


class VolumeBucket(BaseModel):
    camera_id: str
    bucket_start: str
    count: int


class OdMatrix(BaseModel):
    zones: list[str]
    matrix: list[list[int]]


class Corridor(BaseModel):
    from_camera: str
    to_camera: str
    n_trips: int
    median_travel_s: float
    p90_travel_s: float
    free_flow_s: float
    congestion_index: float


class EvalReports(BaseModel):
    reports: dict[str, object]


ReplayAction = Literal["start", "pause", "reset"]


class ReplayRequest(BaseModel):
    action: ReplayAction
    speed: float | None = None


class ReplayState(BaseModel):
    running: bool
    speed: float
    sim_time: str


class ClockData(BaseModel):
    sim_time: str
    speed: float
    running: bool


class LiveEventMessage(BaseModel):
    type: Literal["event"] = "event"
    data: EventSummary


class LiveTrajectoryMessage(BaseModel):
    type: Literal["trajectory"] = "trajectory"
    data: TrajectorySummary


class LiveAlertMessage(BaseModel):
    type: Literal["alert"] = "alert"
    data: Alert


class LiveClockMessage(BaseModel):
    type: Literal["clock"] = "clock"
    data: ClockData
