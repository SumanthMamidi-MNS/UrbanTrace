"""Trajectory and link-evidence contracts (output of the L3 linking engine).

Day 1 only defines the shapes; scoring/association logic lands Day 2-3.
"""

from datetime import datetime

from pydantic import BaseModel


class LinkEvidence(BaseModel):
    from_event_id: str
    to_event_id: str
    plate_lr: float
    appearance_lr: float
    kinematic_lr: float
    prior_log_odds: float
    total_log_odds: float
    delta_t_s: float
    expected_t_s: float
    skipped_cameras: list[str] = []


class Trajectory(BaseModel):
    trajectory_id: str
    event_ids: list[str]
    decoded_plate: str
    plate_confidence: float
    start_time: datetime
    end_time: datetime
    camera_sequence: list[str]
    links: list[LinkEvidence] = []
    gt_vehicle_id: str | None = None
