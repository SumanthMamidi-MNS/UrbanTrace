"""DetectionEvent and related perception-layer contracts.

This is the hard contract between ingest (L1) and the linking engine (L3):
the engine consumes DetectionEvents and neither knows nor cares whether
they came from the simulator, real video, or a CSV replay.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, field_validator

from engine.contracts.plate import NUM_SLOTS, SlotPosterior

EMBEDDING_DIM = 128
_NORM_TOLERANCE = 1e-3


class VehicleAttributes(BaseModel):
    color: str
    vehicle_type: str
    color_confidence: float
    type_confidence: float


class DetectionEvent(BaseModel):
    event_id: str
    camera_id: str
    timestamp: datetime

    plate_posterior: list[SlotPosterior]  # length 10, ALWAYS present
    plate_argmax: str  # convenience only, derived
    plate_confidence: float

    embedding: list[float]  # length 128, L2-normalised
    attributes: VehicleAttributes
    crop_uri: str | None = None
    source: Literal["sim", "video", "csv"]
    gt_vehicle_id: str | None = None  # simulator only, for evaluation

    # Row index into a companion embeddings.npy when this event was loaded
    # via engine.contracts.store.EventStore from the compact on-disk format.
    # Storage bookkeeping only — does not change how the engine reads
    # `.embedding` above, which is always populated.
    embedding_ref: int | None = None

    @field_validator("plate_posterior")
    @classmethod
    def _check_num_slots(cls, v: list[SlotPosterior]) -> list[SlotPosterior]:
        if len(v) != NUM_SLOTS:
            raise ValueError(f"plate_posterior must have {NUM_SLOTS} slots, got {len(v)}")
        return v

    @field_validator("embedding")
    @classmethod
    def _check_embedding(cls, v: list[float]) -> list[float]:
        if len(v) != EMBEDDING_DIM:
            raise ValueError(f"embedding must have length {EMBEDDING_DIM}, got {len(v)}")
        norm = sum(x * x for x in v) ** 0.5
        if norm > 0 and abs(norm - 1.0) > _NORM_TOLERANCE:
            raise ValueError(f"embedding must be L2-normalised, got norm={norm}")
        return v
