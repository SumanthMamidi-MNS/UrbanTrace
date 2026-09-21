"""EventStore: reads the (events.jsonl, embeddings.npy) pair written by
sim.generate.write_dataset back into full DetectionEvent objects.

The engine must never hand-manage the two files itself — always go through
this store so `.embedding` is transparently hydrated from the companion
float16 memmap.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from engine.contracts.codec import decode_event_compact
from engine.contracts.events import DetectionEvent
from engine.contracts.plate import PLATE_SLOTS


class EventStore:
    def __init__(self, events_path: str | Path, embeddings_path: str | Path):
        self.events_path = Path(events_path)
        self.embeddings_path = Path(embeddings_path)
        self._embeddings = np.load(self.embeddings_path, mmap_mode="r")

    def __len__(self) -> int:
        return int(self._embeddings.shape[0])

    def _hydrate(self, data: dict) -> DetectionEvent:
        row = data["er"]
        vec = np.asarray(self._embeddings[row], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return decode_event_compact(data, vec.tolist(), PLATE_SLOTS)

    def __iter__(self) -> Iterator[DetectionEvent]:
        with self.events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield self._hydrate(json.loads(line))

    def read_all(self) -> list[DetectionEvent]:
        return list(self)
