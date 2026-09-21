"""Origin-destination matrix (architecture.md §4; docs/api-contract.md
`/api/analytics/od_matrix`).

Zones are `N, S, E, W, Central`, assigned per-camera from its lat/lon
relative to the city's own camera centroid -- there is no real-world
geocoding available (the city is synthetic, `sim/city.py`), so "North"
etc. are always relative to this dataset's own camera layout, never an
absolute compass reference. A trip's origin zone is its trajectory's first
camera; destination is its last camera (architecture.md §4, verbatim).
"""

from dataclasses import dataclass

from engine.contracts.city import Camera, CityConfig
from engine.contracts.trajectory import Trajectory

ZONES = ["N", "S", "E", "W", "Central"]

# A camera within this fraction of the layout's own max centroid-distance
# counts as "Central" rather than being forced into one of the four
# compass wedges -- otherwise every camera, including ones essentially at
# the centroid, would get an arbitrary N/S/E/W label from noise in its tiny
# lat/lon offset.
CENTRAL_RADIUS_FRACTION = 0.25


@dataclass
class ZoneAssignment:
    centroid_lat: float
    centroid_lon: float
    max_distance: float
    zone_by_camera: dict[str, str]


def _centroid(cameras: list[Camera]) -> tuple[float, float]:
    lat = sum(c.lat for c in cameras) / len(cameras)
    lon = sum(c.lon for c in cameras) / len(cameras)
    return lat, lon


def assign_zones(city: CityConfig) -> ZoneAssignment:
    """Assign every camera to one of `ZONES` relative to the camera-network
    centroid: within `CENTRAL_RADIUS_FRACTION` of the largest camera offset
    -> Central; otherwise the compass direction of whichever axis (lat=N/S,
    lon=E/W) has the larger-magnitude offset."""
    cameras = city.cameras
    if not cameras:
        return ZoneAssignment(0.0, 0.0, 0.0, {})
    centroid_lat, centroid_lon = _centroid(cameras)

    offsets = {c.camera_id: (c.lat - centroid_lat, c.lon - centroid_lon) for c in cameras}
    distances = {cid: (dlat**2 + dlon**2) ** 0.5 for cid, (dlat, dlon) in offsets.items()}
    max_distance = max(distances.values()) if distances else 0.0
    central_radius = CENTRAL_RADIUS_FRACTION * max_distance

    zone_by_camera: dict[str, str] = {}
    for c in cameras:
        dlat, dlon = offsets[c.camera_id]
        if distances[c.camera_id] <= central_radius:
            zone_by_camera[c.camera_id] = "Central"
        elif abs(dlat) >= abs(dlon):
            zone_by_camera[c.camera_id] = "N" if dlat > 0 else "S"
        else:
            zone_by_camera[c.camera_id] = "E" if dlon > 0 else "W"

    return ZoneAssignment(
        centroid_lat=centroid_lat,
        centroid_lon=centroid_lon,
        max_distance=max_distance,
        zone_by_camera=zone_by_camera,
    )


def build_od_matrix(
    trajectories: list[Trajectory], city: CityConfig
) -> tuple[list[str], list[list[int]]]:
    """`(zones, matrix)` -- `matrix[i][j]` = number of trajectories whose
    origin camera is in `zones[i]` and destination camera is in `zones[j]`.
    A trajectory with no cameras at all (should not occur -- every
    trajectory has >= 1 event) contributes nothing rather than raising."""
    assignment = assign_zones(city)
    index_of = {z: i for i, z in enumerate(ZONES)}
    matrix = [[0 for _ in ZONES] for _ in ZONES]

    for traj in trajectories:
        if not traj.camera_sequence:
            continue
        origin_cam = traj.camera_sequence[0]
        dest_cam = traj.camera_sequence[-1]
        origin_zone = assignment.zone_by_camera.get(origin_cam)
        dest_zone = assignment.zone_by_camera.get(dest_cam)
        if origin_zone is None or dest_zone is None:
            continue
        matrix[index_of[origin_zone]][index_of[dest_zone]] += 1

    return list(ZONES), matrix


__all__ = [
    "CENTRAL_RADIUS_FRACTION",
    "ZONES",
    "ZoneAssignment",
    "assign_zones",
    "build_od_matrix",
]
