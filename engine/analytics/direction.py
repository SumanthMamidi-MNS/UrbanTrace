"""Bearing / compass-direction helpers (PRD component 2: "...timestamps,
direction, and route..."; docs/api-contract.md Contract v2,
`PathPoint.heading_deg` / `TrajectoryDetail.overall_heading_deg` +
`direction_label`).

Standard forward-azimuth (initial bearing) formula: 0 = north, clockwise.
Treats lat/lon as ordinary spherical coordinates; `sim/city.py` lays the
synthetic city's cameras out on a small flat local grid, so the
geodesic-vs-flat-earth distinction is immaterial at this scale, but using
the real spherical formula costs nothing and is correct at any scale.
"""

import math

# 8-point compass, index = round(heading_deg / 45) % 8.
COMPASS_POINTS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing FROM (lat1, lon1) TO (lat2, lon2) in degrees, 0 =
    north, clockwise, range [0, 360). Coincident points return 0.0 (the
    formula's atan2(0, 0) = 0, an arbitrary but stable choice -- there is no
    true bearing between two identical points)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    theta = math.degrees(math.atan2(x, y))
    return (theta + 360.0) % 360.0


def compass_label(heading_deg: float) -> str:
    """8-point compass label (N/NE/E/SE/S/SW/W/NW) nearest `heading_deg`."""
    idx = round(heading_deg / 45.0) % 8
    return COMPASS_POINTS[idx]


__all__ = ["COMPASS_POINTS", "bearing_deg", "compass_label"]
