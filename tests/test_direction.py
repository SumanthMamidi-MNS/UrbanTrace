"""Bearing / compass-direction math (docs/api-contract.md Contract v2,
`PathPoint.heading_deg` / `TrajectoryDetail.direction_label`).
`engine.analytics.direction.bearing_deg` on known coordinates: due north and
due east must land exactly on 0 and 90 degrees."""

from __future__ import annotations

import pytest

from engine.analytics.direction import bearing_deg, compass_label

# Roughly 111km per degree of latitude near the equator/mid-latitudes --
# these two points are far enough apart that longitude convergence at this
# latitude doesn't perturb the bearing meaningfully off 0/90.
LAT0, LON0 = 19.0000, 73.0000


def test_due_north_is_zero_degrees():
    bearing = bearing_deg(LAT0, LON0, LAT0 + 0.1, LON0)
    assert bearing == pytest.approx(0.0, abs=1e-6)


def test_due_south_is_180_degrees():
    bearing = bearing_deg(LAT0, LON0, LAT0 - 0.1, LON0)
    assert bearing == pytest.approx(180.0, abs=1e-6)


def test_due_east_is_90_degrees():
    bearing = bearing_deg(LAT0, LON0, LAT0, LON0 + 0.1)
    assert bearing == pytest.approx(90.0, abs=1)


def test_due_west_is_270_degrees():
    bearing = bearing_deg(LAT0, LON0, LAT0, LON0 - 0.1)
    assert bearing == pytest.approx(270.0, abs=1)


def test_coincident_points_bearing_zero():
    assert bearing_deg(LAT0, LON0, LAT0, LON0) == 0.0


@pytest.mark.parametrize(
    "heading,label",
    [
        (0.0, "N"),
        (44.0, "NE"),
        (90.0, "E"),
        (134.0, "SE"),
        (180.0, "S"),
        (224.0, "SW"),
        (270.0, "W"),
        (314.0, "NW"),
        (359.0, "N"),
    ],
)
def test_compass_label(heading: float, label: str):
    assert compass_label(heading) == label
