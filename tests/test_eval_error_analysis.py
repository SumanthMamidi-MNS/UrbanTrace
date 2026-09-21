"""Fast, fully in-memory tests for eval/error_analysis.py's pure functions
(classification + aggregation) -- no dependency on data/run1 on disk, so
this stays quick and doesn't need the full-city dataset to exist."""

from datetime import datetime

from eval.error_analysis import _classify_merge_pair, analyze_merges, analyze_splits

VEH_A = {"true_plate": "KA01AB1234", "is_clone": False, "clone_of": None,
         "vehicle_type": "car", "color": "white"}
VEH_B_CLONE = {"true_plate": "KA01AB1234", "is_clone": True, "clone_of": "veh_a",
               "vehicle_type": "bike", "color": "red"}
VEH_C_NEAR_MISS = {"true_plate": "KA01AB1235", "is_clone": False, "clone_of": None,
                    "vehicle_type": "bus", "color": "black"}
VEH_D_SAME_ATTRS = {"true_plate": "TS99ZZ0001", "is_clone": False, "clone_of": None,
                     "vehicle_type": "car", "color": "white"}
VEH_E_OTHER = {"true_plate": "RJ55QQ9999", "is_clone": False, "clone_of": None,
               "vehicle_type": "bus", "color": "green"}


def test_classify_merge_pair_clone_by_identical_true_plate():
    assert _classify_merge_pair("veh_a", VEH_A, "veh_b", VEH_B_CLONE) == "clone_pair"


def test_classify_merge_pair_clone_by_explicit_clone_of():
    v_clone = {**VEH_B_CLONE, "true_plate": "KA01AB9999"}  # differ, so only clone_of catches it
    assert _classify_merge_pair("veh_a", VEH_A, "veh_b", v_clone) == "clone_pair"


def test_classify_merge_pair_near_miss_plate():
    assert _classify_merge_pair("veh_a", VEH_A, "veh_c", VEH_C_NEAR_MISS) == "near_miss_plate"


def test_classify_merge_pair_same_type_and_colour():
    category = _classify_merge_pair("veh_a", VEH_A, "veh_d", VEH_D_SAME_ATTRS)
    assert category == "same_vehicle_type_and_colour"


def test_classify_merge_pair_other():
    assert _classify_merge_pair("veh_a", VEH_A, "veh_e", VEH_E_OTHER) == "other"


def test_classify_merge_pair_ocr_argmax_collision_overrides_other():
    # True plates are unrelated (would be "other"), but the two OBSERVED
    # reads at the merge point happen to look alike -- OCR noise, not a
    # structural plate ambiguity.
    category = _classify_merge_pair(
        "veh_a", VEH_A, "veh_e", VEH_E_OTHER, pa_a="RJ55QQ9998", pa_b="RJ55QQ9999"
    )
    assert category == "ocr_argmax_collision"


def _event(gt, t, pa):
    epoch = datetime.fromisoformat(t).timestamp()
    return {"gt": gt, "cam": "cam_x", "t": t, "epoch": epoch, "pa": pa}


def test_analyze_merges_counts_one_wrong_transition_and_prior_diagnostics():
    vehicles = {"veh_a": VEH_A, "veh_e": VEH_E_OTHER}
    event_index = {
        "e1": _event("veh_a", "2026-01-01T00:00:00", "KA01AB1234"),
        "e2": _event("veh_a", "2026-01-01T00:00:10", "KA01AB1234"),
        # wrong link e2 -> e3: different, unrelated vehicle ("other" category)
        "e3": _event("veh_e", "2026-01-01T08:30:00", "RJ55QQ9999"),
        "e4": _event("veh_e", "2026-01-01T08:30:20", "RJ55QQ9999"),
    }
    trajectories = [
        {
            "trajectory_id": "traj_0",
            "event_ids": ["e1", "e2", "e3", "e4"],
            "links": [
                {
                    "from_event_id": "e1",
                    "to_event_id": "e2",
                    "plate_lr": 10.0,
                    "appearance_lr": 1.0,
                    "kinematic_lr": 1.0,
                    "total_log_odds": 10.6,
                },
                {
                    "from_event_id": "e2",
                    "to_event_id": "e3",
                    "plate_lr": -4.57,
                    "appearance_lr": -0.78,
                    "kinematic_lr": 6.42,
                    "total_log_odds": -0.32,
                },
                {
                    "from_event_id": "e3",
                    "to_event_id": "e4",
                    "plate_lr": 10.0,
                    "appearance_lr": 1.0,
                    "kinematic_lr": 1.0,
                    "total_log_odds": 10.6,
                },
            ],
        }
    ]

    result = analyze_merges(trajectories, event_index, vehicles, mean_candidates_per_event=218.0)

    assert result["n_predicted_trajectories"] == 1
    assert result["n_trajectories_containing_a_merge"] == 1
    assert result["n_wrong_adjacent_transitions"] == 1
    assert result["category_counts"] == {"other": 1}
    assert result["top_category"] == "other"
    assert result["hour_of_day_counts"] == {8: 1}

    pbd = result["prior_bug_diagnostics"]
    assert pbd["n_wrong_links_with_evidence"] == 1
    # total_log_odds -0.32 < 0 under the current buggy prior.
    assert pbd["n_wrong_links_negative_total_log_odds"] == 1
    assert pbd["share_wrong_links_negative_total_log_odds"] == 1.0
    assert pbd["n_wrong_links_currently_accepted"] == 0


def test_analyze_merges_no_merge_when_single_vehicle():
    vehicles = {"veh_a": VEH_A}
    event_index = {
        "e1": _event("veh_a", "2026-01-01T00:00:00", "KA01AB1234"),
        "e2": _event("veh_a", "2026-01-01T00:00:10", "KA01AB1234"),
    }
    trajectories = [{"trajectory_id": "traj_0", "event_ids": ["e1", "e2"], "links": []}]
    result = analyze_merges(trajectories, event_index, vehicles)
    assert result["n_trajectories_containing_a_merge"] == 0
    assert result["n_wrong_adjacent_transitions"] == 0
    assert result["category_counts"] == {}
    assert result["top_category"] is None


def test_analyze_splits_detects_missed_detection_gap():
    vehicles = {"veh_a": VEH_A}
    # veh_a's true journey has THREE passages, but the middle one was never
    # detected (simulator camera miss) -- the two detected events end up in
    # different predicted trajectories.
    journeys = {
        "veh_a": {
            "passages": [
                {"camera_id": "cam_1", "timestamp": "2026-01-01T00:00:00"},
                {"camera_id": "cam_2", "timestamp": "2026-01-01T00:05:00"},  # missed
                {"camera_id": "cam_3", "timestamp": "2026-01-01T00:10:00"},
            ]
        }
    }
    event_index = {
        "e1": _event("veh_a", "2026-01-01T00:00:00", "KA01AB1234"),
        "e2": _event("veh_a", "2026-01-01T00:10:00", "KA01AB1234"),
    }
    trajectories = [
        {"trajectory_id": "traj_0", "event_ids": ["e1"], "links": []},
        {"trajectory_id": "traj_1", "event_ids": ["e2"], "links": []},
    ]

    result = analyze_splits(trajectories, event_index, journeys, vehicles)
    assert result["n_gt_vehicles_split"] == 1
    assert result["n_split_breaks"] == 1
    assert result["cause_counts"] == {"missed_detection_gap": 1}


def test_analyze_splits_no_split_when_all_in_one_trajectory():
    vehicles = {"veh_a": VEH_A}
    journeys = {"veh_a": {"passages": []}}
    event_index = {
        "e1": _event("veh_a", "2026-01-01T00:00:00", "KA01AB1234"),
        "e2": _event("veh_a", "2026-01-01T00:00:10", "KA01AB1234"),
    }
    trajectories = [{"trajectory_id": "traj_0", "event_ids": ["e1", "e2"], "links": []}]
    result = analyze_splits(trajectories, event_index, journeys, vehicles)
    assert result["n_gt_vehicles_split"] == 0
    assert result["n_split_breaks"] == 0
