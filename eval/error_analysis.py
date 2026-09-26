"""Error analysis of the FULL-CITY UrbanTrace run already on disk (architecture.md
§8's headline scoreboard, eval/reports/trajectory_metrics.json): UrbanTrace
predicts 18,483 trajectories for 19,995 true vehicles over data/run1's
107,234 events -- i.e. it OVER-MERGES. This module answers WHY, by reading
the already-solved data/run1/pipeline/trajectories.jsonl plus
data/run1/ground_truth.json and data/run1/events.jsonl and classifying
every wrongly-joined vehicle pair and every wrongly-split vehicle directly
from that fixed output. No re-solving, no re-running the pipeline.

MERGES: a predicted trajectory containing events from more than one true
vehicle. Walking that trajectory's events in time order, every point where
the ground-truth vehicle id changes from one event to the next is exactly
one wrongly-made link -- these adjacency TRANSITIONS (not "every pair of
vehicles co-occurring in the trajectory", which would double-count when 3+
vehicles land in one trajectory) are what gets classified and bucketed by
hour of day.

SPLITS: a true vehicle whose events land in more than one predicted
trajectory. Walking that vehicle's own detected events in time order, every
point where the assigned predicted trajectory id changes is exactly one
wrongly-made break, classified by comparing against the vehicle's FULL true
passage list (data/run1/ground_truth.json journeys, which lists every
passage the vehicle made -- including ones dropped by the simulator's
camera-miss model and so never producing a detection event) and against the
recomputed sliding-window boundaries (engine.association.window, same
defaults data/run1/pipeline was solved with).

Streamed line-by-line for the two large files (events.jsonl, 56MB;
trajectories.jsonl, 35MB) so nothing beyond the small per-event/per-
trajectory fields this module actually uses sits in memory at once.

Usage:
    python -m eval.error_analysis
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from engine.association.window import (
    DEFAULT_WINDOW_SIZE_S,
    DEFAULT_WINDOW_STEP_S,
    _window_bounds,
)
from engine.scoring.fusion import DEFAULT_EXPECTED_CANDIDATES
from eval.stratified import edit_distance

DATA_DIR = Path("data/run1")
PIPELINE_DIR = Path("data/run1/pipeline")
OUT_PATH = Path("eval/reports/error_analysis.json")

NEAR_MISS_MAX_EDIT_DISTANCE = 2
WINDOW_SEAM_TOLERANCE_S = 60.0
MAX_EXAMPLES_PER_CATEGORY = 5


def _load_event_index(events_path: Path) -> dict[str, dict]:
    """event_id -> {gt, cam, t (ISO string), epoch, pa}, streamed line by
    line -- only the compact JSON fields already on each line are read, no
    embedding hydration (this module never touches embeddings.npy)."""
    index: dict[str, dict] = {}
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            index[d["id"]] = {
                "gt": d.get("gt"),
                "cam": d["cam"],
                "t": d["t"],
                "epoch": datetime.fromisoformat(d["t"]).timestamp(),
                "pa": d.get("pa"),
            }
    return index


def _load_trajectories(traj_path: Path) -> list[dict]:
    """[{trajectory_id, event_ids, links}], streamed line by line. `links` is
    kept (not just event_ids) so analyze_merges can look up the ACTUAL fused
    LinkEvidence (plate_lr/appearance_lr/kinematic_lr/total_log_odds) the
    solver used for each wrongly-made link -- direct evidence of WHY that
    link won, not just a guess reconstructed from ground truth alone."""
    trajs: list[dict] = []
    with traj_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            trajs.append(
                {
                    "trajectory_id": d["trajectory_id"],
                    "event_ids": d["event_ids"],
                    "links": d.get("links", []),
                }
            )
    return trajs


def _classify_merge_pair(
    id_a: str,
    v_a: dict | None,
    id_b: str,
    v_b: dict | None,
    pa_a: str | None = None,
    pa_b: str | None = None,
) -> str:
    """Classify one wrongly-joined (vehicle_a, vehicle_b) pair. `pa_a`/`pa_b`
    are the OBSERVED (possibly OCR-corrupted) plate_argmax strings actually
    read at the two transition events -- distinct from `true_plate`, which
    is the ground-truth plate neither the engine nor this classifier would
    see as "the same string" unless OCR noise made it so. A pair whose TRUE
    plates are unrelated but whose two OBSERVED reads at the merge point
    happen to look alike is exactly the case where OCR noise, not a
    structural plate ambiguity, created the false plate-channel agreement
    that let the link through."""
    if v_a is None or v_b is None:
        return "other"
    if v_a["true_plate"] == v_b["true_plate"]:
        return "clone_pair"
    if (v_a.get("is_clone") and v_a.get("clone_of") == id_b) or (
        v_b.get("is_clone") and v_b.get("clone_of") == id_a
    ):
        return "clone_pair"
    if edit_distance(v_a["true_plate"], v_b["true_plate"]) <= NEAR_MISS_MAX_EDIT_DISTANCE:
        return "near_miss_plate"
    if (
        pa_a is not None
        and pa_b is not None
        and edit_distance(pa_a, pa_b) <= NEAR_MISS_MAX_EDIT_DISTANCE
    ):
        return "ocr_argmax_collision"
    if v_a["vehicle_type"] == v_b["vehicle_type"] and v_a["color"] == v_b["color"]:
        return "same_vehicle_type_and_colour"
    return "other"


def analyze_merges(
    trajectories: list[dict],
    event_index: dict[str, dict],
    vehicles: dict,
    mean_candidates_per_event: float | None = None,
) -> dict:
    """`mean_candidates_per_event`, when given (eval/reports/trajectory_metrics.json's
    own `gating.mean_candidates_per_event`, 218.01 for this run), additionally
    checks each wrong link against a KNOWN, currently-active fusion bug: every
    link's `prior_log_odds` is `-log(DEFAULT_EXPECTED_CANDIDATES - 1) = -log(4)
    = -1.386` regardless of how many candidates the gate actually handed that
    event (engine/scoring/fusion.py `prior_log_odds`/`DEFAULT_EXPECTED_CANDIDATES`
    -- FusionModel is never constructed with a non-default
    `expected_candidates_per_window` anywhere in this codebase). With the
    ACTUAL mean pool size the correct prior would be
    `-log(mean_candidates_per_event - 1) ~= -5.38`, roughly 4 nats more
    negative than what every link actually received -- an unearned bonus on
    every candidate, gated or solved. This is a KNOWN, already-identified bug
    being fixed separately (not something this module patches); it is
    reported here as a mean-pool-size APPROXIMATION (not each transition's
    own exact gated-candidate count, which would need re-running gating) of
    how much of the over-merging this pre-fix run's headline numbers reflect
    is attributable to that bug specifically, versus a genuine channel
    failure."""
    category_counts: Counter = Counter()
    hour_counts: Counter = Counter()
    examples: dict[str, list[dict]] = defaultdict(list)
    n_trajectories_with_merge = 0
    vehicles_involved: set[str] = set()

    # Secondary diagnostic (architecture.md §3's plate channel should
    # STRONGLY veto a link between two truly-different plates unless its own
    # evidence was itself degraded at exactly that boundary): did either
    # transition event's OWN observed read differ from its OWN true plate --
    # i.e. was the plate channel's evidence corrupted right where the wrong
    # link was made, independent of the OTHER vehicle's plate entirely.
    misread_at_transition = Counter()  # category -> n with either side misread
    dt_bucket_counts: Counter = Counter()  # category -> Counter(dt bucket)

    n_wrong_links_with_evidence = 0
    n_wrong_links_negative_total = 0  # already negative under the CURRENT (buggy) prior
    n_wrong_links_prior_only_driven = 0  # positive now, but would flip negative under the
    # approximate corrected prior -- i.e. the buggy prior alone is what let it through
    current_prior_default = -math.log(max(DEFAULT_EXPECTED_CANDIDATES - 1.0, 1.0))
    corrected_prior_approx = (
        -math.log(max(mean_candidates_per_event - 1.0, 1.0))
        if mean_candidates_per_event
        else None
    )

    for traj in trajectories:
        links_by_pair = {
            (link["from_event_id"], link["to_event_id"]): link for link in traj.get("links", [])
        }
        rows = [(eid, event_index[eid]) for eid in traj["event_ids"] if eid in event_index]
        rows.sort(key=lambda r: r[1]["epoch"])
        distinct_gts = {info["gt"] for _, info in rows if info["gt"] is not None}
        if len(distinct_gts) <= 1:
            continue
        n_trajectories_with_merge += 1
        vehicles_involved.update(distinct_gts)

        prev_gt: str | None = None
        prev_pa: str | None = None
        prev_epoch: float | None = None
        prev_eid: str | None = None
        for eid, info in rows:
            gt = info["gt"]
            if gt is None:
                continue
            if prev_gt is not None and gt != prev_gt:
                category = _classify_merge_pair(
                    prev_gt, vehicles.get(prev_gt), gt, vehicles.get(gt), prev_pa, info["pa"]
                )
                category_counts[category] += 1
                hour = datetime.fromisoformat(info["t"]).hour
                hour_counts[hour] += 1

                true_plate_a = (vehicles.get(prev_gt) or {}).get("true_plate")
                true_plate_b = (vehicles.get(gt) or {}).get("true_plate")
                either_misread = prev_pa != true_plate_a or info["pa"] != true_plate_b
                if either_misread:
                    misread_at_transition[category] += 1

                dt = info["epoch"] - prev_epoch if prev_epoch is not None else None
                if dt is not None:
                    bucket = "<=30s" if dt <= 30 else ("30-120s" if dt <= 120 else ">120s")
                    dt_bucket_counts[f"{category}:{bucket}"] += 1

                link = links_by_pair.get((prev_eid, eid))
                prior_diagnostics: dict = {}
                if link is not None:
                    n_wrong_links_with_evidence += 1
                    total = link["total_log_odds"]
                    if total < 0:
                        n_wrong_links_negative_total += 1
                    prior_diagnostics = {
                        "plate_lr": link["plate_lr"],
                        "appearance_lr": link["appearance_lr"],
                        "kinematic_lr": link["kinematic_lr"],
                        "total_log_odds": link["total_log_odds"],
                    }
                    if corrected_prior_approx is not None:
                        channel_sum = total - current_prior_default
                        recomputed_total = channel_sum + corrected_prior_approx
                        prior_only_driven = total >= 0 and recomputed_total < 0
                        if prior_only_driven:
                            n_wrong_links_prior_only_driven += 1
                        prior_diagnostics["recomputed_total_with_corrected_prior_approx"] = (
                            recomputed_total
                        )
                        prior_diagnostics["prior_only_driven"] = prior_only_driven

                if len(examples[category]) < MAX_EXAMPLES_PER_CATEGORY:
                    examples[category].append(
                        {
                            "trajectory_id": traj["trajectory_id"],
                            "vehicle_a": prev_gt,
                            "vehicle_b": gt,
                            "plate_a": true_plate_a,
                            "plate_b": true_plate_b,
                            "observed_plate_a": prev_pa,
                            "observed_plate_b": info["pa"],
                            "either_side_misread": either_misread,
                            "dt_s": dt,
                            "hour": hour,
                            "timestamp": info["t"],
                            **prior_diagnostics,
                        }
                    )
            prev_gt = gt
            prev_pa = info["pa"]
            prev_epoch = info["epoch"]
            prev_eid = eid

    n_transitions = sum(category_counts.values())
    return {
        "n_predicted_trajectories": len(trajectories),
        "n_trajectories_containing_a_merge": n_trajectories_with_merge,
        "n_true_vehicles_involved_in_merges": len(vehicles_involved),
        "n_wrong_adjacent_transitions": n_transitions,
        "category_counts": dict(category_counts),
        "category_fraction": (
            {k: v / n_transitions for k, v in category_counts.items()} if n_transitions else {}
        ),
        "top_category": category_counts.most_common(1)[0][0] if category_counts else None,
        "hour_of_day_counts": dict(sorted(hour_counts.items())),
        "either_side_misread_count_by_category": dict(misread_at_transition),
        "either_side_misread_fraction_by_category": {
            k: misread_at_transition.get(k, 0) / v for k, v in category_counts.items()
        },
        "dt_bucket_counts_by_category": dict(dt_bucket_counts),
        "prior_bug_diagnostics": {
            "note": (
                "KNOWN BUG (being fixed separately, not by this module): every link's "
                "prior_log_odds is a FIXED -log(DEFAULT_EXPECTED_CANDIDATES-1)=-log(4)="
                f"{current_prior_default:.4f} regardless of the gate's actual candidate "
                "pool size. This run's true mean pool size was "
                f"{mean_candidates_per_event} candidates/event "
                "(eval/reports/trajectory_metrics.json gating.mean_candidates_per_event), "
                f"implying a corrected prior of roughly {corrected_prior_approx:.4f} -- "
                "an APPROXIMATION using the mean pool size, not each transition's own "
                "exact gated-candidate count (that would need re-running gating)."
                if corrected_prior_approx is not None
                else "mean_candidates_per_event not supplied; prior-bug diagnostics skipped."
            ),
            "current_fixed_prior_log_odds": current_prior_default,
            "corrected_prior_log_odds_approx": corrected_prior_approx,
            "n_wrong_links_with_evidence": n_wrong_links_with_evidence,
            "n_wrong_links_negative_total_log_odds": n_wrong_links_negative_total,
            "share_wrong_links_negative_total_log_odds": (
                n_wrong_links_negative_total / n_wrong_links_with_evidence
                if n_wrong_links_with_evidence
                else None
            ),
            "n_wrong_links_prior_only_driven": n_wrong_links_prior_only_driven,
            "share_wrong_links_prior_only_driven": (
                n_wrong_links_prior_only_driven / n_wrong_links_with_evidence
                if n_wrong_links_with_evidence
                else None
            ),
            "n_wrong_links_currently_accepted": (
                n_wrong_links_with_evidence - n_wrong_links_negative_total
            ),
            "share_of_currently_accepted_that_are_prior_only_driven": (
                n_wrong_links_prior_only_driven
                / (n_wrong_links_with_evidence - n_wrong_links_negative_total)
                if (n_wrong_links_with_evidence - n_wrong_links_negative_total)
                else None
            ),
            "share_wrong_links_explained_by_negative_total_or_prior_bug": (
                (n_wrong_links_negative_total + n_wrong_links_prior_only_driven)
                / n_wrong_links_with_evidence
                if n_wrong_links_with_evidence
                else None
            ),
        },
        "examples": {k: v for k, v in examples.items()},
    }


def analyze_splits(
    trajectories: list[dict],
    event_index: dict[str, dict],
    ground_truth_journeys: dict,
    vehicles: dict,
) -> dict:
    event_to_traj: dict[str, str] = {}
    for traj in trajectories:
        for eid in traj["event_ids"]:
            event_to_traj[eid] = traj["trajectory_id"]

    by_vehicle: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for eid, info in event_index.items():
        if info["gt"] is not None:
            by_vehicle[info["gt"]].append((eid, info))

    all_epochs = [info["epoch"] for info in event_index.values()]
    t_min, t_max = min(all_epochs), max(all_epochs)
    window_starts = [
        b[0] for b in _window_bounds(t_min, t_max, DEFAULT_WINDOW_SIZE_S, DEFAULT_WINDOW_STEP_S)
    ]
    # Every point a window boundary is crossed: window starts, PLUS the
    # start-of-overlap-zone points solve_windowed actually decides carry-over
    # at (window_end - overlap_s == window_start + step_s, i.e. the NEXT
    # window's own start) -- window_starts already covers both since step_s
    # < window_size_s means consecutive window starts are exactly the
    # carry-over decision points.
    window_boundaries = window_starts

    cause_counts: Counter = Counter()
    examples: dict[str, list[dict]] = defaultdict(list)
    n_gt_vehicles_split = 0
    n_split_breaks = 0

    for gt_id, evs in by_vehicle.items():
        evs.sort(key=lambda r: r[1]["epoch"])
        distinct_trajs = {event_to_traj[eid] for eid, _ in evs if eid in event_to_traj}
        if len(distinct_trajs) <= 1:
            continue
        n_gt_vehicles_split += 1

        true_plate = (vehicles.get(gt_id) or {}).get("true_plate")
        passages = ground_truth_journeys.get(gt_id, {}).get("passages", [])
        passage_epochs = [datetime.fromisoformat(p["timestamp"]).timestamp() for p in passages]
        detected_epochs = {info["epoch"] for _, info in evs}

        prev_eid, prev_info = evs[0]
        for eid, info in evs[1:]:
            if event_to_traj.get(prev_eid) != event_to_traj.get(eid):
                n_split_breaks += 1
                a_misread = true_plate is not None and prev_info["pa"] != true_plate
                b_misread = true_plate is not None and info["pa"] != true_plate
                lo, hi = prev_info["epoch"], info["epoch"]
                cause = "other"
                if any(
                    lo < p < hi and p not in detected_epochs for p in passage_epochs
                ):
                    cause = "missed_detection_gap"
                elif a_misread or b_misread:
                    cause = "plate_misread_at_break"
                elif any(
                    lo - WINDOW_SEAM_TOLERANCE_S <= b <= hi + WINDOW_SEAM_TOLERANCE_S
                    for b in window_boundaries
                ):
                    cause = "window_seam"
                cause_counts[cause] += 1
                if len(examples[cause]) < MAX_EXAMPLES_PER_CATEGORY:
                    examples[cause].append(
                        {
                            "gt_vehicle_id": gt_id,
                            "true_plate": true_plate,
                            "from_event": prev_eid,
                            "to_event": eid,
                            "from_trajectory": event_to_traj.get(prev_eid),
                            "to_trajectory": event_to_traj.get(eid),
                            "gap_s": hi - lo,
                        }
                    )
            prev_eid, prev_info = eid, info

    n_gt_vehicles = len(by_vehicle)
    return {
        "n_gt_vehicles": n_gt_vehicles,
        "n_gt_vehicles_split": n_gt_vehicles_split,
        "n_split_breaks": n_split_breaks,
        "cause_counts": dict(cause_counts),
        "cause_fraction": (
            {k: v / n_split_breaks for k, v in cause_counts.items()} if n_split_breaks else {}
        ),
        "top_cause": cause_counts.most_common(1)[0][0] if cause_counts else None,
        "examples": {k: v for k, v in examples.items()},
    }


def build_report(data_dir: Path = DATA_DIR, pipeline_dir: Path = PIPELINE_DIR) -> dict:
    ground_truth = json.loads((data_dir / "ground_truth.json").read_text(encoding="utf-8"))
    vehicles = ground_truth["vehicles"]
    journeys = ground_truth["journeys"]

    event_index = _load_event_index(data_dir / "events.jsonl")
    trajectories = _load_trajectories(pipeline_dir / "trajectories.jsonl")

    trajectory_metrics_path = Path("eval/reports/trajectory_metrics.json")
    mean_candidates_per_event = None
    if trajectory_metrics_path.exists():
        tm = json.loads(trajectory_metrics_path.read_text(encoding="utf-8"))
        mean_candidates_per_event = tm.get("gating", {}).get("mean_candidates_per_event")

    merges = analyze_merges(trajectories, event_index, vehicles, mean_candidates_per_event)
    splits = analyze_splits(trajectories, event_index, journeys, vehicles)

    pbd = merges["prior_bug_diagnostics"]
    share_negative = pbd["share_wrong_links_negative_total_log_odds"]
    share_accepted_prior_driven = pbd["share_of_currently_accepted_that_are_prior_only_driven"]
    share_explained = pbd["share_wrong_links_explained_by_negative_total_or_prior_bug"]

    headline = (
        f"UrbanTrace predicts {merges['n_predicted_trajectories']} trajectories for "
        f"{splits['n_gt_vehicles']} true vehicles (over-merging). Of "
        f"{merges['n_wrong_adjacent_transitions']} wrongly-made adjacent links across "
        f"{merges['n_trajectories_containing_a_merge']} merged trajectories "
        f"({merges['n_true_vehicles_involved_in_merges']} true vehicles involved), the "
        f"dominant cause is '{merges['top_category']}' "
        f"({merges['category_fraction'].get(merges['top_category'], 0.0):.1%} of transitions). "
        f"Of the wrong links with recorded evidence, {share_negative:.1%} already have a "
        "NEGATIVE total_log_odds even under the current (buggy, over-generous) prior -- "
        "meaning the solver chose the least-bad available option, not a genuinely confident "
        f"wrong link -- and {share_accepted_prior_driven:.1%} of the REMAINING "
        "currently-accepted (total_log_odds>=0) wrong links would flip negative under the "
        f"corrected mean-pool-size prior alone: altogether an approximate {share_explained:.1%} "
        "of all wrongly-made links are either already a forced least-bad choice or a direct "
        "artifact of the fixed-prior bug, not a genuine 3-channel evidence failure."
        if merges["n_wrong_adjacent_transitions"]
        and merges["prior_bug_diagnostics"]["n_wrong_links_with_evidence"]
        else "No wrongly-joined adjacent transitions with recorded link evidence found."
    ) + (
        f" Separately, {splits['n_gt_vehicles_split']} true vehicles are fragmented across "
        f">1 predicted trajectory ({splits['n_split_breaks']} breaks total); the dominant "
        f"identifiable cause is '{splits['top_cause']}' "
        f"({splits['cause_fraction'].get(splits['top_cause'], 0.0):.1%} of breaks)."
        if splits["n_split_breaks"]
        else " No split-causing breaks found."
    )

    return {
        "description": (
            "Error analysis of the ALREADY-SOLVED full-city UrbanTrace run, measured with the "
            "FIXED prior of 5 expected candidates (pre-fix) -- engine/scoring/fusion.py's "
            "prior_log_odds is currently -log(DEFAULT_EXPECTED_CANDIDATES-1)=-log(4) for "
            "EVERY link regardless of the gate's actual ~218 candidates/event, a known bug "
            "being fixed separately (see merges.prior_bug_diagnostics below); these numbers "
            "will change once that fix lands and this report is re-run. "
            f"Data: {data_dir} ({len(event_index)} events, {len(trajectories)} predicted "
            f"trajectories, {splits['n_gt_vehicles']} true vehicles; matches "
            "eval/reports/trajectory_metrics.json's headline). Computed by reading "
            f"{pipeline_dir / 'trajectories.jsonl'}, {data_dir / 'ground_truth.json'} and "
            f"{data_dir / 'events.jsonl'} -- no re-solving."
        ),
        "data_dir": str(data_dir),
        "pipeline_dir": str(pipeline_dir),
        "merges": merges,
        "splits": splits,
        "headline_answer": headline,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_PATH,
        help=(
            "Output path for the JSON report (default: eval/reports/error_analysis.json). "
            "Pass a distinct path (e.g. eval/reports/error_analysis_postfix.json) to compare "
            "a re-run against a prior report without overwriting it."
        ),
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR, help="On-disk dataset directory."
    )
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=PIPELINE_DIR,
        help="Directory containing the solved trajectories.jsonl to analyze.",
    )
    args = parser.parse_args(argv)

    report = build_report(data_dir=args.data_dir, pipeline_dir=args.pipeline_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    top_level = {k: v for k, v in report.items() if k not in ("merges", "splits")}
    merges_summary = {k: v for k, v in report["merges"].items() if k != "examples"}
    splits_summary = {k: v for k, v in report["splits"].items() if k != "examples"}
    print(json.dumps(top_level, indent=2))
    print(json.dumps({"merges_summary": merges_summary}, indent=2))
    print(json.dumps({"splits_summary": splits_summary}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
