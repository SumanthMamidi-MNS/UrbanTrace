# UrbanTrace — detailed results

Every number below traces to a file in `eval/reports/`. Nothing is tuned on what it is scored on: priors, the link threshold and the congestion model are calibrated on separate simulated days with different random seeds, and the OCR is scored once on real plates held out by plate string.

## 1. OCR on real Indian plates (`ocr_real_fpo_finetuned.json`) — PRD component 1

Held-out real test set: **510 images / 276 unique plates**, never seen in training (the team's real Indian dataset, split by plate so no car appears in both training and test).

| Model | Whole plate | Per character | Per unique plate |
|---|---|---|---|
| Our own CRNN, synthetic training only | 9.2% | 51.5% | 8.0% |
| Our own CRNN, + real fine-tune (60 epochs) | 44.3% | 73.8% | 40.2% |
| fast-plate-ocr pretrained, no fine-tune (no Indian plates in its training) | 51.4% | 75.9% | — |
| **fast-plate-ocr fine-tuned on real Indian plates** | **81.0%** | **94.3%** | **77.2%** |

By source: video frames 89.1%, OLX photos 78.8%, Google images 73.0%. Calibration error (ECE) 0.041.

**Against the PRD's ">90%": met per character (94.3%), not met per whole plate (81.0%).** We report both. The gap is training data — 546 unique real training plates; validation reached 88%.

## 2. Plate detection (`detector_holdout_video.json`, `detector_eval.json`)

| | Recall | mAP@0.5 |
|---|---|---|
| Close-up vehicle photos (pretrained, after an RGB-order fix) | 0.946 | 0.960 |
| **Held-out traffic video, small distant plates — pretrained** | 0.333 | 0.308 |
| **Held-out traffic video — fine-tuned on Indian scenes** | **0.506** | **0.530** |

Small, distant plates in general traffic footage are the system's main real-world weakness. The held-out video is small (50 frames, 35 plates) — indicative, not precise.

## 3. Trajectory reconstruction (`trajectory_metrics.json`) — PRD component 2

A simulated city day **with realistic rush-hour congestion**: 50 cameras, 20,000 vehicles, 24 hours, **103,475 camera reads**.

| | **UrbanTrace** | Baseline A (exact plate match) |
|---|---|---|
| **IDF1** | **0.9716** | 0.8745 |
| Identity switches | **2,143** | 19,214 |
| Fragmentation | **1,540** | — |
| Trajectory completeness | **0.982** | — |
| Journeys found (truth: 19,996) | **21,218** | 32,475 |
| Linking time on a laptop | 12 min | — |

Link threshold β = 5, chosen on a separate full-size congested training day (IDF1 there 0.969 — it transferred). On an earlier **uncongested** day the same engine scored **0.9914** ([`trajectory_metrics_run1.json`](../eval/reports/trajectory_metrics_run1.json)); congestion makes travel time less informative, so the honest number is the congested one.

## 4. City analytics on the congested day — PRD component 3

City mean speed falls from **80 km/h at night to 65.3 km/h at 08:00 and 66.0 km/h at 17:00**; the worst corridors average 71–80% of free-flow speed over the day (congestion index 1.20–1.31). Heatmaps (density and speed), OD matrix, volumes, flow trends and a bottleneck ranking are all served live by the API.

## 5. Alerts — PRD component 4

On the congested day: **197 cloned-plate alerts**, 1 impossible-travel, 2 looping-route anomalies. Example clone: plate `KA98LM9842` seen 7.8 km apart in 139 s, when the road network needs at least 235 s — implied 203 km/h, and the two vehicles look different.

**Watchlist:** matching is probabilistic, on each read and on the vehicle's fused plate. Plate `TN15TU5117` was caught at 93% via the fused multi-camera plate at a camera whose own read did not clear the threshold alone.

## 6. Robustness — the gap grows as plates get harder to read (`stress_sweep.json`)

| Per-read plate accuracy | UrbanTrace IDF1 | Exact match IDF1 | Gap |
|---|---|---|---|
| 88.8% | 0.979 | 0.885 | 0.094 |
| 70.4% | 0.979 | 0.717 | 0.262 |
| 63.8% | 0.976 | 0.657 | 0.319 |
| 55.2% | 0.986 | 0.589 | 0.398 |
| **50.0%** | **0.980** | **0.553** | **0.427** |

When cameras **miss** vehicles instead (0% → 30%), UrbanTrace stays at 0.97–0.99 with a steady lead of ~0.09–0.11, but the gap does **not** widen — the widening is specific to plate-reading errors. *(Run on smaller uncongested cities.)*

## 7. Baselines, ablation and fusion by case

*(These use smaller uncongested cities, to isolate each effect.)*

| System (`baselines.json`) | IDF1 |
|---|---|
| A — exact plate match | 0.875 |
| B — fuzzy match (≤ 1 character different) | 0.952 |
| C — fuzzy + **hard** travel-time window | 0.863 |

Baseline C is *worse* than B: a hard travel-time rule rejects genuine links whenever a vehicle dawdles.

| Evidence used (`ablation.json`) | IDF1 |
|---|---|
| Plate only | 0.968 |
| Plate + travel time | 0.977 |
| Plate + appearance | 0.987 |
| **All three** | **0.992** |
| All three, greedy linking instead of the global solver | 0.991 |
| Appearance + travel time, **no plate** | 0.388 |

Every channel adds value. The global solver's accuracy margin over greedy linking is small once channels are calibrated — its value is the guarantee that one read belongs to exactly one vehicle. Appearance and travel time support the plate; they can't replace it.

| Case (`stratified_auc.json`, `clone_overlap_auc.json`) | Plate only | Appearance | Travel time | **Fused** |
|---|---|---|---|---|
| Routine traffic | 1.000 | 0.998 | 0.860 | **1.000** |
| **Cloned plates** | **0.477** | 0.997 | 0.960 | **0.998** |
| Near-identical plates | 0.923 | 0.998 | 1.000 | **1.000** |

With two vehicles sharing one plate string, plate evidence alone is a coin flip; fusion still separates them.

| Camera reads fused (`consensus_accuracy.json`) | Single read | **After fusion** |
|---|---|---|
| 3 | 87.8% | **98.3%** |
| 4 or more | 87.8% | **99.95%** |

## Honest limitations

- **OCR whole-plate accuracy is 81.0%**, below the PRD's 90% (per character 94.3% is above). More real training plates is the lever.
- **Small, distant plates in general traffic footage:** detector recall 0.51 on a held-out video. ANPR-positioned cameras would help most.
- **Tracking over-splits:** 21,218 journeys for 19,996 vehicles on the congested day. We prefer that to over-merging, which would invent journeys that never happened.
- **The tracking evaluation is simulated**, pinned to published real-world figures (per-read plate accuracy 85–95%, appearance re-identification ~70% at benchmark scale, rush-hour bottlenecks at ~40–60% of free-flow speed) with tests that fail if the noise drifts. Every simulated trip starts and ends at the city boundary, so the grid interior carries little traffic.
- **Licensing:** the detector library (Ultralytics) is AGPL-3.0 — fine for a prototype; production needs a commercial licence or an Apache-licensed detector.
- **The Docker image has never been built** — the daemon would not start on the development machine. The non-Docker quickstart is verified end to end.

## Where the evidence lives

`eval/reports/*.json` — generated by the scripts in `eval/`, and served verbatim (keyed by file stem) by
`GET /api/eval`, which the UI's **Results** page renders:

| File | What it shows |
|---|---|
| `trajectory_metrics.json` | The headline scoreboard: UrbanTrace vs. Baseline A (exact plate-string match, same spatio-temporal gate) — IDF1, ID-precision/recall, ID-switches, fragmentation, trajectory completeness (from `eval/run_pipeline.py`) |
| `trajectory_metrics_prefix.json` | The same, on a time-prefix of the dataset (`--max-hours`) — a quick timing/sanity check before committing to a full run |
| `stratified_auc.json` | The stratum x channel AUC matrix (architecture.md §8): plate-only vs. fused, across the *routine*, *clone*, *degraded*, and *plate-similar* strata — the load-bearing evidence that fusion, not just the plate channel, is doing the work |
| `clone_overlap_auc.json` | Clone-detection performance split by route-overlap difficulty |
| `consensus_accuracy.json` | Plate-repair accuracy: consensus decode vs. single reads |
| `appearance_scaling.json` | How the appearance (Re-ID) channel's discriminative power scales |
| `gating.json` / `blocking.json` | Candidate-recall and pruning stats for the spatio-temporal gate (and the plate-based overflow blocker) |
| `error_analysis.json` | Breakdown of the pipeline's remaining failure modes |
