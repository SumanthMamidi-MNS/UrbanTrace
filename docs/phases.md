# Execution plan — SUTRA (SIH26127)

Living plan, organised by what the PRD requires, not by calendar. Each workstream lists what is done, what remains, and what "done" means. Detailed history is in `decisions.md` (why) and `memory.md` (what was built when).

The PRD asks for **four components**. This plan is organised around them, plus the evaluation and packaging that let us defend them.

---

## Status at a glance

| PRD component | Status |
|---|---|
| 1. High-precision OCR module (deep learning, >90% on real Indian plates) | **Built and measured on real plates: 81.0% whole-plate / 94.3% character** — >90% met per character, **not** per whole plate. Detector: held-out video recall 0.51 (small far plates are the main weakness) |
| 2. Trajectory reconstruction engine (query a plate → chronological path on a GIS map) | Built and verified; full-city IDF1 **0.9914** vs 0.8746. Direction of travel in progress |
| 3. City traffic analytics dashboard (heatmaps, speeds, densities, OD, congestion, real time) | Partly built — heatmap, speeds and flow trend in progress (contract v2) |
| 4. Alert system (blacklisted vehicles + suspicious routes, real time) | Partly built — probabilistic watchlist in progress (contract v2) |
| Evaluation & evidence | Built; kinematic defect fixed; link threshold calibrated on a training day |
| Packaging & demo | Built; Docker image never actually built |

Last checkpoint commit: `3761df9` (local only, not pushed). Substantial uncommitted work since — next commit after the v2 features land and tests pass.

---

## W1. OCR module — PRD component 1

**Goal:** a deep-learning plate reader that emits **per-character probability distributions** (which the linking engine already consumes via the `DetectionEvent` contract), with recognition accuracy **measured on real Indian plates** and exceeding 90%.

**Done:** real labelled data from the team (1,587 real crops / 904 plates, split by plate); downloads approved; plate detector fine-tuned on Indian scenes; synthetic-pretrained CRNN (baseline, 44.3% real); **fast-plate-ocr fine-tuned: 81.0% whole-plate / 94.3% character / ECE 0.041 on the held-out real test set**; video → `DetectionEvent` adapter smoke-tested into the engine.

**Remaining:**
- [ ] **Whole-plate accuracy is 81.0%, below the PRD's 90%** (character level 94.3% is above). Next lever: more real training plates — the team is sourcing the ~16k-image "in the wild" dataset.
- [ ] Small, distant plates in multi-lane footage: detector recall 0.51 on a held-out video.
- [ ] Wire the fine-tuned fast-plate-ocr model into `video_to_events.py` as the default reader (currently the CRNN path).

**Done means:** measured whole-plate accuracy >90% on a held-out, human-verified real Indian plate set, reported with its size and conditions; a sample clip runs end to end into the console.

## W2. Trajectory reconstruction — PRD component 2

**Done:** probabilistic three-channel linking, global min-cost-flow association with exact pruning, consensus plate repair, partial-plate search, trajectory detail on the map with timestamps and cameras, WHY panel, kinematic null fixed, link threshold calibrated on a training day. Full-city IDF1 **0.9914** vs 0.8746 for exact matching.

**Remaining:**
- [x] Kinematic null fixed — ablation: plate+kinematic 0.977 > plate-only 0.968; all three 0.992 is the best row.
- [ ] Show direction of travel on the trajectory map (the PRD names "direction" explicitly).

**Done means:** in the ablation, every added channel improves or holds IDF1, and "all three" is the best row.

## W3. Traffic analytics dashboard — PRD component 3

**Done:** OD matrix (5 zones), volumes per camera over time, corridor travel times with congestion index, KPI summary.

**Remaining:**
- [ ] **Traffic density heatmap** on the map, including a **live** mode driven by the replay stream.
- [ ] **Average vehicle speeds** per corridor and per camera, from trajectory link distances and times.
- [ ] Traffic-flow trend view (volumes over time across the network) and congestion bottleneck ranking surfaced on the dashboard.

**Done means:** every analytic named in the PRD — heatmap, average speeds, route densities, OD, congestion bottlenecks, flow trends — is visible in the console from real API data.

## W4. Alert system — PRD component 4

**Done:** cloned-plate, impossible-travel and looping-route alerts, streamed live.

**Remaining:**
- [ ] **Blacklist / watchlist:** operators add plates (full or partial with `?`); matching uses the plate posterior, so a watchlisted vehicle is still caught when its plate is misread. Alerts fire in real time as reads arrive, with match confidence.
- [ ] Watchlist management in the console (add, remove, list, see hits).

**Done means:** a watchlisted plate raises a real-time alert during replay, including on a read with a misread character.

## W5. Evaluation & evidence

**Done:** stratified pairwise AUC, full-city IDF1 vs exact matching, stress sweep (our lead grows from 0.094 to 0.427 as per-read plate accuracy falls from 88.8% to 50%), baselines A/B/C, ablation, error analysis, consensus accuracy, gating and blocking reports, 255 tests.

**Remaining:**
- [x] Ablation and full city rerun after the kinematic fix.
- [x] OCR accuracy reports (synthetic, real progression, fast-plate-ocr) and detector reports.
- [ ] Results page shows ablation, baselines and stress sweep.

## W6. Packaging & demo

**Done:** one-command local serve (verified), docker-compose + Dockerfile, Makefile and `make.ps1`, README with results, demo script, judge Q&A.

**Remaining:**
- [ ] Actually build and run the Docker image on a machine with a working Docker daemon.
- [ ] Refresh the demo script and README for W1–W4 additions.
- [ ] Final full test pass, lint, commit.

---

## Order of work

1. W2 kinematic fix (correctness; affects every headline number)
2. W4 watchlist alerts and W3 heatmap + speeds (PRD gaps that need no downloads)
3. W1 OCR (as soon as the user decides on data and approves downloads)
4. W5 refresh, W6 refresh, final commit

## Working rules for this project

- Nothing is tuned on the evaluation set (run1, seed 42); training and tuning use separate seeds.
- The SQLite database lives outside OneDrive (`%LOCALAPPDATA%\sutra\sutra.db`).

## Progress log (2026-09-25)

- Kinematic channel fixed (empirical null); link threshold β=2 calibrated on a full-size training day, transferred to the test day: full-city IDF1 **0.9914** vs exact-match 0.8746.
- OCR: synthetic renderer hardened (visibility guarantee, partial occlusion, HSRP features); team-supplied real dataset organised and split by plate; our CRNN reached 44.3% on real plates and plateaued; **fast-plate-ocr fine-tuned on real plates: 81.0% whole-plate / 94.3% character on the held-out real test set.**
- Detector: RGB bug fixed; crop-images excluded from evaluation; fine-tuned detector on held-out video recall 0.333 → 0.506.
- Contract v2 written (watchlist, heatmap, speeds, flow trend, direction); API and UI built against it in parallel.
- Remaining: land contract v2 (W2 direction, W3, W4), refresh README/Results/demo, full test pass, commit. Optional: the 16k-image in-the-wild dataset for OCR, if the team obtains it.
