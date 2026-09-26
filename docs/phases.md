# Execution plan — UrbanTrace (SIH26127)

Living plan, organised by what the PRD requires, not by calendar. Each workstream lists what is done, what remains, and what "done" means. Detailed history is in `decisions.md` (why) and `memory.md` (what was built when).

The PRD asks for **four components**. This plan is organised around them, plus the evaluation and packaging that let us defend them.

---

## Status at a glance

| PRD component | Status |
|---|---|
| 1. High-precision OCR module (deep learning, >90% on real Indian plates) | **Built and measured on real plates: 81.0% whole-plate / 94.3% character** — >90% met per character, **not** per whole plate. Fine-tuned reader is the video path default. Detector: held-out video recall 0.51 |
| 2. Trajectory reconstruction engine | **Done** — congested full-city IDF1 **0.9716** vs 0.8745 (uncongested 0.9914); direction of travel on the map |
| 3. City traffic analytics dashboard | **Done** — heatmaps (density/speed, live), corridor speeds, OD, volumes, flow trend, bottleneck ranking |
| 4. Alert system | **Done** — probabilistic watchlist (single read + trajectory consensus), clone, impossible-travel, route anomalies, live |
| Evaluation & evidence | **Done** — rerun on the congested day; gate recall hole found and fixed |
| Packaging & GitHub readiness | **Done** — portfolio README with flowchart and screenshots, MIT licence, CI, 329 MB Docker image verified end to end, published to GitHub |


Published to GitHub (`main`): https://github.com/SumanthMamidi-MNS/UrbanTrace

---

## W1. OCR module — PRD component 1

**Goal:** a deep-learning plate reader that emits **per-character probability distributions** (which the linking engine already consumes via the `DetectionEvent` contract), with recognition accuracy **measured on real Indian plates** and exceeding 90%.

**Done:** real labelled data from the team (1,587 real crops / 904 plates, split by plate); downloads approved; plate detector fine-tuned on Indian scenes; synthetic-pretrained CRNN (baseline, 44.3% real); **fast-plate-ocr fine-tuned: 81.0% whole-plate / 94.3% character / ECE 0.041 on the held-out real test set**; video → `DetectionEvent` adapter smoke-tested into the engine.

**Remaining:**
- [ ] **Whole-plate accuracy is 81.0%, below the PRD's 90%** (character level 94.3% is above). Next lever: more real training plates — the team is sourcing the ~16k-image "in the wild" dataset.
- [ ] Small, distant plates in multi-lane footage: detector recall 0.51 on a held-out video.
- [x] Fine-tuned fast-plate-ocr is the default reader in `video_to_events.py` (CRNN kept as `--reader crnn`).

**Done means:** measured whole-plate accuracy >90% on a held-out, human-verified real Indian plate set, reported with its size and conditions; a sample clip runs end to end into the console.

## W2. Trajectory reconstruction — PRD component 2

**Done:** probabilistic three-channel linking, global min-cost-flow association with exact pruning, consensus plate repair, partial-plate search, trajectory detail on the map with timestamps and cameras, WHY panel, kinematic null fixed, link threshold calibrated on a training day. Full-city IDF1 **0.9914** vs 0.8746 for exact matching.

**Remaining:**
- [x] Kinematic null fixed — ablation: plate+kinematic 0.977 > plate-only 0.968; all three 0.992 is the best row.
- [x] Show direction of travel on the trajectory map (the PRD names "direction" explicitly).

**Done means:** in the ablation, every added channel improves or holds IDF1, and "all three" is the best row.

## W3. Traffic analytics dashboard — PRD component 3

**Done:** OD matrix (5 zones), volumes per camera over time, corridor travel times with congestion index, KPI summary.

**Remaining:**
- [x] **Traffic density heatmap** on the map, including a **live** mode driven by the replay stream.
- [x] **Average vehicle speeds** per corridor and per camera, from trajectory link distances and times.
- [x] Traffic-flow trend view (volumes over time across the network) and congestion bottleneck ranking surfaced on the dashboard.

**Done means:** every analytic named in the PRD — heatmap, average speeds, route densities, OD, congestion bottlenecks, flow trends — is visible in the console from real API data.

## W4. Alert system — PRD component 4

**Done:** cloned-plate, impossible-travel and looping-route alerts, streamed live.

**Remaining:**
- [x] **Blacklist / watchlist:** operators add plates (full or partial with `?`); matching uses the plate posterior, so a watchlisted vehicle is still caught when its plate is misread. Alerts fire in real time as reads arrive, with match confidence.
- [x] Watchlist management in the console (add, remove, list, see hits).

**Done means:** a watchlisted plate raises a real-time alert during replay, including on a read with a misread character.

## W5. Evaluation & evidence

**Done:** stratified pairwise AUC, full-city IDF1 vs exact matching, stress sweep (our lead grows from 0.094 to 0.427 as per-read plate accuracy falls from 88.8% to 50%), baselines A/B/C, ablation, error analysis, consensus accuracy, gating and blocking reports, 255 tests.

**Remaining:**
- [x] Ablation and full city rerun after the kinematic fix.
- [x] OCR accuracy reports (synthetic, real progression, fast-plate-ocr) and detector reports.
- [x] Results page shows ablation, baselines and stress sweep.

## W6. Packaging, demo & GitHub readiness

**Done:** one-command local serve (verified), Makefile and `make.ps1`, demo script and judge Q&A refreshed for the congested run, rename to **UrbanTrace** with portable paths (`engine/paths.py`), portfolio README front page (tagline, PRD components, results at a glance, mermaid architecture), MIT licence, slim runtime Docker image (runtime deps only), CI workflow, vendored OFL font so tests pass on a fresh clone, demo DB re-ingested clean.

**Remaining:**
- [x] Docker image built and run end to end: 329 MB, seed + serve verified.
- [x] README screenshots from the real API, plus a pipeline flowchart.
- [x] Published to GitHub on `main`.
- [ ] Owner: set the GitHub About text and topics.

**Done means:** a fresh clone passes CI, the quickstart works from the README alone, and the image builds.

---

## Order of work

1. W2 kinematic fix (correctness; affects every headline number)
2. W4 watchlist alerts and W3 heatmap + speeds (PRD gaps that need no downloads)
3. W1 OCR (as soon as the user decides on data and approves downloads)
4. W5 refresh, W6 refresh, final commit

## Working rules for this project

- Nothing is tuned on the evaluation set (run1, seed 42); training and tuning use separate seeds.
- The SQLite database lives outside OneDrive (`URBANTRACE_DB_PATH`, default `%LOCALAPPDATA%\urbantrace\urbantrace.db`).

## Progress log (2026-09-25)

- Kinematic channel fixed (empirical null); link threshold β=2 calibrated on a full-size training day, transferred to the test day: full-city IDF1 **0.9914** vs exact-match 0.8746.
- OCR: synthetic renderer hardened (visibility guarantee, partial occlusion, HSRP features); team-supplied real dataset organised and split by plate; our CRNN reached 44.3% on real plates and plateaued; **fast-plate-ocr fine-tuned on real plates: 81.0% whole-plate / 94.3% character on the held-out real test set.**
- Detector: RGB bug fixed; crop-images excluded from evaluation; fine-tuned detector on held-out video recall 0.333 → 0.506.
- Contract v2 written (watchlist, heatmap, speeds, flow trend, direction); API and UI built against it in parallel.
- Remaining: land contract v2 (W2 direction, W3, W4), refresh README/Results/demo, full test pass, commit. Optional: the 16k-image in-the-wild dataset for OCR, if the team obtains it.
