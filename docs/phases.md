# Phases — SUTRA (SIH26127)

One week total. Two tracks run in parallel from Day 3: **engine** (builder) and **UI** (frontend), meeting at the API contract frozen on Day 1.

---

## Day 1: Contracts, simulator, ground truth
- [x] `git init`, feature branch, repo skeleton per `architecture.md` §6
- [x] Freeze `DetectionEvent` / `Trajectory` / `LinkEvidence` pydantic contracts — **OpenAPI stub NOT done**: explicitly out of scope for this session (API is Day 3); flagged, not forgotten
- [x] City simulator: road graph, camera placement, vehicle + route generation, ground-truth passage events
- [x] Corruption model: OCR character-confusion matrix, partial reads, per-camera miss rate, embedding noise — all knob-controlled
- [x] Sanity: generated 50 cams / 20,000 vehicles / 24h (107,175 events) via the CLI; ground truth round-trips verified by tests

**Success criteria:** `sim.generate(...)` produces a reproducible labelled event stream with tunable difficulty. **Met** — see `docs/decisions.md` "Day 1 build" entries for calibration numbers and tuning notes.

## Day 2: Scoring channels + fusion
- [x] `plate_lr.py` — noisy-channel LR with per-character posteriors, plate-format prior, edit-aware alignment
- [x] `kinematic_lr.py` — per camera-pair travel-time priors, time-of-day buckets, `v_max` hard gate, `p_miss` skip penalty
- [x] `appearance_lr.py` — same/different distance densities, density-ratio LR
- [x] `fusion.py` — log-odds sum; `calibration/` fits priors on the train split
- [x] Unit tests per channel with hand-built adversarial pairs
- [x] Stratified pairwise evaluation (`eval/stratified.py`) — a single scalar AUC over uniformly-sampled negatives is not a valid success criterion here (see below)

**Success criteria (revised — see `docs/decisions.md`, "Day 2b"):** not a single "fused pairwise AUC > 0.97" number. A plate is a near-unique identifier, so uniform/hard-in-time-space negative sampling alone makes plate-only look artificially perfect and hides the one case (clones) the architecture exists for. The real criterion is the **stratum x channel matrix** in `eval/reports/stratified_auc.json`: each channel individually better than chance, plate-only near-perfect on routine traffic, plate-only at-or-below chance on the clone stratum, and fused AUC holding up across every stratum. **Met** — measured on a held-out split (shared city, different seed from train; 4000 train / 6000 held-out vehicles, 20 cameras, clone_fraction=0.02, near_miss_fraction=0.02; degraded stratum measured on a separate higher-noise dataset sharing the same city):

| Stratum (frequency in holdout) | plate AUC | appearance AUC | kinematic AUC | fused AUC |
|---|---|---|---|---|
| routine (89.8%) | 1.0000 | 0.9983 | 0.8601 | 1.0000 |
| clone (4.0%) | 0.5654 | 0.9972 | 1.0000 | 1.0000 |
| plate-similar (4.1%) | 0.9233 | 0.9979 | 1.0000 | 1.0000 |
| degraded (2.1%; stress-dataset n=4000) | 0.9963 | 0.9976 | 0.9241 | 0.9992 |

Frequency-weighted composite: plate-only 0.9795, fused 0.99998. Out-of-sample calibration ECE=0.0002 (target <0.05); scoring throughput ~1500 pairs/sec on a laptop CPU. The thesis case is the clone row: plate-only (0.565) is statistically indistinguishable from chance, exactly as it should be since two different vehicles share the identical plate string — and fusion (1.000) still separates them. `tests/test_stratified_auc.py` encodes this as a hard assertion (clone plate AUC < 0.60 AND clone fused AUC > 0.90), not a number to be re-tuned.

Also completed as part of this session: Day 1c fixed a real defect found by measuring against ground truth — 0.44% of read slots had exactly zero posterior probability on the true character (from a Day 1b storage optimisation that over-corrected the OCR confusion model), which would have sent Day 2's plate LR to `-inf` on 4.3% of events. Fixed at the source (`sim/corruption.py`) plus an explicit residual term in the on-disk codec; permanent regression guard added in `tests/test_zero_probability_guard.py`. 73/73 tests passing, ruff clean.

## Day 3: Association, decoding, API  ‖  UI shell starts
**Engine**
- [x] `gating.py` + `blocking.py` — **met**: gate cuts 99.61% of pairs (207 candidates/event vs 53,617 naive) at **99.72% true-predecessor recall**; plate blocking engages on 0.25% of events at **0.0% recall cost** (`eval/reports/gating.json`, `blocking.json`)
- [x] `mincostflow.py` — own SSP solver, agrees with the `networkx` oracle; sliding window with carry-over; full-day partition test (every event in exactly one trajectory) green
- [x] Vectorised batch scoring — batched-vs-scalar agreement test green (28/28 association + scoring tests)
- [x] `consensus.py`, `partial_search.py`, `clone_detect.py`, `engine/analytics/*` — 35 tests green
- [x] Clone realism: overlapping-route clones added. Overlap stratum: plate 0.502, kinematic **0.966** (down from 1.000 on disjoint), appearance 0.999, fused **0.998** (`eval/reports/clone_overlap_auc.json`)
- [x] Consensus outlier-robustness (ε=0.2 contamination) — run1 GT-grouped: 3 reads 97.88%→98.30%, 4+ reads 99.88%→99.95% (`eval/reports/consensus_accuracy.json`)
- [x] **Full-city trajectory result** (107,234 events, 20k vehicles, 24h): SUTRA IDF1 **0.926** vs exact-match **0.875**; ID switches **587 vs 19,940**; fragmentation 380 vs 13,055; completeness 0.995 vs 0.877. Solve 8.7 min after exact pruning (19.1M → 392k arcs). Open weakness: SUTRA over-merges (18,483 trajectories for 19,995 vehicles); error analysis in progress. 2h quiet-hours slice: 0.972 vs 0.884.
- [x] FastAPI + SQLite + replay/WebSocket — 25/25 API tests; routes match `docs/api-contract.md` exactly; `api/openapi.json` exported

**UI (parallel, against the frozen contract + mock data)**
- [x] All 6 pages (Live, Trajectories + WHY panel + consensus, Search, Analytics, Alerts, Results) — tsc clean, build OK, checked in browser in mock mode

**Success criteria:** end-to-end — simulator events in, reconstructed trajectories out of the API, streaming live to a map.

## Day 4: Evaluation, ablations, real video  ‖  UI features
**Engine**
- [ ] `eval/` — IDF1, ID-P/R, ID-switches, fragmentation, plate accuracy, search recall@k, clone P/R
- [ ] Baselines A/B/C implemented and run head-to-head
- [ ] Ablation table + stress sweeps (OCR error 0→40%, miss rate 0→30%)
- [ ] *(Stretch, only if the above is green)* real-video path: YOLO + OCR + Re-ID on 2–3 clips through the same contract

**UI**
- [ ] Trajectory replay, plate/partial-plate search, analytics dashboards, alerts panel, **WHY panel** (per-link LR breakdown)

**Success criteria:** a results table showing SUTRA beating all three baselines, with the gap widening as noise rises.

## Day 5: Polish, packaging, defence
- [ ] `docker-compose up` works clean on a fresh machine, offline
- [ ] Seed a canned demo scenario (a specific vehicle whose plate is misread at 3 of 6 cameras, plus one planted clone)
- [ ] Demo script: 5 minutes, 6 beats — problem → baseline fails live → SUTRA links it → WHY panel → consensus repairs the plate → clone alert + analytics
- [ ] README, architecture diagram, results charts, judge Q&A sheet (the hard ones: scale, privacy/DPDP, false-link cost, real-feed integration)

**Success criteria:** anyone on the team can run the demo end-to-end and defend every number in the results table.

---

## Notes
- If a day runs long, the schedule shifts — don't ship unfinished work just to stay on the calendar.
- The real-video path is **additive, never blocking**. If Day 4 slips, it is the first thing cut.
- Day 1's contract freeze is what lets the frontend run in parallel. Do not let it slip.

## Progress log
*(updated at the end of each session)*
- **2026-09-15** — Problem analysed, architecture framed, docs moved into `docs/`. Build not yet started.
- **2026-09-15** — Day 1 built (contracts, city sim, corruption model, tests) and reviewed. Day 1b follow-up: fixed a real appearance-embedding bug (city-scale rank-1 was collapsing to ~15% because per-instance noise wasn't unit-normalised against the class term) with a 3-level hierarchical embedding, and compacted the on-disk event format 13.9x (7378 -> 771 bytes/event) by sparsifying plate posteriors and moving embeddings to a companion float16 `.npy`. See `docs/decisions.md` "Day 1b" for full detail. 32/32 tests passing, ruff clean.
- **2026-09-15** — Day 1c: found and fixed a zero-probability defect in the on-disk plate posterior codec (see `docs/decisions.md`, "Day 1c") before it could poison Day 2's plate LR. Day 2 built and verified: `engine/scoring/{plate_lr,kinematic_lr,appearance_lr,fusion}.py`, `engine/calibration/{fit_priors,calibrate}.py`, all Day 2 success criteria met (fused AUC 0.9998 > 0.97 target, ECE 0.0002 < 0.05 target, all channels AUC > 0.5). 65/65 tests passing, ruff clean. Next: Day 3 (gating, min-cost flow, consensus decode, clone detect, FastAPI) plus UI shell in parallel.
- **2026-09-16** — Day 3a/3b engine built and verified (gating, blocking, min-cost flow, windowing, vectorised scoring, consensus, partial search, clone detection, analytics). API contract frozen (`docs/api-contract.md`). UI 5/6 pages built, build currently broken (fix in progress). Remaining: IDF1 + baseline, API, consensus robustness, baselines/ablations/stress, packaging, demo.
