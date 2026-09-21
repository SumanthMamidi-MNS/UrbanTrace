# Architecture — SUTRA

**SIH26127 — City-Wide AI Engine for Multi-Camera ANPR Trajectory Tracking and Urban Traffic Analytics (BEL)**

SUTRA = *Stitching Uncertain Traffic Reads into Actionable trajectories*. Sanskrit *sutra* = "thread" — the product threads isolated ANPR beads into one continuous journey.

---

## 1. The problem, precisely stated

Every ANPR camera in a city emits **isolated events**: `(camera, timestamp, plate_guess, confidence, vehicle_crop)`.
Nothing links them. The standard fix — run OCR everywhere and join on `plate_A == plate_B` — fails because real-world plate OCR is wrong often:

| Failure mode | Effect on exact-match join |
|---|---|
| Char confusion (O/0, I/1, B/8, S/5, G/6, Z/2) | One vehicle splits into 2+ phantom vehicles (**fragmentation**) |
| Partial read / occlusion (6 of 10 chars) | Event orphaned, trajectory truncated |
| Motion blur, glare, night IR washout | Same |
| Missed detection (camera never fires) | Gap in trajectory, looks like a teleport |
| Cloned / tampered plates | Two real vehicles collapse into one fake trajectory (**merge error**) |

A single bad read anywhere destroys the whole journey. Exact string equality is a **brittle hard constraint applied to soft evidence**. That is the actual bug in the standard solution.

## 2. The invention

**Stop asking "are the strings equal?" Start asking "what is P(same vehicle | all evidence)?"**

Three *independent* evidence channels are fused into a log-likelihood ratio, then a **global** assignment picks the set of trajectories that best explains the whole city at once.

```
                     +--------------------------------------+
  plate posterior -->|  Plate LR      (noisy-channel model) |--+
  Re-ID embedding -->|  Appearance LR (calibrated density)  |--+--> sum log-odds --> min-cost flow --> trajectories
  dt + road graph -->|  Kinematic LR  (travel-time prior)   |--+                                            |
                     +--------------------------------------+                                              v
                                                                                            consensus plate decoding
```

Three consequences no exact-match system can produce, all falling out of the same math:

1. **Linking repairs OCR.** Once N reads sit on one trajectory, fuse their per-character posteriors — the true plate is recovered far above any single read's accuracy. Bad reads get outvoted.
2. **Partial-plate search works.** `MH12??1234` is just a constraint on the plate posterior; every trajectory can be ranked against it.
3. **Clone detection falls out.** Same plate + physically impossible travel time + different appearance embedding ⇒ flagged as a cloned/spoofed plate instead of silently merged.

## 3. The math (this is the defence)

For two events `e_i` (cam `c_i`, time `t_i`) and `e_j` (cam `c_j`, time `t_j > t_i`):
`H1` = same vehicle, consecutive observations. `H0` = unrelated vehicles.

```
s(i,j) = log prior-odds
       + log [ P(O_i,O_j | H1) / P(O_i,O_j | H0) ]      <- plate
       + log [ p1(d_ij)        / p0(d_ij)        ]      <- appearance
       + log [ p(dt | c_i->c_j) / p0(dt)         ]      <- kinematics
```

**Plate term.** Marginalise over the unknown true plate `y`:
`P(O_i,O_j|H1) = SUM_y pi(y) * P(O_i|y) * P(O_j|y)`
`P(O_i,O_j|H0) = (SUM_y pi(y) P(O_i|y)) * (SUM_y pi(y) P(O_j|y))`

- `P(O|y) = PROD_k P(o_k|y_k)` uses the OCR's **per-character posterior**, not the argmax string. *(Design rule: the perception layer must emit per-character distributions. Throwing them away is exactly what makes every other team's system brittle.)*
- `pi(y)` = Indian plate format grammar `SS DD L(L) DDDD` + RTO-code prior. The sum factorises per character position, so it is O(positions x alphabet) — cheap.
- **Invariant: every slot posterior retains background mass on every character in its alphabet.** No character is ever assigned probability exactly zero. `P(O|y)` is a product over slots, so a single zero would send the log-LR to `-inf` and hard-reject a correct match — reintroducing exact-matching's brittleness inside the probabilistic model. A character outside the confusion group is *unlikely, never impossible*; real OCR softmax never emits a zero either. The on-disk codec carries an explicit residual term so sparse storage cannot violate this, and `plate_lr` floors probabilities defensively as a second line of defence. *(This is also the answer to "what happens on an OCR error your confusion model didn't anticipate?")*
- The true plate `y` lives in a **fixed 10-slot canonical form** (`SS DD LL DDDD`, with legal variants). A slot the OCR never read gets a **uniform posterior**, so it contributes exactly zero to the LR — a partial read is *missing information, not evidence against a match*. This is the clean way to handle occlusion, and it is why partial reads don't sink us.
- Dropped/inserted characters shift the alignment. Rather than a full edit-HMM, we search the small set of plausible alignments (shift 0/±1 at each segment boundary) and take the max-likelihood one.

**Appearance term.** `d_ij` = cosine distance between vehicle Re-ID embeddings. Fit `p1` (same-vehicle distance distribution) and `p0` (different-vehicle) on held-out labelled pairs; the LR is their density ratio. Calibrated, not a hand-tuned threshold.

**Kinematic term.** Per camera-pair, per time-of-day bucket, learn `p(dt)` (log-normal). Hard gate: `dt < dist / v_max` ⇒ physically impossible ⇒ `-inf`. *That gate is what creates clone alerts.* Non-adjacent camera pairs are reachable only through a missed-detection penalty `log p_miss` per skipped intermediate camera.

**Global association.** Time-ordered DAG, one node per event split `u_i -> v_i` (arc cost = detection-confidence cost). Arc `v_i -> u_j` costs `-s(i,j)` for spatio-temporally feasible pairs; source/sink arcs are entry/exit costs learned from the fraction of events at border cameras. Each unit of flow traces one vehicle's trajectory. This is the Zhang–Li–Nevatia network-flow tracker lifted from a single camera to a whole city: globally optimal for the pairwise model, and it enforces mutual exclusion for free (one event belongs to exactly one vehicle).

Solved by **successive shortest paths with Johnson potentials** (our own implementation, ~150 lines). Two reasons this beats calling OR-Tools:
- Costs are log-likelihoods — **floats**. OR-Tools' `SimpleMinCostFlow` takes integer costs, so we'd be rounding likelihood ratios. Unnecessary precision footgun.
- Total cost is convex in *k*, so we **stop augmenting the moment the shortest path cost turns positive** — the optimal number of vehicles *k* comes out of a single pass, instead of binary-searching *k* with repeated solves.

`networkx.min_cost_flow` is kept as a test oracle to verify our solver on small instances.

**Consensus decoding.** For trajectory `T`: `P(y|T) ∝ pi(y) * PROD_m P(O_m|y)`. Argmax = repaired plate, entropy = confidence. Optionally iterate (EM): repaired plates → re-link → better trajectories → better plates.

## 4. System layers

```
L6  WEB UI (React + MapLibre)  live map . trajectory replay . search . analytics . alerts . WHY-panel
L5  API (FastAPI REST + WebSocket)
L4  ANALYTICS  OD matrix . corridor travel time . congestion . volumes . dwell . anomalies
L3  LINKING ENGINE  gating -> 3-channel LR fusion -> min-cost flow -> consensus decode -> clone detect
L2  PERCEPTION  vehicle detect -> in-camera track -> best-shot -> plate OCR (per-char posterior) -> Re-ID embedding
L1  INGEST  SimSource | VideoSource | CsvReplaySource  --> canonical DetectionEvent
```

**The pivotal decision: L1 and L3 are separated by a hard contract.** The engine consumes `DetectionEvent`s and neither knows nor cares whether they came from a simulator or a real camera.

```python
DetectionEvent:
    event_id, camera_id, timestamp
    plate_posterior : list[dict[char, float]]   # per-position distribution (NOT a string)
    plate_argmax    : str                       # convenience only
    embedding       : float[128]                # vehicle Re-ID
    attributes      : {color, vehicle_type, confidence}
    crop_uri        : str | None
    gt_vehicle_id   : str | None                # simulator only — for evaluation
```

This is what makes the week survivable *and* the demo strong: the simulator gives **ground truth** (so we can report real IDF1 / ID-switch numbers and ablations), while the video path gives **realism** (so the demo is not "just a simulation").

## 5. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Engine / API | **Python 3.12, FastAPI, Pydantic v2** | ML ecosystem; async + WebSocket native; schema *is* the contract |
| Numerics | **numpy, scipy, scikit-learn** | LR fitting, calibration (isotonic / Platt) |
| Association | **own successive-shortest-paths solver** (`networkx` as test oracle) | float costs, optimal *k* in one pass — see §3 |
| Gating | **road-graph + time-window gate** (primary), `rapidfuzz` deletion index (overflow only) | see §7 — plate-based blocking must never be the primary filter |
| Perception (optional) | **Ultralytics YOLO** (vehicle + plate), **PaddleOCR / CRNN** (per-char logits), **OSNet / torchreid** (Re-ID) | pretrained, CPU-tolerable on short clips |
| Storage | **SQLAlchemy + SQLite** (dev/demo), Postgres drop-in | zero-friction offline demo, same ORM scales up |
| Frontend | **React + Vite + TypeScript + Tailwind** | fast, standard |
| Map | **MapLibre GL, road graph rendered as GeoJSON** (no tile server) | open source, no API token, fully offline; the synthetic city's roads come straight from `/api/city` |
| Charts | **Recharts** | quick, good enough |
| Transport | REST + **WebSocket** for the live feed | no broker needed at demo scale |
| Packaging | **docker-compose** (api, worker, web) | one command, runs offline on a laptop |

Deliberately **not** used this week: Kafka, Spark, TimescaleDB, Kubernetes. Recorded in `decisions.md` as the production scale-out path, so it reads as a stated choice rather than an omission.

## 6. Folder structure

```
SIH-Project-2/
├─ docs/                 PRD . architecture . phases . decisions . memory
├─ engine/
│  ├─ contracts/         DetectionEvent, Trajectory, LinkEvidence (pydantic)
│  ├─ sources/           sim_source.py . video_source.py . csv_source.py
│  ├─ perception/        detect.py . ocr.py . reid.py . bestshot.py
│  ├─ scoring/           plate_lr.py . appearance_lr.py . kinematic_lr.py . fusion.py
│  ├─ association/       gating.py . blocking.py . mincostflow.py . window.py
│  ├─ decode/            consensus.py . partial_search.py . clone_detect.py
│  ├─ analytics/         od_matrix.py . corridor.py . volumes.py . anomalies.py
│  └─ calibration/       fit_priors.py . calibrate.py
├─ sim/                  city graph, camera layout, vehicle/route generator, corruption model
├─ api/                  FastAPI app, routers, websocket, db models
├─ web/                  React app
├─ eval/                 metrics.py . baselines.py . ablation.py . reports/
└─ docker-compose.yml
```

## 6b. API contract

Frozen in [`docs/api-contract.md`](api-contract.md): REST under `/api`, live stream on `/ws/live`, replay control via `POST /api/replay`. Both `api/` and `web/` build against it; the UI has a contract-exact mock mode.

## 7. Data flow

1. A source emits a `DetectionEvent` (simulated, or a real frame through L2 perception).
2. The event is persisted and pushed onto the live WebSocket feed.
3. **Gating** finds candidate predecessors: events at cameras upstream in the road graph, within `[dt_min, dt_max]`.

   > **The spatio-temporal gate is the primary and sufficient blocker — plate-based blocking is not.** At target scale (50 cameras, ~500 events/camera/hour, a 45-minute window) the gate already leaves only a few hundred candidates per event, all of which we score in full. Plate blocking is engaged *only* if a gate overflows a candidate budget.
   >
   > This matters more than it looks. Using a plate index as the primary filter would silently reintroduce the exact brittleness we are attacking — a badly-misread plate would never surface as a candidate, so no amount of clever downstream scoring could recover it. The recall of the whole system is decided here.
4. **Scoring** computes the three LRs and sums log-odds for each candidate pair.
5. Every `W` seconds a **sliding window** (~30 min, overlapping) is solved by min-cost flow into trajectories; tracks at the window edge carry over.
6. **Consensus decoding** repairs each trajectory's plate; the **clone detector** inspects kinematically impossible same-plate pairs.
7. Analytics aggregates update incrementally (OD matrix, corridor times, volumes, anomalies).
8. The UI renders the live map, trajectory replay, search, dashboards and alerts — plus a **WHY panel** breaking any single link down into its three LR contributions.

## 8. Evaluation plan (the scoreboard we defend with)

| System | Description |
|---|---|
| Baseline A | Exact plate string match (what the problem statement calls the standard solution) |
| Baseline B | Fuzzy match, Levenshtein <= 1 |
| Baseline C | Fuzzy + hard time gate |
| **SUTRA** | Probabilistic 3-channel fusion + global min-cost flow |

Metrics: **IDF1, ID-Precision, ID-Recall, ID-switches, fragmentation count, trajectory completeness, plate accuracy (single read vs consensus), partial-plate search recall@k, clone-detection P/R**, throughput (events/sec).

Ablations: plate-only → +kinematics → +appearance → +global flow → +EM. Every row has to justify its component's existence.

Stress sweeps: OCR error rate 0→40%, camera miss rate 0→30%, camera density, traffic volume. The headline claim is that our gap over Baseline A **widens** as conditions get worse — which is exactly what the real world does.

**Scale target:** 50 cameras, 20,000 vehicles, ~500,000 events over 24 simulated hours. Every throughput and latency number we quote is measured at this scale.

**Anti-strawman guard (important).** A simulator we wrote ourselves could trivially be tuned to make our method look good and the baseline look stupid. So the corruption model is pinned to *published real-world numbers*, not to whatever flatters us:
- per-read plate OCR accuracy ≈ **85–95%** (typical field ANPR)
- Re-ID rank-1 on the appearance channel alone ≈ **70%** (in line with published VeRi-776 results)
- camera miss rate and travel-time variance drawn from realistic ranges

If a single channel performs far better than its real-world counterpart, the simulator is wrong and the result is worthless. This calibration check is a test in `eval/`, not a footnote — and it is the first question a sharp judge will ask.

**Pairwise scoring evaluation is stratified, not a single scalar (docs/decisions.md, "Day 2b").** A plate is a near-unique 10-character identifier. Sampling negative pairs uniformly — or even "hard" only in time/space — makes the plate channel look artificially perfect, because two unrelated vehicles almost always have completely different plates: there is no headroom left for appearance or kinematics to demonstrate value, and reported as one number this would argue the three-channel architecture is unnecessary, the opposite of the project's thesis. Worse, the one case the architecture exists for — a cloned plate — is invisible in an aggregate figure. `eval/stratified.py` scores four named strata separately instead:

| Stratum | Negative construction | What it measures |
|---|---|---|
| **routine** | closest-in-time different vehicle, same destination camera | the everyday case — plate-only is correctly near-perfect here |
| **clone** | different vehicle, exact same true plate | the thesis case — plate-only is structurally blind; fusion must still work |
| **degraded** | as routine, but positives have ≥4 combined unread/misread slots across the pair | plate evidence genuinely weakened by OCR noise, not just diluted by an easy negative |
| **plate-similar** | different vehicle, true plate within edit distance ≤2 (a realistic near-miss registration, e.g. sequential RTO numbering: `MH12AB1234` / `MH12AB1235`) | plate evidence present but weak, not absent |

The headline is the **stratum x channel matrix** (`eval/reports/stratified_auc.json`), reported alongside each stratum's natural frequency in real traffic — not a single number. The honest framing, and the load-bearing test in `tests/test_stratified_auc.py`: plate-only is near-perfect on routine traffic and collapses to chance on clones; fusion holds across all four strata. A frequency-weighted composite is also reported, but only as a secondary summary — the matrix is what the ablation story rests on.

## 9. Deployment

`docker-compose up` → api (FastAPI + worker), web (Vite build behind nginx), volume-mounted SQLite. Fully offline: map drawn from the road graph (no tiles), models bundled. No internet needed at the demo table.

## 10. Known risks

| Risk | Mitigation |
|---|---|
| Real video pipeline eats the week | The L1/L3 contract — the sim path is the guaranteed deliverable, video is additive |
| No GPU | Re-ID falls back to colour-histogram + type classifier; OCR only on short clips |
| Min-cost flow too slow at scale | Sliding window + aggressive gating; complexity measured and reported honestly |
| "It's only a simulation" | Answer with the video path, plus the fact that the sim gives ground truth no real feed can |
| Priors over-fitted to our own simulator | Calibrate on a train split, report on a held-out split generated with different parameters |
