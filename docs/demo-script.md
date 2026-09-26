# Demo script — SUTRA (SIH26127)

About six minutes, seven beats. Every number here is measured and lives in `eval/reports/`. Never quote a number you can't point at.

**Before you start:** `make serve` (or `.\make.ps1 serve`), browser at `http://localhost:8000`, replay paused at 00:00, speed 60×. Have the Results page open in a second tab. Have the Watchlist page ready with the pattern `TN15TU5117` typed but not added.

*(Verified live in the current database — congested city day `run2`: repaired journey `traj_021211`, clone alert on plate `KA98LM9842`, watchlist example `TN15TU5117`. If you re-run the pipeline or reseed, re-check these three before presenting — IDs change.)*

---

## Beat 1 — The problem (40s)

> "A city has hundreds of ANPR cameras. Each one produces an isolated line: camera, time, plate. Nothing joins them into a journey. The standard fix is to read the plate and match the strings exactly."

Point at the Live map, replay running.

> "That breaks, because plate reading is wrong a lot of the time — angle, blur, night glare, a truck in the way. Exact matching is a hard rule applied to soft evidence. One bad character and the journey shatters."

## Beat 2 — Show it shattering (40s)

Results tab → the baseline row.

> "We built the standard solution too, on the same data. One simulated city day with real rush-hour congestion — 50 cameras, 20,000 vehicles, 103,475 reads. Exact matching turns 19,996 real vehicles into **32,475** fragments and makes 19,214 identity errors."

## Beat 3 — The idea (40s)

> "We stopped asking 'are the strings equal?' and started asking 'what is the probability these two reads are the same vehicle?' Three kinds of evidence: how compatible the two plate readings are, character by character; whether the vehicle looks the same; and whether the travel time between those cameras is plausible on the road network at that time of day. Then we solve the whole city at once, with one rule the maths enforces: each read belongs to exactly one vehicle."

Same tab:

> "**IDF1 0.972 versus 0.875.** Identity errors: **2,143 versus 19,214.**"

## Beat 4 — Why, for a single link (50s) — *the differentiator*

Trajectories → open a trajectory → **WHY panel**.

> "Judges usually ask whether this is a black box. It isn't. Click any link and it shows exactly why two reads were joined: prior, plate evidence, appearance, travel time — they add up to the total. That's evidence an operator can challenge, not a hunch."

## Beat 5 — Linking repairs the OCR (45s) — *the counter-intuitive one*

Open `traj_021211`, **consensus panel**.

> "Eight cameras saw this vehicle, and **three of the eight misread the plate** — RJ78IV2340, RJ78IR2345, RJ78LV2385. Fused character by character, the answer is **RJ78IV2345**, and it's correct. Linking doesn't just use plate reads — it repairs them."

Search tab: type `RJ78IV23??`.

> "And because we hold probabilities, not strings, partial plates are searchable. A witness remembers most of a plate — we rank every journey in the city by how well it matches."

## Beat 6 — Alerts: cloned plates and the watchlist (60s)

Alerts tab → the clone alert on `KA98LM9842`.

> "Plate KA98LM9842, seen at two cameras **7.8 km apart, 139 seconds apart**. The road network needs at least 235 seconds even at 120 km/h — this implies **203 km/h**. And the two vehicles don't look alike. An exact-match system silently merges these into one impossible journey. We flag a **cloned plate**, with the evidence attached."

Watchlist page → add `TN15TU5117`, reason "stolen vehicle". Start replay at high speed.

> "Now a blacklist. Watch — the alert fires, and it says *why*: this camera's own read wasn't confident enough to match, but the vehicle's **fused plate across several cameras** matches at 93%. A string blacklist would have let this car through."

## Beat 7 — The city view (30s)

Live map → heatmap on (speed). Then Analytics.

> "The same trajectories give city-wide analytics: the heatmap goes red at the bottlenecks as rush hour builds. City mean speed drops from 80 km/h at night to about 65 at the peaks; the worst corridors run at around 70% of free-flow speed averaged over the day. Origin-destination flows, volumes, and bottlenecks — all from the same links."

**Close:**

> "The plate reader is a deep-learning model fine-tuned on real Indian plates: 94% of characters right on plates it has never seen. No API keys, no cloud, no internet — it runs on this laptop. And the worse the plate readings get, the bigger our lead over exact matching — exactly when a city needs it."

*(Precise versions, if pressed: OCR is 94.3% per character but **81.0% whole-plate**, below the PRD's 90% — say so. The tracking gap widens from 0.09 to 0.43 as per-read plate accuracy falls from 89% to 50%; when cameras miss vehicles (up to 30%) the lead holds at ~0.1 but doesn't widen. Don't claim more than that.)*

---

## If they ask you to prove it live

- **"Is it really running?"** — Alerts and Live stream from the local API; pause and step the clock.
- **"Did you tune it on the test data?"** — No. Priors, the link threshold and the congestion model were all calibrated on separate simulated days with different random seeds. It's in `docs/decisions.md`.
- **"Show me a failure."** — Do it. We over-split: 21,218 journeys for 19,996 vehicles on the congested day. And the detector finds only about half the small, distant plates in general traffic footage. See the Q&A sheet.

## Timing discipline

Beats 4, 5 and 6 are what makes this different from every other entry. If running long, cut Beat 1 to one sentence and shorten Beat 7 — never cut the WHY panel, the consensus repair, or the watchlist catch.
