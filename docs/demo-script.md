# Demo script — SUTRA (SIH26127)

Five minutes, six beats. Every number here is measured and lives in `eval/reports/`. Never quote a number you can't point at.

**Before you start:** `make serve` (or `.\make.ps1 serve`), browser at `http://localhost:8000`, replay paused at 00:00, speed 60×. Have the Results page open in a second tab.

---

## Beat 1 — The problem (40s)

> "A city has hundreds of ANPR cameras. Each one produces an isolated line: camera, time, plate. Nothing joins them into a journey. The standard fix is to run OCR everywhere and match plate strings exactly."

Point at the Live map, replay running.

> "That breaks, because plate OCR is wrong roughly one read in eight — angle, blur, night glare, a truck in the way. Exact string matching is a hard rule applied to soft evidence. One bad character and the journey shatters."

## Beat 2 — Show it shattering (45s)

Results tab → the baseline row.

> "We built the standard solution too, so we can show you what it does. On one simulated city day — 50 cameras, 20,000 vehicles, 107,234 reads — exact matching turns 19,995 real vehicles into **32,945** fragments. It makes 19,940 identity errors."

## Beat 3 — The idea (40s)

> "We stopped asking 'are the strings equal?' and started asking 'what is the probability these two reads are the same vehicle?' Three independent kinds of evidence: how compatible the two plate readings are, character by character, with their uncertainty; whether the vehicle looks the same; and whether the travel time between those two cameras is physically plausible on the road network."
>
> "Then, instead of deciding link by link, we solve the whole city at once — the set of journeys that best explains every read, with one rule the maths enforces for free: each read belongs to exactly one vehicle."

Same tab:

> "**IDF1 0.990 versus 0.875.** Identity errors: **754 versus 19,940.** And we find 20,328 journeys where the truth is 19,995 — the baseline finds 32,945."

## Beat 4 — Why, for a single link (60s) — *the differentiator*

Trajectories → open a trajectory → **WHY panel**.

> "Judges usually ask whether this is a black box. It isn't. Click any link in any journey and it tells you exactly why the two reads were joined."

Point at the bars.

> "Prior, plate evidence, appearance, travel time — they add up to the total, and the total is the log-odds that these are the same vehicle. Here the plate evidence is weaker because of a misread, and appearance plus travel time carry the link. That is a number an operator can challenge in court, not a hunch."

## Beat 5 — Linking repairs the OCR (45s) — *the counter-intuitive one*

Same page, **consensus panel**.

> "Here's the part we didn't expect. Once reads are on one journey, we fuse them character by character. Open `traj_020319`. Eight cameras saw this vehicle, and **three of the eight misread the plate** — TS22DO2447, TS72DO2442, TS22DO0002. The fused answer is **TS22DO2442 at 99.6% confidence**, and it is correct."
>
> "So linking doesn't just use plate reads — it *repairs* them. A single read is right about 88% of the time. After fusing four or more, it's **99.95%**. Better identity out than any camera put in."

*(Verified live in the current database: `traj_020319`, alert on plate KA38XL__22. If you reseed the data, re-check these two IDs before presenting.)*

Search tab: type `TS22DO24??`.

> "And because we hold probabilities, not strings, partial plates are searchable. A witness remembers four characters — we rank every journey in the city by how well it matches."

## Beat 6 — Cloned plates (50s) — *the one nobody else has*

Alerts tab → a clone alert.

> "This falls out of the same maths for free. Plate KA38XL__22, seen at two cameras **7.2 km apart, 118 seconds apart**. The road network says that trip needs at least 215 seconds even at 120 km/h — this implies **219 km/h**. And the two vehicles don't look alike: appearance distance 0.60."
>
> "An exact-match system silently merges these into one impossible journey. We flag it as a **cloned plate**, with the evidence attached. On this day the system raised **175 clone alerts and 5 impossible-travel alerts** — and 47.6% of multi-camera journeys had at least one plate read that the fusion corrected."

**Close:**

> "No API keys, no cloud, no internet. It runs on a laptop at this table. The gains grow as conditions get worse — which is exactly when a city needs it."

---

## If they ask you to prove it live

- **"Is it really running?"** — Alerts and Live are streaming from the local API; press pause and step the clock.
- **"Did you tune it on the test data?"** — No. Priors and calibration are fitted on a separate simulated day with a different random seed. Say so plainly; it's in `docs/decisions.md`.
- **"Show me a failure."** — Do it. Filter journeys to 2 events, find a split. We over-split by 333 journeys out of ~20,000, and we know why (see the Q&A sheet).

## Timing discipline

Beats 4, 5 and 6 are what makes this different from every other team's entry. If you are running long, cut Beat 1 to one sentence and shorten Beat 2 — never cut the WHY panel, the consensus repair, or the clone alert.
