# Judge Q&A — UrbanTrace (SIH26127)

Hard questions, honest answers. Every number is in `eval/reports/`. **If you don't know, say so** — a confident wrong answer is worse than "we haven't measured that yet."

---

## The one that decides the result

**"The tracking results are on simulated data. Why should we believe any of it?"**

> Three answers. First, we need ground truth: no public dataset gives multi-camera plate reads across a city *with* the true vehicle journeys, and without that you cannot measure trajectory accuracy at all — you can only demo. Second, we deliberately pinned the simulator to published real-world figures: per-read plate accuracy 85–95%, appearance-only re-identification around 70% at benchmark gallery size, and rush-hour congestion with bottleneck corridors slowing to ~40–60% of free-flow speed. There are tests that fail the build if the noise drifts outside those bands, precisely so we can't flatter ourselves. Third, we implemented the standard solution on the *same* data, so the comparison is like-for-like.
>
> And the plate reader is **not** simulated: it's measured on 276 real Indian plates it never saw in training.

**"But your simulator could be tuned to make your method look good."**

> It could, which is why the calibration bands are asserted in tests and why we tune on one random seed and report on another. Adding realistic congestion actually made our own numbers *worse* (0.991 → 0.972) — we kept it because it's closer to a real city. Real deployment still needs real calibration: a few days per city, not a redesign.

---

## The OCR requirement

**"The PRD asks for more than 90% OCR accuracy. Did you get it?"**

> Partly, and we'll say exactly how. On a held-out set of **510 real Indian plate images (276 unique plates)** that the model never saw:
> - **94.3% per character** — above 90%.
> - **81.0% whole plate** (every character right) — **below 90%**.
>
> Most ANPR vendors quote whole-plate accuracy, so we don't claim the requirement is met. The gap is training data: we had 546 unique real plates to train on. The model's validation score (88%) sits above its test score, which says more real plates would close most of the gap.

**"So how can the tracking work with 81% plates?"**

> Because we never trust one read. A vehicle seen by several cameras has several reads, and we fuse them character by character. In our experiments, fusing four or more reads reaches **99.95%** plate accuracy even when single reads are only ~88% right. And our stress test shows tracking holds at IDF1 ≈ 0.98 even when half of all plate reads are wrong.

**"Did you fine-tune any AI model?"**

> Yes, two:
> - **Plate reader:** the open-source fast-plate-ocr model (MIT licence), pretrained on ~220,000 plates from many countries — none from India — fine-tuned on our real Indian training plates. It went from 51.4% to 81.0% whole-plate. We also built our own CRNN from scratch; it plateaued at 44.3% because we don't have enough real data to train one from nothing. That's documented, not hidden.
> - **Plate detector:** a pretrained YOLO plate detector, fine-tuned on Indian scene images. On a traffic video it never saw, recall went from 0.33 to **0.51**.

**"0.51 detector recall sounds low."**

> It is, and it's our biggest real-world weakness. Those clips are general highway footage where each plate is ~30 pixels wide. Real ANPR cameras are mounted and zoomed so plates are 100+ pixels. On close-up vehicle photos the detector finds 95% of plates. We'd rather show you the hard number than the easy one.

---

## Method

**"Why not just fuzzy-match plates?"**
> We built that as Baseline B. Edit distance has no probabilistic meaning — you can't combine "one character off" with "arrived 4 minutes late" and "looks like a different colour" in a principled way. Likelihood ratios you *can* add. And fuzzy matching makes the cloned-plate case worse, not better.

**"Where does the probability actually come from?"**
> The OCR gives a distribution over characters per position, not a string. We compare two reads by marginalising over the unknown true plate. Travel times are learned per camera pair, per time of day, against what *unrelated* traffic on the same road does. Appearance is a calibrated ratio of same-vehicle to different-vehicle distance distributions. Everything is a log-likelihood ratio, so the evidence adds.

**"Why min-cost flow instead of just linking the best match?"**
> Greedy linking can assign the same read to two vehicles and can't undo an early mistake. The flow formulation enforces "one read, one vehicle" structurally and is globally optimal for our pairwise model. Honestly, once the evidence channels are well calibrated, the accuracy margin is small — 0.992 vs 0.991 in our ablation. Its value is the guarantee, not a big score gain.

**"What if OCR fails completely — no plate at all?"**
> An unread character contributes exactly zero, not evidence against, so the characters that *were* read still count. But with no plate evidence at all, appearance and travel time alone reach only 0.388 IDF1 in our ablation — in a busy city too many similar-looking vehicles arrive at similar times. They strengthen a plate reading; they can't replace one.

**"How does the watchlist catch a stolen vehicle whose plate was misread?"**
> The watchlist is probabilistic. We check each read, *and* the fused plate of the vehicle's whole journey so far. On our data, plate `TN15TU5117` was caught at 93% from the fused multi-camera plate at a camera whose own read wasn't confident enough to match alone. A string-match blacklist would have missed it.

---

## Results and weaknesses

**"What's your biggest weakness?"**
> Three, honestly: (1) small, distant plates in general traffic footage — detector recall 0.51; (2) whole-plate OCR at 81%, below the PRD's 90%; (3) tracking over-splits slightly: 21,218 journeys against 19,996 real vehicles on the congested day. A vehicle stuck in heavy congestion or missed by several cameras sometimes comes back as a new journey. We'd rather over-split than over-merge: a merge invents a journey that never happened and could send police to the wrong vehicle.

**"You said IDF1 0.972. What does that mean?"**
> Of all the read-to-vehicle identity assignments on a congested city day, about 97% are right. The standard exact-plate-matching approach gets 87%. Precision and recall are identical here because every read is assigned to exactly one journey and the ground truth covers every read, so false positives and false negatives are necessarily equal.

**"Did the numbers ever look wrong?"**
> Yes, several times, and each one made the system better. Our first full-city run quietly merged different vehicles (the engine assumed ~5 possible matches per read instead of ~218). Our ablation later showed the travel-time model was rewarding *any* plausible arrival time. When we added realistic rush-hour congestion, accuracy fell to 0.88 — because our candidate filter was throwing away 30% of true matches for jammed vehicles. Fixing that took it to 0.972. Every earlier set of numbers is still in the repo.

---

## Deployment and scale

**"Will this work on a real city?"**
> At our tested scale — 50 cameras, ~100,000 reads a day with rush-hour congestion — the linking step takes about 12 minutes on a laptop, processing time windows independently, so it runs online. Scaling further means partitioning by region and running windows in parallel; the algorithm doesn't change. We deliberately avoided Kafka/Spark because demo scale doesn't need them.

**"What about privacy? This tracks people."**
> It does, and that deserves a straight answer. This is vehicle data, which under India's DPDP Act is personal data when linked to an owner. Our system stores plate reads and journeys, not identities — it never touches the registration database. A real deployment needs access control, retention limits, and an audit trail of who searched for what. The WHY panel helps: every link has attached evidence, so a decision can be challenged rather than taken on faith.

**"What's the cost of a wrong link in the field?"**
> Serious, which is why the WHY panel exists and why we report confidence rather than a yes/no. An operator should see "91% same vehicle, plate evidence weak" and treat it accordingly. We'd recommend a confidence threshold for automated action and human review below it.

**"Licensing — can BEL actually ship this?"**
> Mostly. Our code, fast-plate-ocr (MIT) and the detector weights (MIT) are fine. The detector *library* (Ultralytics) is AGPL-3.0, which is fine for a prototype but needs a commercial licence or an Apache-licensed detector swap for production. We flagged it rather than hid it.

---

## Engineering

**"What's actually yours versus off-the-shelf?"**
> Ours: the probabilistic plate comparison, the three-channel fusion, the flow formulation and solver, consensus plate repair, partial-plate search, clone detection, the probabilistic watchlist, the congestion-aware candidate filter, the simulator, and the evaluation harness. Off-the-shelf: fast-plate-ocr and a YOLO plate detector (both fine-tuned by us on Indian data), numpy/scipy, FastAPI, React, MapLibre.

**"How do we know the code is correct?"**
> 460+ tests, including exactness guards: our solver's answers are checked against an independent library's optimum; the fast batched scorer is checked against the simple one; a full-day test asserts every read lands in exactly one journey; and a test proves the watchlist catches a vehicle through the fused plate when a single read misses. Most of those exist because each one caught a real bug.
