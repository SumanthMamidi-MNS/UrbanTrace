# Judge Q&A — SUTRA (SIH26127)

Hard questions, honest answers. Every number is in `eval/reports/`. **If you don't know, say so** — a confident wrong answer is worse than "we haven't measured that yet."

---

## The one that decides the result

**"This is simulated data. Why should we believe any of it?"**

> Three answers. First, we need ground truth: no public dataset gives multi-camera plate reads across a city *with* the true vehicle journeys, and without that you cannot measure trajectory accuracy at all — you can only demo. Second, we deliberately pinned the simulator to published real-world figures: per-read plate accuracy 85–95%, appearance-only re-identification around 70% at benchmark gallery size. There's a test that fails the build if our noise drifts outside those bands, precisely so we can't flatter ourselves. Third, we implemented the standard solution on the *same* data, so the comparison is like-for-like. The engine reads camera events through a fixed interface; real video plugs into the same interface.

**"But your simulator could be tuned to make your method look good."**

> It could, which is why the calibration bands are asserted in `eval/` and why we tune on one random seed and report on another. We'd rather show you the guard than ask you to trust us. The honest limitation: our travel-time and appearance models are *our* models of reality. Real deployment needs real calibration, and that's a few days of work per city, not a redesign.

---

## Method

**"Why not just fuzzy-match plates?"**
> We built that as Baseline B. Edit distance has no probabilistic meaning — you can't combine "one character off" with "arrived 4 minutes late" and "looks like a different colour" in a principled way. Likelihood ratios you *can* add. And fuzzy matching makes the cloned-plate case worse, not better.

**"Where does the probability actually come from?"**
> The OCR gives a distribution over characters per position, not a string. We compare two reads by marginalising over the unknown true plate. Travel times are learned per camera pair, per time of day. Appearance is a calibrated ratio of same-vehicle to different-vehicle distance distributions. Everything is a log-likelihood ratio, so the evidence adds.

**"Why min-cost flow instead of just linking the best match?"**
> Greedy linking can assign the same read to two vehicles and can't undo an early mistake. The flow formulation enforces "one read, one vehicle" structurally and is globally optimal for our pairwise model. We measure the difference in the ablation table.

**"What if OCR fails completely — no plate at all?"**
> An unread character contributes exactly zero, not evidence against. That's deliberate: a partial read is missing information. With enough missing characters the plate channel goes quiet and appearance plus travel time carry the link, which is measurable in the ablation row with no plate channel at all.

**"What happens on an OCR error your confusion model didn't anticipate?"**
> Every character keeps a small background probability, so an unexpected error is unlikely, never impossible. We found this the hard way: an early storage optimisation let some probabilities hit exactly zero, which would have hard-rejected the *correct* match on 4.3% of reads. There's now a permanent test that fails if any true character ever gets probability zero.

---

## Results and weaknesses

**"What's your biggest weakness?"**
> We slightly over-split: 20,328 journeys against 19,995 real vehicles. A vehicle that vanishes for a long gap — missed by several cameras in a row — sometimes comes back as a new journey. We'd rather over-split than over-merge: a split loses continuity, a merge invents a journey that never happened and would send police to the wrong vehicle.

**"You said IDF1 0.990. What does that actually mean?"**
> Of all the read-to-vehicle identity assignments, about 99% are right. Precision and recall are identical here because every read is assigned to exactly one journey and the ground truth covers every read, so false positives and false negatives are necessarily equal.

**"Did the numbers ever look wrong?"**
> Yes, and that's worth telling you. Our first full-city run scored 0.926 and quietly merged different vehicles. The cause was that the engine assumed each read had about 5 possible matches when the real number is ~218, so every candidate link started far too likely. Fixing that took IDF1 to 0.990 and more than halved the runtime. The pre-fix numbers are still in the repo.

---

## Deployment and scale

**"Will this work on a real city?"**
> At our tested scale — 50 cameras, 107,234 reads a day — association takes under 4 minutes on a laptop, and it processes windows independently, so it runs online. Scaling further means partitioning by region and running windows in parallel; the algorithm doesn't change. We deliberately avoided Kafka/Spark this week because demo scale doesn't need them; that's a documented choice, not an oversight.

**"What about privacy? This tracks people."**
> It does, and that deserves a straight answer. This is vehicle data, which under India's DPDP Act is personal data when linked to an owner. Our system stores plate reads and journeys, not identities — it never touches the registration database. A real deployment needs access control, retention limits, and an audit trail of who searched for what. The WHY panel helps here: every link has attached evidence, so a decision can be challenged rather than taken on faith.

**"What's the cost of a wrong link in the field?"**
> Serious, which is why the WHY panel exists and why we report confidence rather than a yes/no. An operator should see "91% same vehicle, plate evidence weak" and treat it accordingly. We'd recommend a confidence threshold for automated action and human review below it.

---

## Engineering

**"What's actually yours versus off-the-shelf?"**
> Ours: the probabilistic plate comparison, the three-channel fusion, the flow formulation and solver, consensus plate repair, partial-plate search, clone detection, the simulator, and the evaluation harness. Off-the-shelf: numpy/scipy, FastAPI, React, MapLibre. No pretrained model is doing the hard part.

**"Did you fine-tune any AI model?"**
> No, and we don't need to. There's no neural network in the core system — the models are small statistical fits (character confusion, travel-time distributions, appearance similarity, probability calibration) that fit in seconds. The optional video front-end would use pretrained detection and OCR models, where fine-tuning on Indian plates would be a refinement, not a requirement.

**"How do we know the code is correct?"**
> 150+ tests, including exactness guards: our solver's answers are checked against an independent library's optimum; the fast batched scorer is checked against the simple one to 1e-7; and a full-day test asserts every read lands in exactly one journey. Those exist because each one caught a real bug.
