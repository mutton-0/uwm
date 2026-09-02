# F-3 Temporal Averaging

> Work order: [`../docs/f3_final_cleanup_workorder.md`](../../docs/f3_final_cleanup_workorder.md), Part 2.
> Follows [`f3_corpus_extension_report_en.md`](f3_corpus_extension_report_en.md) (Part 1).
> Autonomous decisions: [`amendments.md`](amendments.md) §FC/A61–A62.
> Artifacts: `f3_tavg_{dd,ltf,ddv2,simlingo}.json`, `f3_tavg_summary.json`;
> scripts `scripts/f3_temporal_average.py`, `scripts/f3_tavg_summary.py`.

---

## 1. Motivation

Each of F-3's four arms currently performs a single forward pass per event ($a_{clean}$,
$a_{ghost}$, $a_{occ}$, $a_{ctrl}$, one frame each). Four of the six candidates are stopped at the
same step — "$b_{ghost}$'s CI crosses 0 ⇒ no baseline response" — and **it had never been ruled out
that this gate is dominated by single-frame noise**. This round adds a layer of averaging at the
**event level**: for each event we take a window of consecutive frames, run inference independently
on each, average, and then apply exactly the same statistics and adjudication, asking whether any
of those "no baseline response" cells is in fact a small real response masked by single-frame noise.

**This does not lengthen the models' input windows.** The multi-frame candidates
(Alpamayo/AutoVLA) keep their own temporal inputs unchanged; this round is run only on the four
**single-frame** candidates on the G1 corpus (DD / LTF / DDv2 / SimLingo), i.e. we run several
neighbouring decision points for the same decision point and average.

---

## 2. Method

**How the window is taken (no re-mining)**: we reuse the `geo` grid already computed during mining
(`scene_camera_frames` provides the filename and timestamp of every CAM_FRONT frame in the scene;
`per_obj[tok]["visible"]` provides per-frame visibility) — the same cache `f3_window_boxes` uses.

| Arm | Window rule |
| --- | --- |
| ghost | from the ghost frame **forward** in time, **requiring the entity to project visible on each frame**, capped at 10 |
| clean | from the clean frame **backward** in time (away from emergence), length-matched to the ghost window |
| occ | the **same ghost window**, each frame projected independently and painted with **that frame's own** mean colour |
| ctrl | the same ghost window, painting the control box at the **same position** on every frame (equal area, same eccentricity band, non-overlapping) |

All four arms remain symmetric and comparable in window length and number of occluded frames; the
shorter of the two sides is used so that **window length itself cannot become a confound**. The
measured mean window length is **9.9 frames** (cap 10; the cap is almost always reached).

**Why the clean side does not additionally require "entity not visible"**: the first version did,
and it discarded 4 of 4 events. The root cause is documented in Part 1 §4 (§FC/A61): in 219 of the
288 G1 class-A events (**76.0%**) the entity is already visible in the clean frame. This module's
job is to isolate **the single variable** of multi-frame averaging, so the clean window follows the
single-frame protocol exactly, with clean-frame visibility recorded as a per-event covariate and the
"entity genuinely absent" subset reported separately (§3.5).

**Pre-registered primary readout**: the scene-level bootstrap **CI half-width** of $b_{ghost}$
(single-frame vs multi-frame) — i.e. did averaging actually denoise. Secondary: whether the verdict
changes, $R$, and the within-frame erasure effect $d_{occ}$ with its control $d_{ctrl}$.
**The adjudication rules are word-for-word those of the single-frame version**; no threshold was
altered because the sampling changed. The single-frame comparison arm uses **the first frame of the
window on the same events**, so "single vs multi" is strictly paired and carries no event-set
difference.

---

## 3. Results

### 3.1 Primary readout: not one verdict changed, and the denoising is small and inconsistent

**Table 1. Single-frame vs multi-frame readouts for four candidates (G1 corpus, n = 282, mean window 9.9 frames).**

| Candidate | $b_{ghost}$ single | $b_{ghost}$ multi | CI half-width single→multi | ratio | Verdict single | Verdict multi | Changed? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DiffusionDrive | −0.0289 [−0.0859, +0.0216] | −0.0236 [−0.0475, +0.0009] | 0.0538 → 0.0242 | **0.450** | indeterminate (no baseline response) | indeterminate (no baseline response) | **no** |
| LTF | −0.0245 [−0.0384, −0.0102] | −0.0239 [−0.0392, −0.0079] | 0.0141 → 0.0156 | **1.106** | **FAIL** | **FAIL** | **no** |
| DiffusionDriveV2 | −0.1803 [−0.3487, +0.0032] | −0.1463 [−0.2974, +0.0097] | 0.1759 → 0.1536 | 0.873 | indeterminate (no baseline response) | indeterminate (no baseline response) | **no** |
| SimLingo | −0.4441 [−0.6979, −0.1897] | −0.3386 [−0.5865, −0.0856] | 0.2541 → 0.2505 | 0.986 | indeterminate ($R$'s CI spans 0.5) | indeterminate ($R$'s CI spans 0.5) | **no** |

**All four verdicts are unchanged.** The denoising ratios are 0.450 / 1.106 / 0.873 / 0.986
(mean 0.854) and are **not even consistent in direction** — LTF's CI became 11% *wider*.

⇒ **The work order's question — is any "no baseline response" in fact a small real response masked
by single-frame noise — is answered in the negative.** After roughly a tenfold increase in
inference per arm, $b_{ghost}$ for both affected candidates (DD, DDv2) remains indistinguishable
from 0. The value of this result is **confirmatory**: the "no baseline response" cells in the F-3
table **are not an artefact of single-frame sampling**.

**One substantive change in a point estimate (which does not change a verdict)**: LTF's necessity
ratio moves from $R$ = +0.113 [+0.044, +0.191] to **+0.058 [+0.008, +0.106]**. Both CIs sit far
below 0.5, so **the verdict is FAIL in both cases**; under averaging the FAIL is if anything more
complete (occluding the key entity removes about 6% of the response, versus 11% single-frame).

### 3.2 Why averaging cannot rescue $b_{ghost}$: that gate is dominated by between-event spread

Decomposing the variance of single-frame $b_{ghost}$ into within-window noise and genuine
between-event differences: if averaging removed **only** the within-window noise, the achievable
lower bound on the CI half-width ratio would be $\sqrt{1-f(1-1/L)}$, where $f$ is the noise variance
share and $L$ the window length.

**Table 2. Variance decomposition, predicted vs observed ratio.**

| Candidate | within-window sd($v_{ghost}$) | within-window sd($v_{clean}$) | between-event sd($b_{ghost}$) | noise variance share $f$ * | predicted ratio (most optimistic) | observed ratio |
| --- | --- | --- | --- | --- | --- | --- |
| DiffusionDrive | 0.0869 | 0.0826 | 0.4168 | 8.3% | 0.962 | **0.450** |
| LTF | 0.0378 | 0.0291 | 0.0963 | 24.6% | 0.883 | **1.106** |
| DiffusionDriveV2 | 0.5232 | 0.5174 | 1.3756 | 28.6% | 0.862 | 0.873 |
| SimLingo | 0.8181 | 0.6368 | 1.9499 | 28.3% | 0.864 | 0.986 |

\* $f = (\mathrm{sd}_{win}(g)^2 + \mathrm{sd}_{win}(c)^2)\,/\,\mathrm{Var}(b_{single})$. The
within-window sd includes the **systematic trend** inside the window (the entity keeps approaching
across the 10 frames), so $f$ is an **upper bound** on the noise share and the predicted ratio is
correspondingly the **most optimistic lower bound** on achievable denoising.

**Within-window noise accounts for only 8.3%–28.6% of the variance of single-frame $b_{ghost}$**:
even eliminating it entirely would only bring the CI half-width down to 0.86–0.96. The uncertainty
in $b_{ghost}$ comes mainly from genuine **between-event** differences, which averaging cannot touch
by construction. ⇒ To make the "no baseline response" cells adjudicable, the right lever is **more
events** (or a lower-variance contrast), **not** more inference passes within the same event.

### 3.3 DiffusionDrive's exception comes from heavy tails, not from noise

DD's observed ratio of **0.450 far exceeds** the most optimistic prediction of 0.962, so the
additive-noise model above **cannot explain it**. The cause:

**Table 3. Shape of the event-level $b_{ghost}$ distribution.**

| Candidate | kurtosis single → multi | $\lvert b\rvert$ p95 single | $\lvert b\rvert$ p99 single → multi |
| --- | --- | --- | --- |
| DiffusionDrive | **63.5 → 10.8** | 0.375 | **2.298 → 0.445** |
| LTF | 2.7 → 3.0 | 0.208 | 0.305 → 0.357 |
| DiffusionDriveV2 | 1.7 → 1.0 | 3.159 | 4.332 → 3.219 |
| SimLingo | 2.7 → 2.4 | — | 7.095 → 4.764 |

DD's single-frame $b_{ghost}$ has **kurtosis 63.5**, with p99 (2.298) **6.1×** its p95 (0.375) — a
handful of events whose single-frame planned speed takes an extreme value dominate the whole CI.
Window averaging drops the kurtosis to 10.8 and p99 to 0.445. ⇒ **DD's CI narrowing comes from
taming rare single-frame outliers, not from reducing typical per-frame noise.** The other three
candidates' single-frame distributions are not heavy-tailed to begin with (kurtosis 1.7–2.7), so
averaging yields no such dividend for them.

**This distinction is operationally meaningful**: multi-frame averaging is worth using on candidates
whose single-frame readout is **heavy-tailed**, and is not worth adopting as a blanket default —
it costs roughly 10× the inference and moved none of the four verdicts.

### 3.4 Within-frame erasure: averaging does help more here, but produced no new specific response

$d_{occ} = v_{plan}(\text{ghost}) - v_{plan}(\text{occ})$ is a paired contrast **within a single
frame** (Part 1 §4), bypassing the clean arm, and should in principle benefit more from averaging.

**Table 4. Within-frame erasure effect and its mandatory control arm.** (Sign convention: erasing
the hazard should make the model **speed up** ⇒ $d_{occ}$ **negative** is the expected direction.)

| Candidate | $d_{occ}$ single | $d_{occ}$ multi | CI half-width ratio | $d_{ctrl}$ multi | $\lvert d_{occ}\rvert-\lvert d_{ctrl}\rvert$ multi | Specific? |
| --- | --- | --- | --- | --- | --- | --- |
| DiffusionDrive | −0.0179 [−0.0589, +0.0041] | **−0.0084 [−0.0155, −0.0026]** | **0.204** | −0.0017 [−0.0094, +0.0042] | −0.0012 [−0.0069, +0.0037] | **no** |
| LTF | −0.0075 [−0.0110, −0.0043] | −0.0047 [−0.0077, −0.0017] | 0.905 | −0.0016 [−0.0046, +0.0014] | +0.0003 [−0.0024, +0.0027] | no |
| DiffusionDriveV2 | +0.0207 [−0.0385, +0.0854] | +0.0171 [−0.0070, +0.0419] | 0.395 | +0.0228 [−0.0036, +0.0529] | −0.0016 [−0.0268, +0.0206] | no |
| SimLingo | −0.1738 [−0.2905, −0.0678] | **−0.1863 [−0.2825, −0.1063]** | 0.791 | +0.0047 [−0.0470, +0.0544] | see below | **yes** |

The CI half-width ratios for $d_{occ}$ (0.204 / 0.905 / 0.395 / 0.791) are generally better than
those for $b_{ghost}$ — as expected, since it is a **within-frame pairing** in which both arms share
everything else about that frame.

**DD's $d_{occ}$ becomes significant under averaging** (single-frame CI spans 0 → multi-frame
−0.0084 [−0.0155, −0.0026]). But it **fails the specificity test**: the paired quantity
$\lvert d_{occ}\rvert-\lvert d_{ctrl}\rvert$ = −0.0012 [−0.0069, +0.0037] has a CI spanning 0, so
**it responds just as much to an equal-area grey patch placed elsewhere**. Under the existing
discipline (the ctrl arm is a mandatory control, not an optional one), this does not count as
"a hazard response uncovered from noise" but as "a **non-specific** perturbation sensitivity
uncovered from noise".

**Only SimLingo passes the full specificity test among the four** (both single- and multi-frame; on
G1, $\lvert d_{occ}\rvert-\lvert d_{ctrl}\rvert$ = +0.130 [+0.031, +0.242]). **No candidate moved
from non-specific to specific because of averaging.**

### 3.5 Exploratory subgroup: events where the clean arm is uncontaminated (**not used for adjudication**)

Looking at $b_{ghost}$ (multi-frame) on the **67** events (23.8%) where the entity is genuinely
**not** visible in the clean frame:

| Candidate | full set $b_{ghost}$ (multi) | uncontaminated subset (n = 67) |
| --- | --- | --- |
| DiffusionDrive | −0.0236 [−0.0475, +0.0009] | +0.0073 [−0.0399, +0.0587] |
| LTF | −0.0239 [−0.0392, −0.0079] | −0.0283 [−0.0533, −0.0022] |
| DiffusionDriveV2 | −0.1463 [−0.2974, +0.0097] | **−0.3186 [−0.6714, −0.0134]** |
| SimLingo | −0.3386 [−0.5865, −0.0856] | −0.4859 [−0.9979, +0.0195] |

DDv2's $b_{ghost}$ **becomes significant** on this subset (it spans 0 on the full set). **This does
not change any verdict and is explicitly marked exploratory**, for three reasons: (i) the subset is
selected by a **post-hoc covariate** (clean-frame visibility) rather than a pre-registered grouping;
(ii) n = 67 and four candidates were inspected simultaneously with no multiple-comparison control;
(iii) directionally, SimLingo moves the other way, from significant to spanning zero, so the subset
is not uniformly "more signal". The correct treatment is to record it as a **hypothesis worth
pre-registering next round** ("re-measure F-3 on events where the entity is genuinely absent"),
not as a readout of this round.

> **Instrument side.** All four arms share one window; the mean window length is 9.9 frames;
> per-frame visibility is verified by the same projection used at mining time. The single-frame
> comparison arm is the first frame of the window on the same events, so "single vs multi" is
> strictly paired. The variance decomposition shows within-window noise accounts for only 8.3%–28.6%
> of $b_{ghost}$'s variance, which explains the limited effect of averaging on the primary readout
> and shows **this round's null result is not because the averaging was done wrong**. The one
> candidate exceeding the model (DD, ratio 0.450 vs predicted 0.962) has been traced to heavy tails
> (kurtosis 63.5 → 10.8).
> **Specimen side.** All four F-3 verdicts are unchanged after roughly a tenfold increase in
> inference; LTF's FAIL is more complete under averaging ($R$ +0.113 → +0.058); and DD's newly
> significant within-frame erasure response **is not hazard-specific**, which is consistent with
> that candidate responding to any grey patch.

---

## 4. Discussion

**1. The work order's question receives a clear negative answer, and that is itself useful.**
After roughly a tenfold increase in inference per arm, not one of the four verdicts moved. Per the
work order this is written as **"the original verdicts are confirmed not to be a noise artefact"** —
the "no baseline response" cells of the F-3 table survive 10-frame averaging and are not a product
of single-frame sampling. **It is not inflated into "multi-frame averaging validated the F-3
method"**: this round validated only that these cells are insensitive to **temporal sampling**, and
nothing else about F-3.

**2. Averaging is not a reliable denoising lever, and the reason is quantifiable.**
Within-window noise is only 8.3%–28.6% of $b_{ghost}$'s variance; the rest is genuine between-event
difference, which averaging cannot touch by construction. Of the four ratios (0.450 / 1.106 / 0.873
/ 0.986), only DDv2's matches the prediction, LTF's got worse, and DD's far exceeds it for a reason
that is not noise. ⇒ **To make "no baseline response" adjudicable, add events, not passes.**

**3. A methodological by-product: within-frame pairings are better candidates for averaging than
cross-frame contrasts.** $d_{occ}$'s CI half-width ratios (best 0.204) are uniformly better than
$b_{ghost}$'s (best 0.450), because the within-frame pairing cancels everything else about the frame.
Should F-3's gate ever be rebuilt (raised but not attempted in Part 1 §4), this supports the view
that the numerator is the more stable half.

**4. The ctrl arm is again shown to be mandatory rather than optional.** DD's $d_{occ}$ becomes
significant under averaging. Without the ctrl arm this would read as "multi-frame averaging revealed
a hazard response masked by noise" — whereas the paired test shows it responds just as strongly to
an equal-area grey patch elsewhere. **The one newly significant effect this round is precisely the
one the mandatory control caught.**

---

## 5. Record of self-correction

1. **The first version of the multi-frame clean-side window required "entity not visible", which
   discarded 4 of 4 events.** Investigation confirmed this is a semantic problem with the clean arm
   itself (76.0% of events already show the entity), not a bug in the window logic. The rule was
   changed to match the single-frame protocol so as to isolate the single variable of averaging; the
   finding is written up separately in Part 1 §4 and §FC/A61. **The condition was not quietly
   relaxed to make the window run.**
2. **SimLingo's first run completed all 288 events and then failed at the write step**: `--out` was
   given as a relative path while `SimLingoRunner` does an `os.chdir` into its own repo. The output
   path is now resolved to absolute **before** the runner is constructed, and the run was repeated.
   **The log values from the first run were not substituted for the JSON** — the rerun reproduces
   them digit for digit (the model is seed-locked and deterministic), but substituting a log for a
   stored artifact is not acceptable practice, so the rerun was done anyway.
3. **The "denoising ratio" must not be read simply as the magnitude of denoising.** DD's 0.450
   exceeds the most optimistic additive-noise prediction of 0.962, showing the ratio mixes in a
   separate mechanism (outliers being tamed). This is quantified separately in Table 3, and **0.450
   is not offered as evidence that "averaging halves the noise"**.
4. **The work order's "extend it if it works" branch was not taken** (lead-brake / NAVSIM and the
   two VLAs). This round shows zero verdict changes and small, directionally inconsistent denoising
   ⇒ following the work order's instruction to "report honestly, and do not adjust the protocol in
   order to show this step was useful", **no extension was performed**. The cost (roughly 10×
   inference) does not match the demonstrated benefit.

---

## 6. Unfinished items

| Item | Status | Reason |
| --- | --- | --- |
| Extension to lead-brake / NAVSIM / the two VLAs | not done | this round shows a small and inconsistent effect; the work order's instruction for that branch is to report honestly rather than extend |
| Window-length sensitivity (L = 3 / 5 / 10) | not done | the primary readout already localizes the bottleneck to between-event variance rather than within-window noise, which changing L does not address |
| Pre-registered re-measurement on the "entity genuinely absent" subset | not done | this round's grouping is post-hoc, n = 67, and directionally inconsistent (§3.5); it belongs in a pre-registered round |
| Rebuilding F-3's gate around $d_{occ}$ | not done | a methodological change; the denominator question must be settled first (Part 1 §4) |
