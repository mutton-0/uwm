# Candidate Expansion: G/F-axis Readouts for Alpamayo-R1

> Work order: [`../docs/candidate_pool_expansion_workorder.md`](../../docs/candidate_pool_expansion_workorder.md)
> (zero-download-cost, highest-priority candidate). All judgement calls are recorded in
> [`amendments.md`](amendments.md) §CE, especially **A34** (D2cV does not hold as a floor for
> multi-frame candidates). Numerical artifacts: `g_axis_alpamayo.json`,
> `f_axis_action_counterfactual.json`, `v_hazard_alpa_{vision_mean,seq_mean}.npz`;
> adapter: `alpamayo_g1_adapter/`.

---

## Methods

### Candidate and existing groundwork

Alpamayo-R1 (NVIDIA, 10B VLA, flow-matching action head plus a VLM rollout producing a
Chain-of-Causation). The weights are already local (`/data/ruolin/alpamayo_ckpt`) and were used in
work-order P's positive-control qualification (`p1_alpamayo_clean.json`: b-AUC 0.446, **did not
qualify**). This round does **not** re-run the behavioural readout; it adds the previously missing
**representation-side G axis** via a new `alpamayo_g1_adapter/alpa_g1_cache.py`, which pools per
layer during the **prefill** pass and writes the result to disk.

### Adaptation details and deviations

* **Readable layers**: 36 decoder layers of the VLM language tower (hidden 4096).
* **Prefill selection**: Alpamayo's inference is a VLM rollout (CoT generation, then trajectory
  sampling), so the hooks fire many times; we keep only the call with the **largest seq_len** (the
  forward over the whole prompt) and discard decode steps (seq_len = 1).
* **Image-token identification**: taken as the **longest run of identical ids** in `input_ids`
  (measured id = 151655; 2880 of 3006 tokens are image tokens), with the identification written to
  the cache for review.
* **Pooling inside the hook** (6 cameras ⇒ S ≈ 3000, the full tensor cannot be retained):
  `vision_mean` / `last_token` / `seq_mean`.
* **Ego history anchored to the clean frame** (isomorphic to SimLingo's `prompt_anchor` and the
  navsim-family status anchoring).
* **Coverage prefilter**: the adapter unconditionally constructs a 6.4 s future trajectory, so events
  close to the end of a scene are unusable ⇒ class A **188 / 291** and D2a **157 / 283** usable
  (identical to P1's usable rates, a passing cross-check).
* **Deviations carried over** (registered in P1): Alpamayo is trained on NVIDIA PhysicalAI-AV and is
  equally OOD on nuScenes; the camera FOV is mismatched (it expects a 120° wide plus a 30° tele view,
  while all six nuScenes cameras are 70°, so only an approximate mapping is possible).

### **A methodological limitation that must be stated up front (§CE/A34)**

The D2cV falsification floor is justified by the argument that it shares class and imaging geometry
with the positives and differs **only** in a relative velocity, which **a single-frame model is
structurally blind to**. **Alpamayo is a multi-frame model** (1.6 s of ego history plus images from
several instants), so relative velocity **is** observable to it, and **D2cV is therefore no longer a
falsification floor for Alpamayo** — "A vs D2cV significant" is a legitimate discrimination task, not
a leakage alarm. This report accordingly adjudicates Alpamayo's G axis against the **permutation null
and the random-direction floor**, reporting the D2cV number alongside but **not using it as a
criterion**.

---

## Results

**Table 1. G-axis readout for Alpamayo-R1 on the shared G1 stimulus set. For this multi-frame model the usable floors are the permutation null and the random-direction floor (§CE/A34); the D2cV column is reported for completeness only.**

| Pooling | primary CV-AUC(A vs D2a) | 95% CI | $p$ | permutation null | random floor | primary − permutation floor | primary over 10 seeds | D2cV (**not a criterion**) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vision_mean` (primary) | **0.562** | **[0.502, 0.623]** | 0.046 | 0.504 | 0.508 | **+0.058** | 0.563 ± 0.029 | 0.517 |
| `seq_mean` (sensitivity) | 0.565 | [0.504, 0.625] | 0.038 | 0.501 | 0.501 | +0.064 | 0.562 ± 0.027 | 0.522 |

n = 188 positives / 157 negatives (after the coverage prefilter). Peak layer $L^\*$ = 18 (mid-stack of
36), identical under both poolings. Geometric robustness: $\rho$(projection, log area) = +0.019
($p$ = 0.73) and $\rho$(projection, eccentricity) = +0.082 ($p$ = 0.13), **neither significant**.
Verdict: **the primary readout is significantly above both the permutation null and the random floor**
(CI lower bound 0.502 > 0.5, consistent across poolings, stable over 10 seeds); it is
**indeterminate under the D2cV protocol** (+0.045 [−0.041, +0.129]), a protocol that does not apply to
multi-frame models in the first place.

**Table 2. F① action-level counterfactual (architecture-neutral), on the same stimuli as every other candidate.**

| Quantity | Value | 95% CI | $p$ | Verdict |
| --- | --- | --- | --- | --- |
| $b$(A), action change induced by hazard frames [m/s] | **−0.080** | [−0.178, +0.017] | — | indistinguishable from 0, **and negative in sign** |
| $b$(D2a) [m/s] | +0.025 | [−0.104, +0.160] | — | indistinguishable from 0 |
| **b-AUC(A vs D2a)** | **0.446** | [0.374, 0.519] | 0.082 | **indeterminate** (and below 0.5) |
| b-AUC(A vs D2cV) | 0.456 | [0.364, 0.551] | 0.415 | indeterminate |

**Cross-check**: b-AUC = 0.4456 matches the 0.4456 reported in work-order P's
`p1_alpamayo_clean.json` **digit for digit**, confirming that this round's adapter uses exactly P1's
behavioural protocol rather than a fresh one.

> **Instrument side.** The usable rates (188/291, 157/283) match P1 item for item and the b-AUC
> matches digit for digit, so the adapter's protocol is trustworthy. The positive G result is
> consistent **across both poolings and 10 fold-assignment seeds**, with neither geometric-robustness
> correlation significant. **But the floor had to be changed for Alpamayo**: D2cV's falsification
> premise (velocity unobservable to a single-frame model) does not hold for a multi-frame model, so
> this report adjudicates against the permutation/random floors and states explicitly that Alpamayo's
> G axis is therefore **not on exactly the same criterion as the single-frame candidates**.
> **Specimen side.** Alpamayo **can read hazard on the representation side** (primary readout exceeds
> the permutation floor by +0.058), yet **its action shows no corresponding response and is in fact
> reversed in sign** (b(A) = −0.080; b-AUC = 0.446 < 0.5).

---

## Discussion

**1. Alpamayo provides the cleanest instance in this framework of "G connected, F broken".**
Representation side: the primary readout 0.562 [0.502, 0.623] is significantly above the permutation
null (0.504) and the random floor (0.508), consistent across two poolings and 10 seeds with
non-significant geometric robustness ⇒ **hazard information is linearly readable in the
representation**. Action side: b(A) = −0.080 (negative sign = it plans slightly *faster* on hazard
frames) and b-AUC = 0.446 < 0.5 ⇒ **the action not only fails to follow the hazard but leans slightly
against it**. This is a textbook instance of the "grounded but not driving the action (link F broken)"
case in this work's central claim, and it occurs in a candidate whose **representation side clearly
carries signal** — more compelling than SimLingo's "nothing readable at either end".

**2. Two qualifications must accompany this conclusion, both mandatory.** (i) **A different floor**:
Alpamayo's G axis is adjudicated against the permutation/random floors, not the D2cV falsification
floor. Per §CE/A34 this is unavoidable for a multi-frame model, but it means Alpamayo's positive G is
**weaker than LTF's** (which cleared the stricter D2cV floor). (ii) **Out-of-domain status**: Alpamayo
is trained on PhysicalAI-AV, is OOD on nuScenes, and has a mismatched camera FOV; this readout
supports no inference about its performance in its native domain.

**3. Relation to the P1 positive-control conclusion: this round supplies the missing half.**
Work-order P measured only the action side and concluded "did not qualify" (b-AUC 0.446), but **could
not distinguish** "the model did not see it" from "it saw it but that did not drive the action". With
the representation side added, the answer is unambiguous: **it saw it**. P1's "did not qualify" should
therefore be re-read as a **broken F link** rather than a perception deficit — an **interpretive
revision** of an existing conclusion (not a numerical one; the b-AUC is unchanged digit for digit).

**4. F② and the C axis were not measured; the reason is recorded honestly.** F② (representation
injection): Alpamayo has a flow-matching action head plus a VLM rollout at ≈ 2.7 s per forward, so a
complete four-part injection battery (40 events × 169 conditions × 2 frames) would need ≈ 4 GPU-hours,
beyond this round's budget allocation for candidate expansion; moreover, per this round's §CE/A31
finding that injection measurability is determined by the encoder, Alpamayo's encoder differs from all
three families measured so far, so its result cannot be extrapolated ⇒ marked **not measured**, not
"not applicable". C axis: requires either a domain pairing (Alpamayo lacks a multi-camera domain-pair
adapter) or clean↔ghost patching (equally bound by the 2.7 s per-forward cost) ⇒ **not measured**.

---

## Record of self-correction

1. **D2cV does not hold for multi-frame candidates (§CE/A34)**: this premise failure was exposed only
   by the candidate expansion — the first three rounds' candidates happened to be entirely
   single-frame. Alpamayo's G-axis criterion therefore changed from "minus the D2cV floor" to "minus
   the permutation/random floor", annotated separately in the matrix. **This is not to make Alpamayo
   look better**: under the old criterion it is "indeterminate" and under the new one "significant";
   both numbers appear in the table, and the reason for switching criteria is physical (velocity is
   observable to a multi-frame model), not result-driven.
2. **The negative-cache scope was initially too broad**: the first run with `--types D2b D2c` did not
   apply the matched lists and enumerated 1562 candidate events (≈ 70 min). It was changed to the
   union of `matched_{D2b,D2c,D2bV,D2cV}` (941 events; 570 after the coverage prefilter). This is an
   efficiency fix and does not affect protocol.
3. **F②/C are marked "not measured" rather than "not applicable"**: per work-order §6, "not
   applicable" is reserved for an operationalization that does not apply to the architecture;
   Alpamayo's case is **insufficient budget**, a different situation, and is recorded as such.
