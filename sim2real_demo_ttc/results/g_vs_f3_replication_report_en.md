# Cross-scenario and cross-data-source replication of G-VS / F-3

> Follows [`g_vs_f3_DONE.md`](g_vs_f3_DONE.md) (the v1 primary results on G1). This round re-runs both
> new axes on the **lead-braking** corpus (same source, new scenario structure) and on
> **NAVSIM/OpenScene** (same scenario structure, new data source), reusing the existing mining
> outputs without re-mining.
> Autonomous decisions: [`amendments.md`](amendments.md) §GF/A53–A55.

---

## 0. Direct answer to the core question

> Do the two conclusions established on G1 replicate on the other two corpora?

| Conclusion | Replication | Verdict |
| --- | --- | --- |
| **LTF's F-3 FAIL** (blind-action signature) | **Replicates on NAVSIM** (R = +0.008 [−0.068, +0.078]; b_ghost −0.0249 vs G1's −0.0245); **untestable** on lead braking (the baseline response vanishes) | ✅ **replicates wherever it is testable** |
| **DiffusionDrive's high G-VS selectivity + zero F-3 response** | "zero F-3 response" replicates **3/3**; "high G-VS selectivity" **1/3** (PASS on G1 only, indeterminate on the other two) | ⚠️ **one half replicates, the other does not** |

**And one finding that must come first**: this round uncovered a **methodological problem in G-VS not
previously exposed** — on the NAVSIM corpus the `position_only` floor (token coordinates alone)
**exceeds** the trained mIoU (DDv2 0.371 vs 0.355; SimLingo 0.411 vs 0.408). That is, **a trivial
baseline that knows only "where in the frame this token sits" outperforms the representation-based
probe.** Selectivity is a paired difference against `random_init` and can still be positive; but
"selectivity PASS" **does not mean** "the representation is more useful than knowing the coordinates".
See §3.

**We did not relax any standard because G-VS/F-3 are the axes this paper newly proposes as "more
objective"**: three-state adjudication, both required control arms and scene-level bootstrap were all
applied as before, and "no baseline response" is still reported separately from "FAIL". The
unfavourable finding above is a direct consequence of applying them unchanged.

---

## 1. F-3: cross-corpus replication

**Table 1. F-3 across the three corpora. $b_{ghost} = v_{plan}(\text{ghost}) - v_{plan}(\text{clean})$; necessity ratio $R = 1 - b_{occ}/b_{ghost}$.**

| Candidate | Corpus | n | $b_{ghost}$ | Sig.? | $R$ | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| **LTF** | G1 | 282 | −0.0245 [−0.0384, −0.0102] | **yes** | **+0.113 [+0.044, +0.191]** | **FAIL** |
| **LTF** | lead braking | 90 | −0.0068 [−0.0311, +0.0144] | no | — | indeterminate (no baseline response) |
| **LTF** | **NAVSIM** | 372 | **−0.0249 [−0.0399, −0.0099]** | **yes** | **+0.008 [−0.068, +0.078]** | **FAIL** |
| SimLingo | G1 | 282 | −0.4441 [−0.6979, −0.1897] | yes | +0.461 [+0.166, +0.789] | indeterminate |
| SimLingo | lead braking | 90 | +0.0520 [−0.2597, +0.3962] | no | — | indeterminate (no baseline response) |
| SimLingo | NAVSIM | 372 | −0.2482 [−0.5256, −0.0175] | yes | +0.114 [−0.399, +0.627] | indeterminate |
| DiffusionDrive | G1 / lead braking / NAVSIM | 282 / 90 / 372 | all CIs span 0 | no ×3 | — | **indeterminate ×3 (no baseline response)** |
| DiffusionDriveV2 | G1 / lead braking / NAVSIM | 282 / 90 / 372 | all CIs span 0 | no ×3 | — | indeterminate ×3 (no baseline response) |
| Alpamayo-R1 | G1 / lead braking | 99 / 51 | all CIs span 0 | no ×2 | — | indeterminate ×2 (no baseline response) |
| AutoVLA | G1 / lead braking | 119 / 88 | all CIs span 0 | no ×2 | — | indeterminate ×2 (no baseline response) |

### 1.1 LTF's FAIL replicates cleanly on NAVSIM

| Quantity | G1 | NAVSIM |
| --- | --- | --- |
| $b_{ghost}$ | −0.0245 [−0.0384, −0.0102] | **−0.0249 [−0.0399, −0.0099]** |
| $R$ (necessity ratio) | +0.113 [+0.044, +0.191] | **+0.008 [−0.068, +0.078]** |
| $R_{ctrl}$ (patch elsewhere) | +0.029 [−0.031, +0.081] | −0.033 [−0.132, +0.054] |
| Verdict | **FAIL** | **FAIL** |

$b_{ghost}$ is nearly identical digit for digit (−0.0245 vs −0.0249), both $R$ CIs sit far below 0.5,
and the ctrl arm is near 0 on both sides. **This is a rare cell in this line of work where both the
point estimate and the verdict replicate.** $R$ is even lower on NAVSIM (+0.008 — occluding the hazard
entity removes essentially none of the response).

**Untestable on lead braking**: LTF's $b_{ghost}$ falls to −0.0068 (CI spans 0). Under the standing
discipline that adjudicates as "indeterminate: the baseline response itself does not exist" and
**not** as FAIL. This is consistent with F① and the G axis being uniformly indeterminate on that
corpus (see `second_scenario_generality_report_*`).

### 1.2 "No baseline response" is the most stable readout in the whole table

DiffusionDrive **3/3**, DiffusionDriveV2 **3/3**, Alpamayo-R1 **2/2** and AutoVLA **2/2** have
$b_{ghost}$ indistinguishable from 0 on every corpus measured. **That is: "these four candidates'
planned speed shows no detectable response to hazard frames" holds across scenarios and across data
sources.**

The robustness of this negative conclusion **exceeds that of any positive conclusion in this paper**,
which is worth recording on its own.

### 1.3 The ctrl arm behaves correctly on all three corpora

All 16 computable $R_{ctrl}$ cells sit near 0 with CIs spanning 0, i.e. "an equal-sized grey patch
elsewhere" never returns the action to baseline. **This is the precondition for both of LTF's FAIL
verdicts**, and it holds on both new corpora.

---

## 2. G-VS: point estimates replicate, verdicts do not

**Table 2. G-VS selectivity = mIoU(trained) − mIoU(random_init), across the three corpora.**

| Candidate | G1 | Lead braking | NAVSIM | Point-estimate range | Verdict sequence |
| --- | --- | --- | --- | --- | --- |
| **SimLingo** | **+0.0273 [+0.013, +0.042] PASS** | **+0.0397 [+0.021, +0.059] PASS** | **+0.0350 [+0.019, +0.050] PASS** | 0.027–0.040 | **PASS ×3** |
| DiffusionDrive | **+0.0381 [+0.021, +0.055] PASS** | +0.0281 [−0.002, +0.058] indet. | +0.0171 [−0.000, +0.034] indet. | 0.017–0.038 | PASS → indet. → indet. |
| LTF | +0.0191 [−0.001, +0.038] indet. | +0.0209 [−0.007, +0.048] indet. | **+0.0239 [+0.007, +0.041] PASS** | 0.019–0.024 | indet. → indet. → PASS |
| DiffusionDriveV2 | +0.0159 [−0.003, +0.035] indet. | **+0.0409 [+0.007, +0.078] PASS** | **+0.0193 [+0.001, +0.037] PASS** | 0.016–0.041 | indet. → PASS → PASS |

**This table must be read on two levels, and they give different answers**:

* **At the point-estimate level: replication is quite good.** All 12 selectivity cells fall in the
  narrow band **[+0.016, +0.041]**; not one changes sign, not one collapses toward 0. This contrasts
  sharply with the old G axis, whose LTF positive fell from +0.070 to +0.011 on NAVSIM — the point
  estimate itself collapsed.
* **At the verdict level: replication fails.** Of the four candidates, only SimLingo passes all three
  times; the other three flip between PASS and indeterminate across corpora. The cause is that the
  **effect size (0.016–0.041) is the same order as the CI half-width (0.015–0.035)**, so the verdict
  turns on whether the CI happens to clear 0.

**The accurate statement**: G-VS's **readouts** are stable across scenarios and data sources, but
**at the current sample size and probe setting its three-state verdicts are not**. This is not "the
new axis is unreliable either"; it is "the new axis's effect is small enough that stable adjudication
needs a larger sample or a stronger readout". Directly actionable improvements are in §4.

---

## 3. A problem not previously exposed: on NAVSIM the position_only floor exceeds trained

**Table 3. Absolute mIoU for the three arms. `position_only` uses only each token's (row, col).**

| Candidate | Corpus | trained | random_init | **position_only** |
| --- | --- | --- | --- | --- |
| SimLingo | G1 / LB | 0.396 / 0.406 | 0.369 / 0.366 | 0.356 / 0.337 |
| SimLingo | **NAVSIM** | 0.4075 | 0.3725 | **0.4108 ← above trained** |
| DiffusionDrive | G1 / LB / NS | 0.382 / 0.387 / 0.384 | 0.344 / 0.359 / 0.367 | 0.333 / 0.331 / 0.371 |
| LTF | G1 / LB / NS | 0.367 / 0.371 / 0.384 | 0.348 / 0.350 / 0.360 | 0.333 / 0.331 / 0.371 |
| DiffusionDriveV2 | G1 / LB | 0.348 / 0.366 | 0.332 / 0.325 | 0.333 / 0.331 |
| DiffusionDriveV2 | **NAVSIM** | 0.3551 | 0.3358 | **0.3711 ← above trained** |

**On NAVSIM the `position_only` floor rises from 0.33 to 0.371**, exceeding trained mIoU in two cells.

**Why**: NAVSIM's camera is 1920×1080 with principal point (960, 560), so the 4:1 crop band covers a
different composition than nuScenes and objects are distributed more regularly within the frame ⇒
token coordinates alone predict better. This is a **corpus property**, not a model property.

**Consequences, which must be stated**:
1. **Selectivity (the paired difference against random_init) remains valid** — it controls for "probe
   capacity + spatial prior", and both arms share the same token coordinates, so the coordinate prior
   cancels in the paired difference. The four PASS verdicts are unaffected.
2. **But "G-VS PASS" must not be read as "the representation is more useful than knowing the
   coordinates".** On NAVSIM, for SimLingo and DDv2, a trivial coordinate baseline beats the
   representation probe in absolute mIoU.
3. **G-VS therefore currently supports only the weaker claim** "the representation contains
   object-ness information beyond random initialization", not the stronger claim "the representation
   provides practically useful object localization".

**This unfavourable finding is a direct consequence of applying the adjudication discipline
unchanged**: had we reported selectivity without the position_only arm, this round would not have
exposed it. **Of the two required control arms, this one turned out to be the more informative here.**

---

## 4. Robustness compared with the old G/F (the comparison the work order asked for)

**Table 4. Replication rates across the three corpora, new axes vs old.**

| Axis | Cross-scenario (lead braking) | Cross-data-source (NAVSIM) | Character |
| --- | --- | --- | --- |
| Old G (primary − D2cV floor) | **0/3 replicate** | **0/4 replicate** (incl. the only positive, LTF, at comparable power) | **point estimate collapses** (LTF +0.070 → +0.011) |
| Old F① (b-AUC) | 0/3 | 1/4 (only SimLingo, digit for digit) | both PASSes fell |
| Old C-hazard | **6/6** | **8/8** | architecture-level regularity, the most stable |
| **New G-VS** | verdicts 1/4; **point estimates 4/4 within a narrow band** | verdicts 2/4; **point estimates 4/4 within a narrow band** | **stable readout, unstable verdict** |
| **New F-3** | untestable ×6 (baseline responses all vanish) | **LTF's FAIL replicates**; "no baseline response" 3/3 | **replicates where testable; the negative conclusion is the most stable** |

**Three conclusions**:

1. **The new axes are more stable than the old G/F, but not as stable as C-hazard.** The old G's point
   estimate collapses (+0.070 → +0.011); the new G-VS's does not (all within [0.016, 0.041]). But
   G-VS's three-state verdicts still flip, falling short of C-hazard's 8/8.
2. **F-3 replicates well when its precondition holds** (LTF's FAIL twice, with nearly identical
   $b_{ghost}$), but **that precondition is itself unstable across corpora**: both SimLingo and LTF
   lose their baseline response on lead braking. **F-3's testability depends on the candidate having a
   response on that corpus, which is its structural limitation.**
3. **The most stable readout is a negative one**: "these four candidates show no detectable response to
   hazard frames" replicates 3/3, 3/3, 2/2, 2/2. The most robust conclusion in this line of work
   remains a negative conclusion.

---

## 5. Limitations

1. **G-VS still covers only 4/6 candidates** (the two VLAs' video-token layouts remain unconnected,
   §GF/A50), on all three corpora.
2. **G-VS's effect size is the same order as its CI half-width**, leaving verdicts on the boundary.
   Feasible improvements: a finer token grid (currently 8×32 for the TransFuser family, ≈ 50×50 px per
   token), more events, or concatenating features from several layers. **None was attempted this
   round** — doing so would no longer be "re-running on another corpus".
3. **The lead-braking F-3 has only 90 usable events** (103 LB positives minus those without a control
   box), 3–4× fewer than the other two corpora, so "no baseline response" there is partly attributable
   to power. Note however that DiffusionDrive and DDv2 also show no baseline response at n = 282 and
   372, so this is not purely a power issue.
4. **Alpamayo-R1 and AutoVLA were not measured on NAVSIM (n/a)**: their adapters depend on the nuScenes
   devkit and nuScenes `sd_token` respectively, and NAVSIM data is not in that format. This is a
   **stimulus-side interface gap**, not model-side unmeasurability.
5. **Only the A / LB positive classes were used**; B and C classes were not measured.
