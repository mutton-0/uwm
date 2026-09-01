# Cross-corpus generality test: nuScenes (G1) → NAVSIM/OpenScene, point by point

> Work order: [`../docs/navsim_openscene_independent_corpus_workorder.md`](../../docs/navsim_openscene_independent_corpus_workorder.md), **core deliverable**.
> Feasibility assessment and corpus QC: [`navsim_openscene_mining_report_en.md`](navsim_openscene_mining_report_en.md).
> Per-candidate numbers: `axis_navsim_corpus_{dd,ltf,ddv2}_en.md`.
> Autonomous decisions: [`amendments.md`](amendments.md) §NS/A45–A48.

---

## 0. Direct answer to the core question

> **Does LTF's G-axis positive (+0.070) replicate on an independent data source?**

### **No. And this time it is not a power problem.**

| | G1 (nuScenes) | Lead braking (same source, new scenario) | **This round (NAVSIM, independent source)** |
| --- | --- | --- | --- |
| Primary − falsification floor | **+0.070 [+0.017, +0.126]** ✅ | +0.032 [−0.098, +0.164] | **+0.011 [−0.038, +0.065]** |
| CI width | 0.109 | 0.262 (**2.4×**) | **0.103 (0.94×, slightly narrower)** |
| 10 fold-assignment seeds | +0.056 ± 0.028 | +0.000 ± 0.067 | +0.022 ± 0.022 |
| Positives n / floor n | 291 / 212 | 103 / 31 | **397 / 134** |
| Verdict | **PASS** | indeterminate (power-limited) | **indeterminate, but adequately powered ⇒ an informative non-replication** |

**When the lead-braking round adjudicated "indeterminate", the reason was a CI 2.4× wider** — power
alone explained the shrunken point estimate, so that round could neither confirm nor exclude.
**That reason does not apply here**: the CI width of 0.103 is **slightly narrower than G1's 0.109**
(more positives, 397 > 291, and more geometry-matched negatives, 382 > 283, offset the smaller floor,
134 < 212), while the effect fell from +0.070 to +0.011. **This is a non-replication under adequate
power.**

### What collapsed is not the primary readout but its gap to the floor

This must be stated precisely, or it will be misread as "LTF reads nothing on the new corpus":

| Quantity | G1 | NAVSIM | Change |
| --- | --- | --- | --- |
| Primary CV-AUC(A vs D2a) | 0.623 [0.577, 0.668] | **0.630 [0.592, 0.669], p = 3.1 × 10⁻¹⁰** | **stronger** |
| Falsification floor CV-AUC(A vs D2cV) | 0.553 | **0.620** | **up by 0.067** |
| Difference | **+0.070** | **+0.011** | collapsed |

**That is: on the new corpus LTF separates "A vs geometry-matched static objects" just as well or
better, but it separates "A vs same-class, same-geometry VRUs differing only in relative velocity"
equally well.** By D2cV's construction, the latter is a discrimination a **single-frame model is
structurally unable** to make (relative velocity requires inter-frame differencing). The direction is
therefore more likely reading **"a VRU is present"** than **"this VRU is dangerous"**.

**This is the entire reason the falsification floor exists.** Looking at the primary readout alone,
this round would conclude "LTF's G-axis positive replicated *more strongly* on an independent data
source (0.623 → 0.630, p = 3 × 10⁻¹⁰)" — an attractive conclusion, and a wrong one.

### Geographic sensitivity control: the conclusion is not driven by the Boston/Singapore overlap

nuScenes was collected in Boston and Singapore, and this corpus contains both (42% of events).
Re-running LTF on the **geographically disjoint** subset (Las Vegas + Pittsburgh, 8750 events):

| Subset | A / D2cV | Primary | Floor | Primary − floor |
| --- | --- | --- | --- | --- |
| Full | 397 / 134 | 0.630 | 0.620 | +0.011 [−0.038, +0.065] |
| **Geographically disjoint** | 248 / 79 | 0.629 | **0.673** | **−0.044 [−0.111, +0.038]** |

**On the subset with no geographic overlap with nuScenes at all, the effect is more negative.** The
non-replication is therefore not an artifact of geographic dilution.

### What is *not* claimed

This round **cannot** conclude that G1's +0.070 was noise. The two corpora differ in camera geometry,
annotation pipeline and city composition, and "the effect is real but corpus-sensitive" cannot be
distinguished from "the effect was noise all along" under this design. The accurate statement is:
**the positive is not robust across data sources and therefore cannot serve as evidence that "LTF
possesses a hazard concept".**

---

## 1. The G axis: all four candidates indeterminate on the independent source

**Table 1. G-axis primary readout (vision_mean), cell by cell against G1.**

| Candidate | NAVSIM primary | NAVSIM floor | **NAVSIM difference** | G1 difference | Verdict change |
| --- | --- | --- | --- | --- | --- |
| **LTF** | 0.630 [0.592, 0.669] | 0.620 | **+0.011 [−0.038, +0.065]** | **+0.070 [+0.017, +0.126]** ✅ | **PASS → indeterminate** |
| SimLingo | 0.575 [0.535, 0.615] | 0.584 | −0.009 [−0.060, +0.042] | +0.035 [−0.026, +0.097] | indeterminate → indeterminate |
| DiffusionDrive | 0.554 [0.514, 0.595] | 0.589 | −0.034 [−0.085, +0.018] | +0.009 [−0.047, +0.065] | indeterminate → indeterminate |
| DiffusionDriveV2 | 0.573 [0.533, 0.613] | 0.525 | +0.048 [−0.020, +0.111] | +0.025 [−0.044, +0.097] | indeterminate → indeterminate |

**All four differences lie within [−0.034, +0.048] with CIs spanning zero.** The one candidate that
crossed the floor on G1 does not cross it here.

**One symmetric observation worth recording**: LTF still has the highest **raw** primary readout of
the four (0.630 against 0.554–0.575), matching its G1 ranking (LTF 0.623, highest there too).
**"Which model's representation most linearly separates positives from negatives" replicated across
data sources; "whether that separation constitutes a hazard concept" did not.**

---

## 2. The C axis: the architecture-level regularity replicates **a third time**, now across data sources

**Table 2. C-hazard across the two data sources, cell by cell.**

| Candidate | Family | Corpus | Spearman | Profile | Responsible-layer mode | Commitment layer | patch-ALL | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SimLingo | pure transformer | G1 | −0.997 | cascade | L0 | L3 / 24 | +1.009 | indeterminate (premise) |
| SimLingo | pure transformer | **NAVSIM** | **−0.997** | **cascade** | **L0** | **L4 / 24** | +1.011 | indeterminate (premise) |
| LTF | TransFuser | G1 | +0.929 | interior peak | L6 | none | +1.000 | PASS |
| LTF | TransFuser | **NAVSIM** | **+0.786** | **interior peak** | **L7** | **none** | +1.000 | **PASS** |
| DiffusionDrive | TransFuser | G1 | +0.929 | interior peak | L6 | none | +1.000 | PASS |
| DiffusionDrive | TransFuser | **NAVSIM** | **+0.857** | **interior peak** | **L6** | **none** | +1.000 | **PASS** |
| DiffusionDriveV2 | TransFuser | G1 | +0.810 | interior peak | L4 | none | +0.552 ✗ | indeterminate (self-check) |
| DiffusionDriveV2 | TransFuser | **NAVSIM** | **+0.667** | **interior peak** | **L4** | **none** | **+0.642 ✗** | **indeterminate (self-check)** |

Four things hold at once:

1. **The signs separate perfectly by architecture family, with no exception on either data source**
   (pure transformer ρ < 0, TransFuser family ρ > 0);
2. **The presence or absence of a commitment layer is identical per candidate** (absent for all three
   TransFuser members on both corpora, present for SimLingo on both);
3. **The responsible-layer mode barely moves** (SimLingo L0 → L0, DD L6 → L6, LTF L6 → L7,
   DDv2 L4 → L4);
4. **Even the self-check failure replicates** — DiffusionDriveV2's patch-ALL lands at 0.55–0.64 on
   both corpora and adjudicates indeterminate both times. **A candidate-specific failure signature
   replicated across data sources.**

**The regularity has now crossed three dimensions**: architecture family (§CE/A39, six candidates),
scenario causal structure (lead braking), and **data source (this round)**. It is the only conclusion
in this line of work that has replicated along all three.

SimLingo's Spearman is **−0.997** on both corpora, with the commitment layer moving only L3 → L4
(depth 0.17 → 0.21) — **both the shape class and the depth are stable**, more so than in the
lead-braking round (L3 → L8). This again supports §CE/A40's judgement: "cascade" is
architecture-determined and "plateau width" is not, but the width is not arbitrary either.

---

## 3. F①: the distribution of the three failure shapes is **partly** consistent

**Table 3. F① b-AUC(A vs D2a) and b(A) across the two corpora.**

| Candidate | G1 b-AUC | NAVSIM b-AUC | G1 b(A) | NAVSIM b(A) | Shape change |
| --- | --- | --- | --- | --- | --- |
| SimLingo | 0.534 [0.469, 0.592] | **0.534 [0.481, 0.585]** | +0.307 sig. | +0.188 n.s. | non-specific → **non-specific (b-AUC replicates digit for digit)** |
| **LTF** | **0.583 [0.519, 0.639]** ✅ | 0.546 [0.496, 0.593] | +0.020 sig. | +0.024 sig. | **specific → indeterminate** (CI lower bound 0.4957, on the line) |
| DiffusionDrive | 0.553 [0.486, 0.623] | 0.472 [0.406, 0.539] | +0.010 n.s. | −0.016 n.s. | indeterminate → indeterminate |
| **DiffusionDriveV2** | **0.556 [0.501, 0.613]** ✅ | 0.471 [0.421, 0.524] | +0.201 sig. | −0.078 n.s. | **PASS → indeterminate** |

* **SimLingo's b-AUC replicates digit for digit** (0.534 → 0.534, with a slightly narrower CI). This
  is the most stable cell in the table.
* **LTF's b(A) also replicates** (+0.020 → +0.024, significant both times), but its b-AUC fell from
  0.583 to 0.546 with a CI lower bound of 0.4957 — on the 0.5 line, adjudicated indeterminate. Its
  b-AUC against the falsification floor D2cV is only **0.510**, i.e. the action side likewise fails
  to separate "dangerous VRU" from "harmless VRU".
* **Both candidates that PASSed on G1 (LTF, DDv2) fall to indeterminate on the independent source.**
* **DiffusionDrive's and DiffusionDriveV2's b-AUC both land below 0.5** (0.472 / 0.471), and DD's
  b-AUC against D2cV is **0.407 (MWU p = 0.0013)** — significantly **below** 0.5, meaning its action
  responds *more* to **harmless** VRUs. This is a new failure shape not seen on G1; recorded, but not
  over-read (one corpus, one candidate).

**Conclusion**: F①'s "reacts non-specifically" shape replicates across data sources (SimLingo digit
for digit); **the "reacts specifically" shape does not** — both of G1's PASSes fell.

---

## 4. Summary: adjudication of the three questions

| # | Question | Verdict | Basis |
| --- | --- | --- | --- |
| **1** | Does LTF's G-axis positive replicate across data sources? | ❌ **No (adequately powered)** | +0.070 → +0.011 with CI width 0.103 vs 0.109; more negative still (−0.044) on the geographically disjoint subset |
| **2** | Does the C-axis architecture-level regularity cross one more data source? | ✅ **Yes** | 4 candidates × 2 corpora = 8 cells, all consistent; non-overlapping signs; commitment-layer presence, responsible-layer mode and even the self-check failure all replicate |
| **3** | Is the distribution of F①'s three shapes consistent? | ⚠️ **Partly** | "non-specific" replicates (SimLingo 0.534 → 0.534); **"specific" does not** (both PASSes fell) |

**One observation that runs through both generality rounds**, consistent with and stronger than the
lead-braking round's: **the mechanism side (C) is far more robust than the representation side (G)
and the action side (F).** Changing scenario structure: C replicated 6/6, G 0/3. Changing data
source: C replicated 8/8, G 0/4, F① 1/4. The C axis measures "at which layer information enters the
network" — a routing property independent of the stimulus; G and F measure "whether some direction or
action separates two classes of stimuli" — which **depends directly on what those classes are**.

---

## 5. Effect on the paper's global conclusions

**This round changes a conclusion already written into the paper body.**

§4.2.6 and §4.4.1 previously used LTF's G-axis positive as a **positive calibration**: "under the
same D2cV floor some candidate did cross, so the other candidates' indeterminacy cannot be attributed
to the floor being uncrossable."

**That argument's premise is now weakened**: the positive does not replicate on an
adequately-powered independent data source, and turns negative on the geographically disjoint subset.
It therefore **cannot support the general claim "the floor is crossable"**, only the narrower "on the
G1 corpus, that floor was crossed once".

The body has been rewritten accordingly (see `paper_experiments_section_{zh,en}.md` §4.2.6 / §4.4.1 /
§4.5). **The original readout was not deleted**; its scope was narrowed and annotated with the
cross-corpus result.

---

## 6. Limitations

1. **"Independently collected" ≠ "geographically disjoint"**: this corpus includes Boston and
   Singapore, overlapping nuScenes. A geographically disjoint sensitivity analysis is provided (same
   conclusion, stronger), but that subset halves the sample.
2. **D2cV's n = 134** is the one quantity smaller than G1's (212). The CI width of the
   primary-minus-floor readout is recovered by the larger A and D2a (0.103 vs 0.109), but the floor
   side alone is less precise than G1's.
3. **Imaging geometry is not comparable across corpora** (different resolution, principal point, and
   NAVSIM has distortion while our projection is pinhole). Every readout this round is a within-corpus
   between-group contrast and is unaffected, but absolute area/ecc values must not be quoted across
   corpora.
4. **Only the test split was used** (147 logs). NAVSIM's trainval is not downloaded; there is room to
   scale up.
5. **Alpamayo-R1 and AutoVLA were not measured** (excluded by the work order). On the data-source
   dimension the C-axis regularity therefore rests on 1 pure-transformer stack against 3
   TransFuser-family members, less balanced than the 3 vs 3 of §CE/A40.
