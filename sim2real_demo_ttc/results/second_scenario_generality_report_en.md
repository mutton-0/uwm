# Generality test for the second scenario type: ghost probe → lead-vehicle braking, point by point

> Work order: [`../docs/severity_gradient_and_second_scenario_workorder.md`](../../docs/severity_gradient_and_second_scenario_workorder.md), task two, **core deliverable**.
> Corpus and matching QC: [`lead_vehicle_brake_mining_report_en.md`](lead_vehicle_brake_mining_report_en.md).
> Per-candidate numbers: `axis_lead_vehicle_brake_{simlingo,dd,ltf}_en.md`.
> Autonomous decisions: [`amendments.md`](amendments.md) §LB/A42–A44.

---

## 0. One-sentence conclusion

**All four axis definitions transferred; of the three previously established patterns, one replicated
fully, one is unresolvable at this power, and one was contradicted.**

| Pattern established on the ghost-probe scenario | Result on the second scenario | Verdict |
| --- | --- | --- |
| **The C-axis architecture-level regularity** (pure transformer = cascade + commitment layer; TransFuser family = interior peak @L6 + no commitment layer) | **Replicated point by point across all three candidates; non-overlapping signs; the same responsible layer L6** | ✅ **replicated** |
| **LTF's G-axis positive** (+0.070 [+0.017, +0.126]) | +0.032 [−0.098, +0.164]: point estimate halved, CI 2.4× wider | ⚠️ **indeterminate (power-limited)** |
| **LTF's F① specificity** (b-AUC 0.583, PASS) | 0.560 [0.471, 0.651]: point estimate almost unchanged, CI 1.5× wider | ⚠️ **indeterminate (power-limited)** |
| **SimLingo's G axis, "readable but not grounded"** | Primary readout 0.600 is significant, but the **falsification floor is higher at 0.645** | ❌ **overtaken by the floor in the new scenario** |

---

## 1. Method transfer itself: the definitions transferred, with no formula changed

The work order asks whether the four axes' **definitions** transfer, not whether thresholds do. The
operational evidence:

| Component | Changed? | Note |
| --- | --- | --- |
| Geometry computation | **unchanged** | `g1_mine_events.compute_scene_geometry` / `frame_record` / `pick_frames` imported directly |
| clean / ghost windows | **unchanged** | [−1.5, −0.5] s / [0.0, 1.0] s, 2 frames per condition |
| The three input adapters | **unchanged** | SimLingo / DiffusionDrive / LTF only had `--work` repointed at the new corpus |
| G-axis script | **class names parameterized only** | `--pos LB --neg LBn --floor LBv`; formulas, fold count, floor convention and null distributions untouched |
| F① script | **class names parameterized only** | as above |
| C-hazard script | **work dir and positive-class name parameterized only** | patching method, sufficient-cut-set check and shape diagnostic untouched |

**A regression check was run on the G1 corpus after parameterization**: LTF's G axis
+0.070 [0.017, 0.126] and F① 0.583 [0.519, 0.639] reproduce digit for digit, confirming the changes
did not disturb the original readouts.

**This is already half of what the work order asked**: had the four axes required new formulas for a
new causal structure, the claim "the definitions transfer" would have been refuted on the spot. They
did not.

---

## 2. The C axis: the architecture-level regularity **replicates in full** — the strongest generalization evidence this round

**Table 1. C-hazard across the two scenarios. The pairing sources are entirely different: ghost probe pairs clean (hazard absent) ↔ ghost (hazard present); lead braking pairs clean (steady following) ↔ ghost (hard braking).**

| Candidate | Family | Scenario | Spearman(layer, recovery) | Profile | Responsible-layer mode | Commitment layer | patch-ALL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SimLingo | pure transformer | ghost probe | −0.997 | cascade | L0 | L3 / 24 | +1.009 |
| SimLingo | pure transformer | **lead braking** | **−0.921** | **cascade** | **L0** | **L8 / 24** | +0.965 |
| LTF | TransFuser | ghost probe | +0.929 | interior peak | L6 | none | +1.000 |
| LTF | TransFuser | **lead braking** | **+0.786** | **interior peak** | **L6** | **none** | +1.000 |
| DiffusionDrive | TransFuser | ghost probe | +0.929 | interior peak | L6 | none | +1.000 |
| DiffusionDrive | TransFuser | **lead braking** | **+0.952** | **interior peak** | **L6** | **none** | +1.000 |

Three things hold at once:

1. **The signs separate perfectly by architecture family, with no exception in either scenario** —
   pure transformer stacks ρ < 0, TransFuser family ρ > 0, with no overlap between them;
2. **The responsible-layer mode is identical per candidate across scenarios** — SimLingo L0 in both,
   LTF and DiffusionDrive **L6** in both;
3. **The presence or absence of a commitment layer is identical per candidate** — absent for the
   TransFuser family in both scenarios, present for SimLingo in both.

All three candidates pass the patch-ALL sufficient-cut-set self-check in both scenarios, so this is
not an instrument artifact.

**Why this generalization matters**: §CE/A39–A40 upgraded "profile shape is determined by
residual-stream topology" from a prediction to "an architecture-level regularity verified on six
candidates" — but those six candidates all shared **one** ghost-probe corpus. This round changed to a
scenario with a **different causal structure**, and the regularity replicated point by point. **That
lifts the claim from "holds for one class of stimuli" to "holds for the architecture"** — the only
one of the established patterns to reach that strength.

**One magnitude change recorded honestly**: SimLingo's commitment layer moved from L3 (ghost probe)
to L8 (lead braking), depth 0.17 → 0.38. **The shape class is unchanged; the plateau width is not** —
which independently confirms §CE/A40's correction that "cascade" is architecture-determined while
"how wide the plateau is" is not. This round supplies a **cross-scenario** corroboration of that
correction.

---

## 3. The G axis: LTF's positive **does not replicate, but this is largely a power problem**

**Table 2. G-axis primary readout (primary minus own falsification floor) across the two scenarios.**

| Candidate | Ghost probe (A vs D2a − D2cV) | Lead braking (LB vs LBn − LBv) | CI width ratio |
| --- | --- | --- | --- |
| **LTF** | **+0.070 [+0.017, +0.126]** (PASS) | +0.032 [−0.098, +0.164] (indeterminate) | 0.262 / 0.109 = **2.4×** |
| SimLingo | +0.035 [−0.026, +0.097] (indeterminate) | **−0.045 [−0.177, +0.081]** (indeterminate, negative point estimate) | 2.1× |
| DiffusionDrive | +0.009 [−0.047, +0.065] (indeterminate) | −0.127 [−0.263, +0.003] (indeterminate, negative point estimate) | 2.4× |

**LTF's verdict must be written as "indeterminate", not "did not replicate".** Because:

* the CI is **2.4×** as wide as in the ghost-probe scenario (samples 103/48/31 against 291/283/212);
* but **the point estimate did shrink** (+0.070 → +0.032), and the 10-seed fold-assignment mean of
  the difference is **+0.000 ± 0.067**, i.e. in the primary pooling (vision_mean) the effect has
  fallen to about zero;
* the sensitivity pooling region_mean gives +0.061 ± 0.045 over 10 seeds, inconsistent with the
  primary.

**The correct statement is therefore**: "On a second scenario with roughly one third the sample,
LTF's G-axis positive was not replicated; the CI width alone would explain this, but the primary
pooling's point estimate also fell to about zero, so this round **can neither confirm nor exclude**
that the positive holds under the new causal structure." Calling it "failure to generalize"
over-reads; calling it "merely underpowered" evades.

**Scaling up is a feasible route to a verdict**: relaxing the LB threshold to $a \le -2.0$ would
raise the positives to ≈ 150, but the real bottleneck is LBv (only 37 candidates in the whole
trainval; see the mining report).

---

## 4. SimLingo's G axis: the new scenario produced a **floor-overtakes-primary** result

This is the one pattern this round that is **contradicted** rather than merely indeterminate, so it
must be stated plainly.

| Quantity | Value |
| --- | --- |
| Primary CV-AUC(LB vs LBn) | **0.600 [0.505, 0.694], p = 0.0495** ⇒ significantly above chance |
| Falsification floor CV-AUC(vs LBv) | **0.645** ⇒ **higher than the primary** |
| Primary − floor | −0.045 [−0.177, +0.081]; 10 seeds −0.033 ± 0.036 |

**How to read it**: SimLingo's representation **can** separate hard-braking lead vehicles from
steady-following ones (0.600, significant). But the same direction separates hard-braking lead
vehicles from **mildly decelerating** ones even better (0.645). LBv is matched to LB on distance,
lead speed and ego speed, and **is also decelerating** — so what this direction reads is not "this
vehicle is braking hard" but something **non-monotonic** in braking severity.

**This is exactly why the falsification floor exists**: reporting only the primary 0.600 with
p = 0.0495 would yield the conclusion "SimLingo reads lead-vehicle hard braking in the new
scenario", and that conclusion would be wrong. **The new scenario's floor caught a would-be false
positive on its first use.**

**Qualification**: LBv's matching quality is the weakest in this batch ($d_{long}$ SMD −0.328, lead
speed −0.291; see the mining report), and the residual imbalance may itself contribute to the
overtaking. The accurate statement is therefore "**the primary readout cannot be separated from the
falsification floor, and the direction is unfavourable**", not "it is proven that SimLingo is reading
something else".

---

## 5. F①: all three candidates indeterminate; LTF's point estimate essentially holds

**Table 3. F① b-AUC (positive vs geometry-matched negative) across the two scenarios.**

| Candidate | Ghost probe | Lead braking | Δ point estimate | CI width ratio |
| --- | --- | --- | --- | --- |
| **LTF** | **0.583 [0.519, 0.639]** (PASS) | 0.560 [0.471, 0.651] (indeterminate) | −0.023 | 1.50× |
| SimLingo | 0.534 [0.469, 0.592] (indeterminate) | 0.491 [0.383, 0.589] (indeterminate) | −0.043 | 1.68× |
| DiffusionDrive | 0.553 [0.486, 0.623] (indeterminate) | 0.502 [0.411, 0.596] (indeterminate) | −0.051 | 1.35× |

**LTF is the only candidate whose point estimate barely moves** (0.583 → 0.560, a shift of 0.023 far
below the CI half-width of 0.090), and its b-AUC against the falsification floor LBv is **0.594**
(higher than the 0.560 against LBn) — the same direction as in the ghost-probe scenario. **This is
the textbook shape of "the point estimate replicates, the significance is lost to power"**, which is
a different situation from the G-axis case in §3 where the point estimate shrank as well; the two
must not be conflated.

**All three candidates' $b$(LB) are indistinguishable from zero**, which is itself worth recording:
in the lead-braking scenario, all three models' planned speed shows **no detectable response** to a
lead vehicle beginning to brake hard. In the ghost-probe scenario at least SimLingo (+0.307) and
DiffusionDriveV2 (+0.201) did react. **Once the causal structure changes, the action-side response
weakens** — but this round's sample is not sufficient to establish that as a conclusion.

---

## 6. Summary: grading the generalization strength of the four patterns

| Strength | Pattern | Evidence |
| --- | --- | --- |
| **Strong (replicates across scenarios)** | The C-axis architecture-level regularity | 3 candidates × 2 scenarios = 6 cells, all consistent; non-overlapping signs; responsible layer and commitment-layer presence identical per candidate |
| **Medium (point estimate replicates, significance lost)** | LTF's F① specificity | 0.583 → 0.560, Δ 0.023 ≪ CI half-width; consistent direction against the floor |
| **Weak (indeterminate, point estimate also fell)** | LTF's G-axis positive | +0.070 → +0.032; 10-seed mean falls to +0.000; CI 2.4× wider |
| **Adverse (floor overtakes primary in the new scenario)** | SimLingo's G-axis primary readout | Primary 0.600 significant, but floor 0.645 higher |

**A methodological conclusion**: **the representation side (G) is more scenario-dependent than the
mechanism side (C).** The C axis measures "at which layer information enters the network", which is a
routing property of the network and has little to do with the causal structure of the stimulus; the G
axis measures "whether some linear direction separates two classes of stimuli", which **depends
directly on what those two classes are**. This round's results match that prior: C replicates 6/6,
G replicates 0/3. **The four axes are not homogeneous — their sensitivity to a change of scenario
differs, and that had never been measured before.**

---

## 7. Limitations

1. **Sample size is this round's first constraint** (Tier-S scale; the work order explicitly does not
   chase power). LB/LBn/LBv = 103/48/31 against G1's 291/283/212 gives CIs 1.4–2.4× wider.
   **Every "indeterminate" on G and F① should be read as power first, replication second.**
2. **LBv's (the falsification floor's) matching quality is the weakest link** ($d_{long}$ SMD
   −0.328), and its pool is already exhausted by the full trainval set (31 of 37 candidates
   selected). Any adjudication resting on LBv — including the floor-overtaking in §4 — must be
   discounted accordingly.
3. **Brake lights are unannotated**, so LBv's "both arms have brake lights on" is an **unverified**
   assumption (see mining report §LB/A43).
4. **Only three candidates were measured** (the work order's first batch). DiffusionDriveV2,
   Alpamayo-R1 and AutoVLA were not, so on this scenario the C-axis architecture-level regularity
   rests on 1 pure-transformer sample (SimLingo) against 2 TransFuser-family samples — weaker than
   the 3 vs 3 available on the ghost-probe scenario.
5. **The I axis was not re-measured on this scenario**, following the work order ("the domain pairing
   is an independent asset; reuse the existing one"). Strictly, this round therefore tested the
   scenario transferability of three axes (G/F/C), not four.
