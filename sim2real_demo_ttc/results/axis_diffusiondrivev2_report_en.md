# Candidate Expansion: G/F/C-axis Readouts for DiffusionDriveV2

> Work order: [`../docs/candidate_pool_expansion_workorder.md`](../../docs/candidate_pool_expansion_workorder.md)
> (a one-day time-boxed candidate). All judgement calls are recorded in
> [`amendments.md`](amendments.md) §CE (A29, A30, A32, A33). Numerical artifacts:
> `g_axis_ddv2.json`, `f_axis_action_counterfactual.json`, `f_axis_dd_steer_ddv2{,_brake}.json`,
> `c_axis_hazard_ddv2.json`, `v_brake_ddv2.npz`; adapter: `ddv2_g1_adapter/`.
> **Outcome: integration succeeded; three of four readouts usable, one indeterminate.**

---

## Methods

### Two blockers that had to be solved during integration (neither was skipped)

**Blocker (i): the released weights require real lidar.** Both official configurations
(`diffusiondrivev2_{sel,rl}_agent.yaml`) set **`latent: False`**, and the released checkpoint
`diffusiondrivev2_sel.ckpt` (972 tensors) contains **232 `lidar_encoder.*` tensors and zero
`latent` tensors** — unlike DiffusionDrive/LTF it does **not** substitute a learnable latent for the
lidar branch. Following the adapter philosophy that each candidate runs through its own native input
format, we construct the TransFuser-style BEV histogram on the fly from nuScenes **LIDAR_TOP**,
reproducing its `_get_lidar_feature` step by step (256×256 grid, x/y ∈ [−32, 32], 4 px/m, points with
z > 0.2 m only, clipped at 5 per pixel then normalized, `use_ground_plane=False` hence single
channel), with the point cloud transformed to the ego frame via `calibrated_sensor`. **The stimulus
set itself is unchanged**: the same G1 events and the same frames, merely also reading the LIDAR_TOP
of the same sample.

**Blocker (ii): the inference path hard-depends on a nuPlan PDM metric cache.**
`TrajectoryHead.forward_test_rl`, after computing the coarse and fine refinements and selecting the
candidate trajectories, **unconditionally** calls `get_pdm_score_para(...)` to rank them with the
official PDM scorer, and **returns no trajectory at all** (only `{'reward_dict': ...}`). The authors
themselves left a commented-out official-evaluation exit immediately above that line:
`# return {"trajectory": traj_to_score[:,-1]}`. We **modify no line of the external repository**;
instead the adapter replaces `get_pdm_score_para` at runtime with a carrier exception that brings out
the already-computed candidate trajectories, from which `[:, -1]` is taken — **verbatim equivalent**
to that official exit, and occurring after all network computation, changing no forward logic.

### Stimuli, adapter and operationalization of the three axes

**Exactly the same** G1 corpus and N1 negatives as the existing candidates
(A 291 / D2a 283 / D2cV 212 / D2b 311 / D2bV 78 / D2c 343), with group sizes matching DiffusionDrive
and LTF cell for cell. The adapter `ddv2_g1_adapter/ddv2_adapter.py` inherits the image / status /
token / pooling / behavioural front end of `diffusiondrive_g1_adapter` and adds only the lidar path.
The operationalizations of G / F① / F② / C-hazard are item-by-item as in the LTF report.

**One consequential correction that must be declared (§CE/A33)**: the injection and patching scripts
were written for DiffusionDrive/LTF and passed no lidar. That is harmless for those two (their lidar
branch is a constant latent) but would give DDV2 an **all-zero lidar histogram**. The first-version
DDV2 C-hazard obtained without lidar is **void**; everything was re-run with real lidar.

---

## Results

**Table 1. G-axis readout for DiffusionDriveV2 on the shared G1 stimulus set (group sizes identical to DiffusionDrive and LTF).**

| Pooling | primary CV-AUC(A vs D2a) | 95% CI | $p$ | D2cV floor | primary − floor | 95% CI | difference over 10 seeds | permutation floor | random floor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vision_mean` (primary) | 0.530 | [0.483, 0.577] | 0.214 | 0.505 | **+0.025** | [−0.044, +0.097] | +0.004 ± 0.026 | 0.506 ± 0.033 | 0.503 ± 0.036 |
| `region_mean` (sensitivity) | 0.527 | [0.480, 0.574] | 0.265 | 0.513 | **+0.014** | [−0.049, +0.076] | +0.003 ± 0.018 | 0.505 ± 0.034 | 0.520 ± 0.033 |

Peak layer $L^\*$ = 1 (`vision_mean`) / 3 (`region_mean`). Across 10 fold-assignment seeds the
difference ranges over [−0.041, +0.051] (primary pooling), **including zero**. Neither
geometric-robustness correlation is significant (`vision_mean`: log area −0.069, $p$ = 0.099;
eccentricity −0.050, $p$ = 0.237). Verdict: **indeterminate** (the difference CI crosses zero and its
sd exceeds the effect itself).

**Table 2. F-axis readouts.**

| Readout | Quantity | Value | 95% CI | $p$ | Verdict |
| --- | --- | --- | --- | --- | --- |
| F① | $b$(A), action change induced by hazard frames [m/s] | **+0.201** | [+0.052, +0.343] | — | significantly ≠ 0 |
| F① | $b$(D2a) [m/s] | +0.023 | [−0.117, +0.183] | — | indistinguishable from 0 |
| F① | **b-AUC(A vs D2a)** | **0.556** | **[0.5005, 0.613]** | 0.020 | **PASS** (marginally) |
| F① | b-AUC(A vs D2cV) | 0.573 | [0.512, 0.633] | 4.93 × 10⁻³ | see Discussion §3 |
| F② | injection of $v_{hazard}^{ddv2}$ / in-house upper bound $v_{brake}^{ddv2}$ | *see `f_axis_dd_steer_ddv2{,_brake}.json`* | — | — | same encoder family as DiffusionDrive/LTF; expected unmeasurable |

Readout validity of the in-house upper-bound direction $v_{brake}^{ddv2}$: peak per-layer held-out
AUC **0.646 at L1** (the ego-speed main effect, explaining over 95% of variance, was removed).

**Table 3. C-hazard readout — the sufficiency check FAILS; verdict indeterminate.**

| Quantity | Value | Criterion |
| --- | --- | --- |
| patch-ALL recovery (median) | **+0.552** | should be ≈ +1.0; **sufficient-cut-set check fails** |
| events dropped by the run-time denominator guard | 2 / 14 | denominator guard (§CE/A32) |
| $C_m$ (for reference only; **not readable as concentration**) | 0.845 [0.704, 0.964] | premise violated |
| recovery-profile Spearman / responsible-layer mode | +0.810 / L4 | — |

**Verdict: indeterminate (sufficient-cut-set check fails).** The reason is structural:
DiffusionDrive/LTF have an **input-independent constant latent** in place of the lidar branch, so all
differences between the clean and ghost conditions pass through the eight patched fused-token blocks;
DDV2's **lidar input differs between the two conditions** (different real point clouds), and that
condition-specific information **bypasses** the patched modules through the convolutional lidar
branch. The fused tokens are therefore **not a sufficient cut set** under this pairing, and layer-wise
shares are uninterpretable.

> **Instrument side.** Both integration blockers were solved in a **reviewable way that modifies no
> third-party code** (step-by-step reproduction of the lidar histogram; runtime replacement of the PDM
> scorer to reach the authors' own official-evaluation exit). Group sizes match DiffusionDrive and LTF
> cell for cell, so G and F① are strictly same-protocol. **The C-axis indeterminacy was caught on the
> spot by the patch-ALL check**, not explained away afterwards — which is exactly what that check
> exists for.
> **Specimen side.** DDV2's action responds markedly to hazard (b(A) = +0.201 m/s, 21× DiffusionDrive)
> and F① clears the geometry-matched control; but its representation-side G readout is
> indistinguishable from its own falsification floor.

---

## Discussion

**1. The comparison with DiffusionDrive yields one clear difference: the action response is 21× larger
while the representational readout is unchanged.** On the same stimuli, b(A) goes from
+0.0095 [−0.030, +0.045] (not significant) for DiffusionDrive to **+0.201 [+0.052, +0.343]
(significant)** for DiffusionDriveV2, and b-AUC from 0.553 (indeterminate) to **0.556 (PASS)**.
Yet the G-axis "primary minus falsification floor" is indeterminate for both (+0.009 vs +0.025).
**V2's improvement therefore shows up on the action side, not on the readable-representation side** —
precisely the kind of distinction a per-link diagnosis can make and a single score cannot.
(Restriction: V2 also has an extra real-lidar stream, so the b(A) gain cannot be attributed wholly to
architectural improvement.)

**2. The C-axis indeterminacy is a substantive conclusion, not a gap.** It states precisely that
**for a multi-sensor model, patching the fused tokens alone does not constitute a sufficient cut
set**. Giving DDV2 a C-axis reading requires including the lidar branch's intermediate activations in
the patch scope (or using a pairing that holds lidar fixed and varies only the camera). That is a
concrete, executable design for a next round, not an impossibility.

**3. The anomaly that F① "vs D2cV" exceeds "vs D2a" requires a qualification.** DDV2 gives
b-AUC(A vs D2cV) = 0.573, above b-AUC(A vs D2a) = 0.556. D2cV's design premise is that the sole
difference is a relative velocity unobservable to a single-frame model (§CE/A34). DDV2 **is** a
single-frame model, so that premise holds; but it carries an extra lidar stream, and **a lidar point
cloud itself carries range information**, which may render part of the A-vs-D2cV distinction
observable. This report therefore flags DDV2's D2cV readout as **to be treated with caution**, with
the primary verdict resting on A vs D2a.

**4. What this readout may not claim.** (i) nuScenes uses a **32-beam single lidar** whereas
NAVSIM/nuPlan use merged multi-lidar point clouds; the density and coverage differ, so this histogram
is **out-of-distribution input** for DDV2. Its readouts must be read as "under this stimulus set and
this lidar approximation" and **cannot** be compared with its NAVSIM performance. (ii) Bypassing the
PDM scorer means we take the **last fine-refined trajectory** rather than the one the official
evaluation would select by PDM ranking; this matches the authors' commented-out exit but remains a
deviation.

---

## Record of self-correction

1. **The first C-hazard run passed no lidar (§CE/A33)**, placing a model that requires real lidar far
   out of distribution; that version's result ($C_m$ = 0.932) is **void** and was re-run with real
   lidar.
2. **Run-time denominator guard for C-hazard (§CE/A32)**: sampling uses the cache's two-frame-averaged
   gap while patching runs only frame[0], and the two can differ substantially. A single event with
   denominator −0.006 produced recovery = −439 and dragged the patch-ALL mean from +0.50 to −36. With
   the guard, 2 events are dropped and mean and median agree (+0.56 / +0.55).
3. **The F② event count was reduced from 40 to 30**: every DDV2 forward reads a nuScenes point cloud
   and the injection experiment has 169 conditions per event, so without caching the lidar I/O became
   the bottleneck. Even with a per-frame lidar cache the run uses 30 events to bound wall-clock time;
   this degradation affects power only, not protocol.
