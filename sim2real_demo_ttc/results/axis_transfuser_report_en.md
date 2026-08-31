# Candidate Expansion: TransFuser — Determined to be **the Same Executable Object as LTF**, Not Listed Separately

> Work-order §1 requires first confirming whether TransFuser shares `transfuser_agent.yaml` with LTF.
> This file reports the outcome of that check; it is registered in
> [`amendments.md`](amendments.md) §CE/A28. This candidate produced **no independent axis readouts**,
> for the reasons below; its slot is covered by [`axis_ltf_report_en.md`](axis_ltf_report_en.md).

---

## Methods: what was checked

Three independent checks, all pointing to the same conclusion.

**(i) Repository configuration.** `navsim/planning/script/config/common/agent/transfuser_agent.yaml`
targets `navsim.agents.transfuser.transfuser_agent.TransfuserAgent` and hard-codes **`latent: True`**
in its config body. `latent: True` *is* **Latent TransFuser** — the real lidar branch replaced by a
learnable BEV latent. The repository provides **only this one TransFuser-family configuration**;
there is no `latent: False` variant.

**(ii) Official checkpoint list.** `ckpt/download_ckpts.sh` lists three official SimScale checkpoints:
DiffusionDrive, GTRS_Dense and **LTF**. For the TransFuser family there is only
`LTF/ltf_sim_navtest.ckpt`; **no lidar-equipped TransFuser weights are released**.

**(iii) Stimulus-set constraint (the decisive one).** The shared stimulus set of this line of work is
the nuScenes **monocular camera** G1 corpus (adapted through a 4:1 sky-removing crop) and contains
**no lidar**. Even with weights obtained elsewhere, a lidar-equipped TransFuser could not be run on
this stimulus set under its training protocol; feeding it zero lidar would substitute a different
model (this round provides a direct cautionary case on DiffusionDriveV2, §CE/A33).

---

## Results

**Table 1. Determination for the TransFuser candidate under this repository and this stimulus set.**

| Check | Fact | Conclusion |
| --- | --- | --- |
| Repository configuration | `transfuser_agent.yaml` hard-codes `latent: True` | that configuration **is** LTF |
| Official weights | only `ltf_sim_navtest.ckpt` in the TransFuser family | no separate TransFuser weights available |
| Stimulus set | nuScenes monocular camera, no lidar | a lidar TransFuser is **not executable** |
| Final status | — | **the same executable object as LTF; not listed separately** |

> **Instrument side.** The three checks are mutually independent (configuration, weights, input
> modality); any one of them alone would establish that TransFuser is not a candidate distinguishable
> from LTF under these conditions, and all three hold simultaneously, so the conclusion is unambiguous.
> **Specimen side.** The TransFuser family's slot in this matrix is occupied by LTF, whose readouts
> appear in the LTF report.

---

## Discussion

**1. This is not "skipping" but "the two are one executable object under these conditions".** The work
order listed TransFuser and LTF as two candidates on the basis of the observation that the README's
LTF entry links directly to `transfuser_agent.yaml`; the check confirms that suspicion — they share
one configuration and one set of weights. Running the same weights twice and writing two rows in the
matrix would **inflate the candidate count and manufacture an appearance of independence**.

**2. The incidental gain is worth more than an extra row.** LTF and DiffusionDrive share the identical
`TransfuserBackbone(latent=True)` encoder and differ only in the action head, so the pair constitutes
a **same-encoder, different-head controlled comparison**. Last round attributed DiffusionDrive's
injection unmeasurability to its diffusion action head; this comparison directly refutes that
attribution (§CE/A31 and LTF report, Discussion §3). **Had TransFuser been forced into a separate
row, this comparison would not have been identified.**

**3. What it would take to include a genuine lidar TransFuser.** Two things simultaneously: (i)
obtaining official lidar-variant weights (not in the SimScale release), and (ii) constructing a
nuScenes lidar BEV input for this stimulus set (implemented this round for DiffusionDriveV2, §CE/A29,
and technically reusable). Only with both would it be a candidate **distinguishable from LTF**. Not
done this round; recorded as such.

---

## Record of self-correction

1. This report is a **determination**, not a record of experimental failure: TransFuser did not "get
   stuck" at some step; it is indistinguishable from LTF under this repository and this stimulus set.
   It is issued in accordance with work-order §3 ("failed candidates must also get a report, not be
   silently skipped"), but its status should be read as **"merged into LTF"** rather than
   **"abandoned"**.
