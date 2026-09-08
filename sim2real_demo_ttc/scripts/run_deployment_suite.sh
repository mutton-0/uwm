#!/bin/bash
# 传感器下载完成后的完整重跑链：池 -> F-3 -> GT轴 -> C轴 -> G-VS -> 部署指标
# 主口径 = 限城池 (vp)：deployment 只含维加斯+匹兹堡，与 nuScenes(波士顿+新加坡) 城市零重叠。
# 同时保留合并池 (all) 作对照，量化"限城"本身的代价。
set -u
R=/data/ruolin/uwm/sim2real_demo_ttc
S=/tmp/claude-1001/-data-ruolin-uwm/007773d2-b773-4b39-be8d-122fceddf6fe/scratchpad
PY=/home/mut0/.conda/envs/simscale/bin/python
PYSL=/data/ruolin/envs/simlingo/bin/python
NS=/data/dataset/navsim/dataset/sensor_blobs
L=$S/suite.log
cd $R

say(){ echo "[$(date +%H:%M:%S)] $*" >> $L; }

say "===== 1. 重建 deployment 池 ====="
for tag in vp all; do
  if [ "$tag" = vp ]; then GP="pool_ghost_test_vp.json:test pool_ghost_trainval_vp.json:trainval"
                           LP="pool_lead_test_vp.json:test pool_lead_trainval_vp.json:trainval"
  else GP="brake_first_pool_navsim_fx_final.json:test brake_first_pool_ghost_trainval.json:trainval"
       LP="brake_first_pool_lead_navsim_fx_final.json:test brake_first_pool_lead_trainval.json:trainval"; fi
  $PY scripts/build_deploy_pool.py --scenario ghost --tag $tag --pools $GP >> $L 2>&1
  $PY scripts/build_deploy_pool.py --scenario lead  --tag $tag --pools $LP >> $L 2>&1
done

say "===== 2. F-3 三臂（主口径 vp）====="
for sc in ghost lead; do
  for m in dd ltf ddv2; do
    say "F3 $sc $m"
    $PY scripts/f3_brakefirst.py --model $m --corpus navsim --split auto \
      --pool $R/results/deploy_pool_${sc}_vp.json \
      --out results/f3_dep_${sc}_${m}_traj.json >> $L 2>&1
  done
  say "F3 $sc simlingo"
  $PYSL scripts/f3_brakefirst.py --model simlingo --corpus navsim --split auto \
    --pool $R/results/deploy_pool_${sc}_vp.json \
    --out results/f3_dep_${sc}_simlingo_traj.json >> $L 2>&1
done

say "===== 3. GT 轴 ====="
for sc in ghost lead; do
  for m in simlingo dd ltf ddv2; do
    [ -f results/f3_dep_${sc}_${m}_traj.json ] || continue
    $PYSL scripts/f3_gt_axis_full.py --model $m --corpus navsim --split auto \
      --f3 results/f3_dep_${sc}_${m}_traj.json \
      --out results/f3_gt_dep_${sc}_${m}.json >> $L 2>&1
  done
done

say "===== 4. 及时性 + 危险场景避障 ====="
for sc in ghost lead; do
  $PY scripts/brake_timeliness.py --scenario $sc --pool deploy_pool_${sc}_vp.json \
    --f3-prefix f3_dep_${sc} --out results/timeliness_dep_${sc}.json >> $L 2>&1
  $PY scripts/hazard_avoidance.py --scenario $sc --f3-prefix f3_dep_${sc} \
    --out results/hazard_avoidance_dep_${sc}.json >> $L 2>&1
done

say "===== 5. C 轴 ====="
for sc in ghost lead; do
  $PY scripts/pool_to_c_work.py --pool deploy_pool_${sc}_vp.json --corpus navsim \
    --work $R/work_c_dep/$sc >> $L 2>&1
  for m in dd ltf ddv2; do
    $PY scripts/c_axis_hazard_patch.py --model $m --work $R/work_c_dep/$sc --all-events \
      --readout arc_full --corpus navsim --nuscenes-root $NS --crop-center-row 560 \
      --out $R/results/c_dep_${sc}_${m}.json >> $L 2>&1
  done
  $PYSL scripts/c_axis_hazard_patch.py --model simlingo --work $R/work_c_dep/$sc --all-events \
    --readout arc_full --corpus navsim --nuscenes-root $NS \
    --out $R/results/c_dep_${sc}_simlingo.json >> $L 2>&1
done

say "===== 6. G-VS ====="
for sc in ghost lead; do
  $PY scripts/pool_to_gvs_work.py --pool deploy_pool_${sc}_vp.json --corpus navsim \
    --work $R/work_gvs_dep/$sc --window 2 --stride 2 >> $L 2>&1
  $PYSL scripts/gvs1_sam_pseudo_gt.py --work $R/work_gvs_dep/$sc \
    --nuscenes-root $NS --device cuda:1 >> $L 2>&1
  for m in dd ltf ddv2; do
    $PY scripts/gvs3_extract_tokens.py --model $m --work $R/work_gvs_dep/$sc --corpus navsim \
      --nuscenes-root $NS --crop-center-row 560 \
      --out $R/work_gvs_dep/$sc/feats_${m}.npz >> $L 2>&1
    $PY scripts/gvs2_probe.py --feats $R/work_gvs_dep/$sc/feats_${m}.npz \
      --label dep_${sc}_${m} --out $R/results/gvs_dep_${sc}_${m}.json >> $L 2>&1
  done
  $PYSL scripts/gvs3_extract_tokens.py --model simlingo --work $R/work_gvs_dep/$sc \
    --corpus navsim --nuscenes-root $NS \
    --out $R/work_gvs_dep/$sc/feats_simlingo.npz >> $L 2>&1
  $PY scripts/gvs2_probe.py --feats $R/work_gvs_dep/$sc/feats_simlingo.npz \
    --label dep_${sc}_simlingo --out $R/results/gvs_dep_${sc}_simlingo.json >> $L 2>&1
done

say "===== SUITE ALLDONE ====="
