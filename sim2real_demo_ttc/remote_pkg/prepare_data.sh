#!/bin/bash
# 重建因果迁移实验的前置数据。池子与配置都在 git 里，激活缓存不在（体量大），故本脚本重建它们。
# 幂等：已存在的产物直接跳过。
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
DEV=${DEV:-cuda:0}
MODELS=${MODELS:-"dd ltf ddv2"}
R=$(pwd)

echo "===== 0. 检查数据集根目录 ====="
for d in /data/dataset/navsim/dataset/sensor_blobs /data/dataset/navsim/dataset/navsim_logs; do
  [ -d "$d" ] && echo "  ✓ $d" || { echo "  ✗ 缺 $d —— 无法继续"; exit 1; }
done

echo "===== 1. work_c_dep/lead（遮挡图 + 事件表）====="
if [ -f work_c_dep/lead/mining/events_all.jsonl ]; then
  echo "  已存在，跳过"
else
  $PY scripts/pool_to_c_work.py --pool deploy_pool_lead_vp.json --corpus navsim \
      --work $R/work_c_dep/lead || exit 1
fi

echo "===== 2. vdecel 激活（速度轴 / v_decel 的原料）====="
for m in $MODELS; do
  f=results/vdecel_acts_navsim_test_${m}.npz
  if [ -f "$f" ]; then echo "  已存在 $f"; continue; fi
  echo "  --- $m ---"
  $PY scripts/f_vdecel_extract.py --model $m --pool cruise_pool_navsim_test.json \
      --corpus navsim --device $DEV || exit 1
done

echo "===== 3. vbright 激活（I 轴的原料）====="
for m in $MODELS; do
  f=results/vbright_acts_navsim_lead_${m}_night_global.npz
  if [ -f "$f" ]; then echo "  已存在 $f"; continue; fi
  echo "  --- $m ---"
  $PY scripts/i_bright_extract.py --model $m --work $R/work_c_dep/lead --corpus navsim \
      --scen lead --kind night --scope global \
      --nuscenes-root /data/dataset/navsim/dataset/sensor_blobs \
      --crop-center-row 560 --device $DEV || exit 1
done

echo
echo "===== 前置数据就绪，可跑 run_causal_transfer.sh ====="
ls -la results/vdecel_acts_navsim_test_*.npz results/vbright_acts_navsim_lead_*_night_global.npz 2>/dev/null | awk '{print "  "$5"  "$9}'
