#!/bin/bash
# 因果迁移实验：左舵学方向 -> 分别注入左舵/右舵场景
#
# 判定逻辑（顺序不可颠倒）：
#   1) 左舵注入**必须先有效**。无效 => 方法未建立，右舵的阴性结果不可解读，实验到此为止。
#   2) 左舵有效后看右舵：
#        右舵也有效  => 因果不变（invariance）
#        右舵无效    => 因果偏移（causal shift）—— 这是本实验要找的东西
#
# 方向与层权重**一律只在左舵巡航样本上学**；--stim-side 只换被注入的场景。
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
N=${N:-60}          # 每格刺激场景数
NR=${NR:-30}        # 随机方向对照数（p 值分辨率 = 1/NR）
DEV=${DEV:-cuda:0}
MODELS=${MODELS:-"dd ltf ddv2"}

echo "=============================================================="
echo " 第 1 步：阳性对照 —— 左舵注入速度轴，必须有效才能继续"
echo "=============================================================="
for m in $MODELS; do
  $PY scripts/steer_repe.py --model $m --axis speed --mode add --stim-side LHD \
      --n $N --n-rand $NR --device $DEV 2>&1 | grep -avE "^  skip|^  [0-9]+/"
done

echo
echo "=============================================================="
echo " 第 2 步：同一根轴注入右舵场景"
echo "=============================================================="
for m in $MODELS; do
  $PY scripts/steer_repe.py --model $m --axis speed --mode add --stim-side RHD \
      --n $N --n-rand $NR --device $DEV 2>&1 | grep -avE "^  skip|^  [0-9]+/"
done

echo
echo "=============================================================="
echo " 第 3 步：I 轴（v_bright）同样的左右舵对照"
echo "=============================================================="
for m in $MODELS; do
  for sd in LHD RHD; do
    $PY scripts/steer_repe.py --model $m --axis bright --mode add --stim-side $sd \
        --n $N --n-rand $NR --device $DEV 2>&1 | grep -avE "^  skip|^  [0-9]+/"
  done
done

echo
echo "=============================================================="
echo " 第 4 步：分段算子（RepE 的第二个算子）复核第 1/2 步"
echo "=============================================================="
for m in $MODELS; do
  for sd in LHD RHD; do
    $PY scripts/steer_repe.py --model $m --axis speed --mode piecewise --stim-side $sd \
        --n $N --n-rand $NR --device $DEV 2>&1 | grep -avE "^  skip|^  [0-9]+/"
  done
done

echo
$PY remote_pkg/summarize_causal.py
