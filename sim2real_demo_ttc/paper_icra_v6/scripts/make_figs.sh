#!/bin/bash
# 重出所有图，然后同步到 v6 并编译。用法：make_figs.sh [论文目录，默认 paper_icra_v6]
# 各图脚本读的都是 paper_icra_v5/ 下的数据 json，图先落在 v5/figures/，再拷到目标目录。
set -e
PY=/home/boyuewang/micromamba/envs/navsim/bin/python
V5=/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5   # 图脚本读的数据 json 仍在这里
DST=${1:-/home/boyuewang/120/uwm/sim2real_demo_ttc/paper_icra_v6}
for f in fig_frame.py:wide fig_frame.py:col fig_fi_dash.py: fig_teaser.py: fig_reach.py: \
         fig_cases.py: fig_sample.py:; do
  s=${f%%:*}; a=${f#*:}
  [ -f "$V5/scripts/$s" ] || { echo "跳过 $s（不存在）"; continue; }
  echo "== $s $a"; $PY "$V5/scripts/$s" $a
done
cp -f $V5/figures/*.pdf $V5/figures/*.png "$DST"/figures/ 2>/dev/null || true
bash /home/boyuewang/120/uwm/sim2real_demo_ttc/paper_icra_v6/scripts/build_paper.sh "$DST"
