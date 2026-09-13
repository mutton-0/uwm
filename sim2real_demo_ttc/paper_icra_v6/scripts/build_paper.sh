#!/bin/bash
# 编译论文：把目录同步到 exx 上用 tectonic 编译，再把 main.pdf 取回本地。
# 用法：build_paper.sh <论文目录绝对路径> [预览 dpi，默认 70]
#   例：bash build_paper.sh /home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v6
# 只编译，不跑 make_tables / fig_*；tables 与 figures 用目录里现成的。
set -e
P=${1:?用法: build_paper.sh <论文目录> [dpi]}; N=$(basename "$P")
OUT=$P/build_preview; mkdir -p "$OUT"; rm -f "$OUT"/p-*.png
X=exx@10.141.211.75; RP=/data/ruolin/repe_vla/$N
rsync -a --delete --exclude main.pdf --exclude '*.zip' --exclude build --exclude build_preview \
      --exclude '*.npz' --exclude '*.json' -e "ssh -o BatchMode=yes" "$P"/ $X:"$RP"/
ssh -o BatchMode=yes $X "cd $RP && rm -rf build && mkdir build && \
  timeout 600 /data/ruolin/envs/tectonic/bin/tectonic -X compile --keep-intermediates --outdir build main.tex \
  > build.log 2>&1; grep -E 'Overfull|error|undefined' build.log | sort -u | head"
scp -q -o BatchMode=yes $X:"$RP"/build/main.pdf "$P"/main.pdf
pdftoppm -r "${2:-70}" -png "$P"/main.pdf "$OUT"/p
echo "pages $(ls "$OUT"/p-*.png | wc -l)  ->  $P/main.pdf"
