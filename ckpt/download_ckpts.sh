#!/usr/bin/env bash
# 下载 SimScale 官方 ckpt（A 组）到本目录。详见 ckpt/README.md。
set -e
DEST="$(cd "$(dirname "$0")" && pwd)"     # 下到 ckpt/ 目录自身
USE_MODELSCOPE=${USE_MODELSCOPE:-0}        # 国内改 1 走 ModelScope

HF="https://huggingface.co/datasets/OpenDriveLab/SimScale/resolve/main/SimScale_ckpts"
# 相对路径 -> 落地文件名
FILES=(
  "DiffusionDrive/diffusiondrive_sim_navhard.ckpt"
  "GTRS_Dense/gtrs_dense_resnet_sim_expert_navhard.ckpt"
  "LTF/ltf_sim_navtest.ckpt"
)

for rel in "${FILES[@]}"; do
  out="$DEST/$(basename "$rel")"
  if [ -f "$out" ]; then echo "[skip] $out 已存在"; continue; fi
  echo "[get ] $rel"
  wget -c "$HF/$rel" -O "$out"
done
echo "完成，文件在 $DEST"
[ "$USE_MODELSCOPE" = "1" ] && echo "(提示: ModelScope 镜像见 README 表格 MS 链接)"
