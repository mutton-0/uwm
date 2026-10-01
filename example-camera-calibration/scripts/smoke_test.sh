#!/usr/bin/env bash
# 自检:检查环境能用、关键路径存在、入口脚本能启动。几秒钟跑完,不需要接相机。
# 用法:bash scripts/smoke_test.sh
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
fail=0

echo "== Python: $($PY --version 2>&1)"

for mod in cv2 numpy yaml; do
  if $PY -c "import $mod" 2>/dev/null; then
    echo "ok   import $mod"
  else
    echo "FAIL import $mod    → pip install -r requirements.txt"
    fail=1
  fi
done

# 相机 SDK 只有采集时需要,没有不算失败
if $PY -c "import pyrealsense2" 2>/dev/null; then
  echo "ok   import pyrealsense2"
else
  echo "skip import pyrealsense2(没装,只能离线计算标定结果,不能采集)"
fi

for d in configs scripts data; do
  if [ -d "$d" ]; then echo "ok   目录 $d/"; else echo "FAIL 缺目录 $d/"; fail=1; fi
done

for s in scripts/capture.py scripts/run_calib.py; do
  if $PY "$s" --help >/dev/null 2>&1; then
    echo "ok   $s --help"
  else
    echo "FAIL $s 启动不了"
    fail=1
  fi
done

if [ "$fail" -eq 0 ]; then echo "== 自检通过"; else echo "== 自检失败,看上面的 FAIL"; fi
exit $fail
