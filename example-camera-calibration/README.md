# camera-calibration

RealSense D435i 与机械臂末端的手眼标定流程。

维护人: @boyue
状态: active

## 这是什么

用于标定相机坐标系到机械臂末端坐标系的变换矩阵,标定结果给 arm-setup 仓库里的抓取流程用。

## 环境

- Python 3.10, OpenCV 4.9
- RealSense SDK (pyrealsense2==2.55.1)
- pip install -r requirements.txt

## 怎么跑

1. 采集标定板图像:
       python scripts/capture.py --output data/calib_0930/
2. 计算标定结果:
       python scripts/run_calib.py --input data/calib_0930/ --output configs/extrinsics.yaml
   跑完会在 configs/ 下生成 extrinsics.yaml,里面是 4x4 变换矩阵。

## 数据 / 权重

标定板图像不上传,示例数据在组内 NAS: /data/sky-lab/camera-calib-samples/

## 已知问题

- D435i 在近距离(<0.3m)畸变较大,标定板别放太近
- 目前只支持棋盘格标定板,ChArUco 还没接
