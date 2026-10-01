# camera-calibration

RealSense D435i 与机械臂末端的手眼标定流程。

维护人: @boyue(副 manager:待定)
状态: active

## 这是什么

标定相机坐标系到机械臂末端坐标系的变换矩阵。标定结果 `configs/extrinsics.yaml` 给 arm-setup 仓库里的抓取流程用。

## 环境

- Python 3.10,OpenCV 4.9
- GPU:不需要,CPU 即可
- RealSense SDK:pyrealsense2==2.55.1(只有采集图像时需要;离线计算标定结果不需要)
- 安装:
  ```bash
  pip install -r requirements.txt
  ```

## 硬件

| 项 | 要求 |
|---|---|
| 相机 | Intel RealSense D435i,固件 5.15 以上 |
| 安装位置 | 用支架固定在机械臂末端法兰上,镜头朝向夹爪方向。安装照片放在 `docs/img/mount.jpg`(示例仓库里没有) |
| 线材 | USB 3.0 Type-C 转 Type-A,**至少 2 米**,机械臂运动时不能被拉紧。必须插在电脑的 USB 3.0 口上,USB 2.0 口会掉帧 |
| 标定板 | 9×6 棋盘格,格子边长 25 mm,打印后贴在硬板上 |

挪过相机位置之后,`configs/extrinsics.yaml` 就失效了,必须重新标定。

## 怎么跑

先跑自检,确认环境没问题(几秒钟):
```bash
bash scripts/smoke_test.sh
```

两步**按顺序串行**执行,第二步读第一步存下的图像:
```bash
# 1. 采集标定板图像(需要接上相机)
python scripts/capture.py --output data/calib_0930/

# 2. 计算标定结果
python scripts/run_calib.py --input data/calib_0930/ --output configs/extrinsics.yaml
```

跑完会在 `configs/` 下生成 `extrinsics.yaml`,里面是一个 4×4 变换矩阵。

## 数据 / 权重

标定板图像不上传 git,示例数据在组内 NAS:`/data/sky-lab/camera-calib-samples/`。

## 本机专属路径

- `/data/sky-lab/...`:组内 NAS 的挂载点。没有挂载 NAS 的机器上,把示例数据拷到本地,`--input` 指向本地目录即可

## 已知问题

- D435i 在近距离(<0.3 m)畸变较大,标定板别放太近
- 目前只支持棋盘格标定板,ChArUco 还没接
- `scripts/` 里是占位脚本,只演示目录结构和 README 写法,没有真正的标定实现
