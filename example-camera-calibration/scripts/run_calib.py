"""计算手眼标定结果。

用法:
    python scripts/run_calib.py --input data/calib_0930/ --output configs/extrinsics.yaml

同样是示例仓库的占位脚本,不是真实实现。
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="标定图像目录")
    parser.add_argument("--output", required=True, help="输出的外参 yaml 路径")
    args = parser.parse_args()

    # TODO: 读取 args.input 下的图像,跑 cv2.calibrateHandEye,
    # 把结果写成 4x4 变换矩阵存到 args.output
    print(f"[demo] 从 {args.input} 计算标定结果,写入 {args.output}")


if __name__ == "__main__":
    main()
