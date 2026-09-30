"""采集标定板图像。

用法:
    python scripts/capture.py --output data/calib_0930/

这里只是示例仓库的占位脚本,展示 README 里的命令要能对上实际代码,
不是真实的标定实现。
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="图像输出目录")
    parser.add_argument("--num-frames", type=int, default=30)
    args = parser.parse_args()

    # TODO: 接入 pyrealsense2,循环采集 args.num_frames 张棋盘格图像
    # 保存到 args.output 下
    print(f"[demo] 将采集图像保存到 {args.output}")


if __name__ == "__main__":
    main()
