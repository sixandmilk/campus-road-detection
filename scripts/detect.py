"""使用训练好的 YOLO 权重对图片或文件夹执行目标检测。"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对图片或文件夹执行校园道路病害检测")
    parser.add_argument("--weights", required=True, help="训练权重路径，例如 runs/.../best.pt")
    parser.add_argument("--source", required=True, help="待检测图片或文件夹路径")
    parser.add_argument("--output", default="runs/detect", help="预测结果保存根目录")
    parser.add_argument("--name", default="predict", help="本次预测名称")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值，默认 0.25")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸，默认 640")
    parser.add_argument("--device", default=None, help="推理设备，例如 0 或 cpu；省略时自动选择")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)
    source_path = Path(args.source)
    if not weights_path.is_file():
        raise FileNotFoundError(f"找不到模型权重：{weights_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"找不到待检测图片或文件夹：{source_path}")
    if not 0 < args.conf <= 1:
        raise ValueError("--conf 必须大于 0 且不超过 1。")
    if args.imgsz <= 0:
        raise ValueError("--imgsz 必须是正整数。")

    model = YOLO(str(weights_path))
    predict_args = {
        "source": str(source_path),
        "conf": args.conf,
        "imgsz": args.imgsz,
        "project": args.output,
        "name": args.name,
        "save": True,
    }
    if args.device is not None:
        predict_args["device"] = args.device
    model.predict(**predict_args)


if __name__ == "__main__":
    main()
