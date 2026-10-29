"""使用 Ultralytics YOLO 训练校园道路病害检测模型。"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练校园道路病害 YOLO 检测模型")
    parser.add_argument("--data", default="datasets/campus_road/data.yaml", help="数据集 YAML 配置文件路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数，默认 100")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸，默认 640")
    parser.add_argument("--batch", type=int, default=16, help="batch size；显存不足时可调小，默认 16")
    parser.add_argument("--device", default="0", help="训练设备，例如 0、1、cpu；默认第 0 张 GPU")
    parser.add_argument("--project", default="runs/campus_road", help="训练输出根目录")
    parser.add_argument("--name", default="train", help="本次实验名称")
    parser.add_argument(
    "--patience",
    type=int,
    default=15,
    help="早停耐心值：验证指标连续多少轮没有改善后停止，默认 15",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"找不到数据配置文件：{data_path}\n请检查 --data 路径，并先填写 data.yaml。")
    if args.epochs <= 0 or args.imgsz <= 0 or args.batch <= 0:
        raise ValueError("--epochs、--imgsz 和 --batch 必须是正整数。")

    # 使用已安装的 Ultralytics，不复制或改写 YOLO 框架源码。
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        save=True,
        save_period=-1
    )


if __name__ == "__main__":
    main()
