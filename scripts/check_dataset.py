"""检查 YOLO 检测数据集，不修改任何原始图片或标签。"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover - 友好的环境提示
    raise SystemExit("缺少 PyYAML，请先运行：pip install -r requirements.txt") from exc


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IGNORED_LABEL_FILES = {"classes.txt"}
DEFAULT_CLASS_NAMES = ("crack", "manhole", "pothole", "speed_bump")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查校园道路病害 YOLO 数据集")
    parser.add_argument("--data", default="datasets/campus_road/data.yaml", help="数据集 YAML 配置文件路径")
    return parser.parse_args()


def resolve_dataset_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def read_yaml(data_path: Path) -> dict:
    try:
        with data_path.open("r", encoding="utf-8") as handle:
            content = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML 格式错误：{exc}") from exc
    if not isinstance(content, dict):
        raise ValueError("数据配置文件必须是 YAML 映射对象。")
    return content


def read_class_names(config: dict) -> tuple[str, ...]:
    """读取 data.yaml 中的类别名称，并按类别编号排列。"""
    names = config.get("names")
    if isinstance(names, list):
        return tuple(str(name) for name in names)
    if isinstance(names, dict):
        try:
            return tuple(str(names[index]) for index in range(4))
        except (KeyError, TypeError):
            pass
    print("[警告] data.yaml 的 names 不完整，暂使用默认四类名称。")
    return DEFAULT_CLASS_NAMES


def check_split(
    split_name: str,
    split_value: str,
    dataset_root: Path,
    counts: Counter[int],
) -> tuple[int, int]:
    image_dir = resolve_dataset_path(split_value, dataset_root)
    if not image_dir.is_dir():
        print(f"[错误] {split_name}: 图片目录不存在：{image_dir}")
        return 0, 1

    # 标准 YOLO 布局：images/<split> 与 labels/<split>。
    label_dir = dataset_root / "labels" / Path(split_value).name
    if not label_dir.is_dir():
        print(f"[错误] {split_name}: 推测的标签目录不存在：{label_dir}")
        print("       期望结构通常是 images/<split> 与 labels/<split>。")
        return 0, 1

    images = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    errors = 0
    for image_path in images:
        relative = image_path.relative_to(image_dir)
        label_path = label_dir / relative.with_suffix(".txt")
        if not label_path.is_file():
            print(f"[错误] {split_name}: 图片缺少标签：{image_path} -> {label_path}")
            errors += 1
            continue
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            print(f"[错误] {split_name}: 标签不是 UTF-8 文本：{label_path}")
            errors += 1
            continue
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                print(f"[错误] {label_path}:{line_number}: 应有 5 个字段，实际 {len(fields)} 个")
                errors += 1
                continue
            try:
                class_id = int(fields[0])
                coords = [float(value) for value in fields[1:]]
            except ValueError:
                print(f"[错误] {label_path}:{line_number}: 类别编号和坐标必须是数字")
                errors += 1
                continue
            if not 0 <= class_id <= 3:
                print(f"[错误] {label_path}:{line_number}: 类别编号 {class_id} 不在 0～3 范围内")
                errors += 1
            if any(not 0 <= value <= 1 for value in coords):
                print(f"[错误] {label_path}:{line_number}: 坐标必须全部位于 0～1 之间")
                errors += 1
            if 0 <= class_id <= 3:
                counts[class_id] += 1

    image_stems = {path.relative_to(image_dir).with_suffix("") for path in images}
    for label_path in sorted(label_dir.rglob("*.txt")):
        if label_path.name.lower() in IGNORED_LABEL_FILES:
            continue
        relative_stem = label_path.relative_to(label_dir).with_suffix("")
        if relative_stem not in image_stems:
            print(f"[错误] {split_name}: 标签找不到对应图片：{label_path}")
            errors += 1
    return len(images), errors


def main() -> None:
    args = parse_args()
    yaml_path = Path(args.data).resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"找不到数据配置文件：{yaml_path}")
    config = read_yaml(yaml_path)
    if "path" not in config or "train" not in config or "val" not in config:
        raise ValueError("data.yaml 必须包含 path、train 和 val 字段。")
    class_names = read_class_names(config)

    dataset_root = resolve_dataset_path(str(config["path"]), yaml_path.parent)
    print(f"检查数据集：{dataset_root}")
    print("类别顺序（来自 data.yaml）：")
    for class_id, class_name in enumerate(class_names):
        print(f"  {class_id}={class_name}")
    print("请确认这与原始标签的类别编号约定一致。")
    counts: Counter[int] = Counter()
    total_images = total_errors = 0
    for split in ("train", "val"):
        image_count, errors = check_split(split, str(config[split]), dataset_root, counts)
        total_images += image_count
        total_errors += errors
        print(f"{split}: 图片 {image_count} 张，问题 {errors} 个")

    print("\n类别实例统计：")
    for class_id, class_name in enumerate(class_names):
        print(f"  {class_id} ({class_name}): {counts[class_id]}")
    if total_errors:
        print(f"\n检查完成：发现 {total_errors} 个问题。未修改任何原始数据。")
        raise SystemExit(1)
    print(f"\n检查通过：共检查 {total_images} 张图片，未发现结构或标签格式问题。")


if __name__ == "__main__":
    main()
