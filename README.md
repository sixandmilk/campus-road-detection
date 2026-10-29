# 校园道路病害检测（YOLO）

这是一个面向初学者的 Ultralytics YOLO 检测项目，用于检测四类校园道路病害：`crack`、`manhole`、`pothole`、`speed_bump`。

项目只调用已经安装的 `ultralytics`，不复制或重写 YOLO 框架源码。当前目录中的 `datasets/coco8`、`runs`、`weights` 和已有模型文件不会被移动、删除或修改。

## 项目结构

```text
datasets/campus_road/data.yaml  # 数据集配置模板
scripts/train.py                # 训练
scripts/detect.py               # 图片/文件夹推理
scripts/check_dataset.py        # 数据集检查与类别统计
requirements.txt                # Python 依赖（不安装 PyTorch）
.gitignore                      # 忽略数据、权重和训练输出
README.md                       # 使用说明
```

## 1. 激活环境

在 Anaconda Prompt 或 PowerShell 中执行：

```powershell
conda activate campus-yolo
python --version
python -c "import torch, ultralytics; print(torch.__version__); print(ultralytics.__version__); print(torch.cuda.is_available())"
```

`requirements.txt` 不包含 `torch` 或 `torchvision`，因此不会覆盖已经配置好的 CUDA 版 PyTorch。如需补充项目依赖，可执行：

```powershell
pip install -r requirements.txt
```

## 2. 准备数据集

建议使用标准 YOLO 检测目录：

```text
你的数据集目录/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

每张图片对应一个同名 `.txt` 标签。每行格式为：

```text
class_id center_x center_y width height
```

坐标应为 0～1 之间的归一化值。

编辑 [`datasets/campus_road/data.yaml`](datasets/campus_road/data.yaml)，至少填写：

```yaml
path: D:/你的数据集目录
train: images/train
val: images/val
```

注意：`names` 当前明确按 `0=crack`、`1=manhole`、`2=pothole`、`3=speed_bump` 排列。**开始训练前必须确认原始标签的类别编号与此顺序一致**；如果不一致，请先整理标签映射或修改配置，不能仅凭类别名称猜测。

## 3. 检查数据集

脚本不会自动修复或修改原始数据：

```powershell
python scripts/check_dataset.py --data datasets/campus_road/data.yaml
```

脚本会检查图片与标签是否一一对应、标签是否为 5 个字段、坐标是否在 0～1、类别编号是否在 0～3，并统计四类实例数量。发现问题时会输出文件路径和行号。

## 4. 开始训练

先用一个较短的试运行确认配置：

```powershell
python scripts/train.py `
  --data datasets/campus_road/data.yaml `
  --epochs 10 `
  --imgsz 640 `
  --batch 16 `
  --device 0
```

正式训练可以增加 `--epochs`。显存不足时降低 `--batch`，例如 `--batch 8` 或 `--batch 4`。脚本使用 `YOLO("yolo11n.pt")` 作为初始模型，训练输出默认写入 `runs/campus_road/train`。

## 5. 推理

对单张图片：

```powershell
python scripts/detect.py `
  --weights runs/campus_road/train/weights/best.pt `
  --source path/to/test.jpg `
  --output runs/detect `
  --name campus_demo `
  --conf 0.25 `
  --device 0
```

对整个文件夹时，将 `--source` 改为文件夹路径。带框结果会保存在 `runs/detect/campus_demo`。

## 6. GitHub 提交流程

当前工作区尚未初始化 Git。确认 `data.yaml` 中没有敏感信息后，可执行：

```powershell
git init
git add README.md requirements.txt .gitignore datasets/campus_road/data.yaml scripts
git commit -m "搭建校园道路病害 YOLO 检测项目"
git branch -M main
git remote add origin https://github.com/你的用户名/你的仓库.git
git push -u origin main
```

`.gitignore` 会排除数据集、训练输出和模型权重，避免将大型文件上传到 GitHub。请在 GitHub 或其他数据存储中单独管理图片、标签和 `.pt` 权重。
