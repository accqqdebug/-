# MSF-Paper-Transfer：GEN1 / COCO 统一 SNN 工程

这是 2026-10-21 从服务器新工程导出的源码包。包含统一模型、论文模块迁移、训练、验证、检查点评测和性能队列；文件总数见 `PACKAGE_MANIFEST.json`。不含数据集、模型权重、Python 环境、缓存、训练日志或嵌套压缩包。

## 目录与入口

- `msf-paper-port/yolox/`：MSF-YOLOX 框架和移植模块，保留原目录供导入。
- `performance.py`：GEN1/COCO 训练与验证、断点恢复。
- `evaluate_checkpoint.py`：单独评测 `best.pt` / `last.pt`。
- `smoke_port.py`：双数据集三种模型的 CPU 集成检查。
- `run_queue.py`：当前服务器性能队列示例，含该服务器绝对数据路径；迁移到其他机器请改数据路径与 Python 路径。
- `download_coco.py`：完整 COCO2017 下载示例，需改 ROOT 为自己的数据目录。
- `EXPERIMENT_CHECKLIST.zh-CN.md`：尚未启动的论文实验清单。
- `SOURCE_AND_ENV.json`：源码来源、导出时间、已验证服务器的依赖版本。

此包不依赖另一个本地源码副本；运行入口会从包内 `msf-paper-port` 导入模型。无需执行原 YOLOX 的 `setup.py` 或编译快速 COCO 扩展；当前评测使用 pycocotools。保留的 C++ 源文件供原框架可选模块使用。

## 安装与检查

推荐 Linux + NVIDIA CUDA，创建独立 Python 环境，先按机器 CUDA 安装匹配的 torch/torchvision，再安装其余依赖。`requirements-runtime.txt` 列出实际训练路径依赖；`SOURCE_AND_ENV.json` 记录服务器已验证版本。新版本 PyTorch 兼容性未全面验证，不应假定任意最新版均可直接训练。

```bash
python -m pip install -r requirements-runtime.txt
CUDA_VISIBLE_DEVICES='' python smoke_port.py
```

Windows 下可直接 `python smoke_port.py` 做 CPU 检查；本包服务器队列使用 `fcntl` 与 `nvidia-smi`，仅适用于 Linux。

## GEN1 性能训练

```bash
python performance.py --dataset gen1 --data /path/to/gen1_standard --variant baseline --out runs/gen1_baseline --epochs 20 --batch 2 --accumulate 4 --size 320 --steps 5 --seed 1
python performance.py --dataset gen1 --data /path/to/gen1_standard --variant paper_binary --out runs/gen1_paper_binary --epochs 20 --batch 2 --accumulate 4 --size 320 --steps 5 --seed 1
python performance.py --dataset gen1 --data /path/to/gen1_standard --variant paper --out runs/gen1_paper --epochs 20 --batch 2 --accumulate 4 --size 320 --steps 5 --seed 1
```

GEN1 目录需包含 `images/train.txt`、`images/val.txt`、`images/test.txt`，以及对应 `images/<split>/*.npy` 和 `labels/<split>/*.txt`。索引每行为相对 images 目录的路径；数组要求 `[20,240,304]`，前后各 10 bins 分别为两极性；标签为 YOLO 格式 `class x_center y_center width height`，坐标归一化。当前 loader 不直接读取原始 DAT/H5，也不把 50ms 十 bin 数据解释为连续原始事件流。

## COCO 性能训练

```bash
python performance.py --dataset coco --data /path/to/coco2017 --variant baseline --out runs/coco_baseline --epochs 20 --batch 2 --accumulate 4 --size 320 --steps 5 --seed 1
```

COCO 目录需含 `train2017/`、`val2017/`、`annotations/instances_train2017.json` 和 `annotations/instances_val2017.json`；80 类，静态图像重复 T 次。将 `--variant` 改为 `paper_binary` 或 `paper` 可评测相应迁移模型。

## 独立评测与续跑

```bash
python evaluate_checkpoint.py --checkpoint runs/gen1_paper/best.pt --data /path/to/gen1_standard --split test --out evaluation/gen1_paper
```

重复执行原训练命令，并使用相同 `--out` 与相同超参数，会读取该目录 `last.pt` 继续训练。不要把新的实验指向另一个实验已有的输出目录。仅加载可信检查点。

## 模型与实验口径

- `baseline`：MSF 多阈值放电，固定衰减。
- `paper_binary`：二值 PLIF、深层 MaxSpikeFormer、FCBR；GEN1 加 KCL。
- `paper`：保留多阈值 MSF 放电并学习衰减，再加入深层 MaxSpikeFormer、FCBR；GEN1 加 KCL。它是兼容移植版，与严格二值版不同。

原框架已有时序 PANet 和解耦头，属于继承结构。FCBR/KCL 仅训练使用；新增 MaxSpikeFormer 本身仍有推理开销。基础检测框损失遵循 YOLOX，因此不是原论文网络与训练配方的逐层等价复制。

当前 20 epoch 是性能筛查：320 输入、无预训练、无增强。它不是官方充分训练的 SOTA 配方。GEN1 的指标为已有标准标签上的 COCOeval，尚未对齐官方 Prophesee 过滤协议。CPU 检查通过、数据下载完成或预检 loss 均不能当作检测性能提升证据。

## 来源

MSF 官方仓库：https://github.com/fanliangwei/Multisynaptic-spiking-neurons ，论文 DOI：10.1038/s41467-025-62251-6。框架继承 YOLOX，许可证见 `LICENSE`。作者公开可读的检测配置主要为 COCO；本包 GEN1 adapter 和论文模块迁移为本项目新增。更高分 SpikeDet 的完整对应公开代码尚未核实，因此不将本包称为“2026 最新 SOTA 已复现”。
