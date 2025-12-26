# PathoSpace

PathoSpace 是一个面向计算病理（WSI / Patch）的实验代码仓库，聚合了特征提取、MIL 训练、可解释性热力图与分析脚本。

本仓库代码包含/改写自多个上游项目（见 `docs/Vim4Path.md` 与各子目录的 `LICENSE*` 文件），主要目录如下：

- `dino/`: DINO 预训练与 Vim/Vision-Mamba 相关实现
- `preprocess/`: WSI -> patch 的预处理与特征准备（基于 CLAM pipeline）
- `MIL/`: 多实例学习（MIL）训练/评估/热力图（含 ABMIL/CLAM/DTFD/TransMIL 等实现）
- `generate_camelyon16_heatmaps.py`: 生成 WSI 级热力图（基于 MIL attention/显著性）
- `generate_analysis_plots.py`: 生成聚类与 long-range distance 等分析图

## Quickstart

> 该仓库依赖 CUDA / PyTorch / OpenSlide 等组件，环境差异较大；建议直接参考 `docs/Vim4Path.md` 的安装与数据处理流程，再按你的本地路径传参运行脚本。

示例：

```bash
# 预训练（示例）
python -m torch.distributed.launch --nproc_per_node=4 dino/main.py \
  --data_path /path/to/patches --output_dir checkpoints/exp --arch vim-s --disable_wand

# 生成 CAMELYON16 测试集热力图（示例）
python generate_camelyon16_heatmaps.py --wsi_dir /path/to/CAMELYON16_Test
```

## Upstream

- Vim4Path: `docs/Vim4Path.md`
- DINO: `dino/README.md`

