🔬 PathoSpace: A Unified Framework for WSI Analysis with 2D-SSM

PathoSpace is an experimental repository for computational pathology (WSI/Patch level analysis). It integrates self-supervised feature extraction, Multiple Instance Learning (MIL) training, and interpretability analysis into a unified workflow.

The core of this project introduces a 2D-SSM (State Space Model) encoder that resolves spatial discrepancies in histopathology image analysis while maintaining linear computational complexity.

(Note: This repository aggregates and adapts code from upstream projects. Please refer to docs/Vim4Path.md and LICENSE files in subdirectories for details.)

📂 Directory Structure

🦕 dino/: Implementation of DINO pre-training combined with Vim/Vision-Mamba architecture.

✂️ preprocess/: WSI preprocessing pipeline (WSI $\to$ patches) and feature preparation, based on CLAM.

🧠 MIL/: Downstream Multiple Instance Learning frameworks for training and evaluation. Includes implementations of:

ABMIL, CLAM, DTFD-MIL, TransMIL.

🗺️ generate_camelyon16_heatmaps.py: Script for generating WSI-level heatmaps based on MIL attention/saliency scores.

📊 generate_analysis_plots.py: Tools for generating analysis plots, such as clustering visualizations (t-SNE) and long-range dependency histograms.

🛠️ Installation & Environment

This repository depends on CUDA 🎮, PyTorch 🔥, and OpenSlide 🖼️. Due to environmental variations, we recommend referring to docs/Vim4Path.md for detailed installation steps.

Basic setup:

git clone [https://github.com/](https://github.com/)[YourUsername]/PathoSpace.git
cd PathoSpace
pip install -r requirements.txt


⚡ Quickstart

1. 🔄 Pre-training (DINO + 2D-SSM)

Train the feature encoder using the DINO framework.
Note: Adjust nproc_per_node based on your GPU availability.

python -m torch.distributed.launch --nproc_per_node=4 dino/main.py \
  --data_path /path/to/patches \
  --output_dir checkpoints/exp \
  --arch vim-s \
  --disable_wandb


2. 🎯 Downstream MIL Training

Train the slide-level classifier using extracted features.

# Example: Training with TransMIL
python MIL/main_mil.py --model transmil --feats_dir /path/to/features --split_dir /path/to/splits


3. 🎨 Visualization & Analysis

Generate Heatmaps (CAMELYON16):
Visualize model attention on whole slide images.

python generate_camelyon16_heatmaps.py --wsi_dir /path/to/CAMELYON16_Test


Generate Analysis Plots:
Create t-SNE clustering and dependency distance plots.

python generate_analysis_plots.py --results_dir /path/to/results


📜 Upstream & References

This project builds upon the following excellent open-source works:

Vim4Path: See docs/Vim4Path.md

DINO: See dino/README.md

CLAM: For data preprocessing pipeline.

📝 Citation

If you use this code for your research, please cite our paper:

@article{li2025pathospace,
  title={PathoSpace: Efficient Self-Supervised 2D Visual State Space Models for Histopathology Image Analysis},
  author={Li, Xiangyu and Yang, Haojie and Zhang, Fengyun and Wu, Runzhi and Liu, Liangliang and Su, Ran},
  journal={Medical Image Analysis},
  year={2025}
}
