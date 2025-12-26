#!/usr/bin/env python3
"""
CAMELYON16 高级分析生成器 (Cluster & Distance) [最终修复版]

功能：
1. 生成 t-SNE 聚类图：展示 Tumor (High Attn) 和 Normal (Low Attn) 的特征可分性。
2. 生成 Attention Distance 直方图：展示长程依赖特性。

修复点：
- [Critical] 修复 TransMIL 报错：完全延续热力图代码的逻辑，当模型不返回权重时，自动降级为梯度显著性计算 (Gradient Saliency)。
- [Robustness] 强制过滤掉 2d-ssm 和 resnet，只跑 4 个核心 Backbone。
"""

import sys
import argparse
import traceback
from pathlib import Path
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist
from tqdm import tqdm

# ================= 路径配置 =================
# Repo root = the folder containing this script (i.e. `code/` in the original workspace layout)
REPO_ROOT = Path(__file__).resolve().parent
MIL_DIR = REPO_ROOT / "MIL"

for path in (REPO_ROOT, MIL_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    from MIL.models.abmil import DAttention
    from MIL.models.model_clam import CLAM_SB
    from MIL.models.dtfd_mil import DTFD_MIL
    from MIL.models.transmil import TransMIL
except ImportError as exc:
    print(f"[Critical] 导入失败: {exc}")
    sys.exit(1)

# ================= 1. 模型定义与配置 =================

# 原始配置
RAW_MODEL_SPECS = {
    "vim_s": {"feature_dim": 384, "path": "vim-s"},
    "vim_t": {"feature_dim": 192, "path": "vim-t"},
    "vit_s": {"feature_dim": 1536, "path": "vit-s"},
    "vit_t": {"feature_dim": 768, "path": "vit-t"},
    # 以下模型将被过滤
    "vim_s_2d": {"feature_dim": 384},
    "vim_t_2d": {"feature_dim": 192},
    "resnet50": {"feature_dim": 1536},
}

# 过滤：剔除 2d 和 resnet
TARGET_MODELS = {k: v for k, v in RAW_MODEL_SPECS.items() if "2d" not in k and "resnet" not in k}
METHODS = ["abmil", "clam_sb", "dtfd", "transmil"]

# 权重路径
CKPT_BASE = REPO_ROOT / "MIL" / "C-results"
CKPT_MAP = {
    "vim_s": {
        "abmil": CKPT_BASE / "vim-s-224-10at10-abmil-1d_s1" / "s_0_checkpoint.pt",
        "clam_sb": CKPT_BASE / "vim-s-224-10at10-clam_sb-1d_s1" / "s_0_checkpoint.pt",
        "dtfd": CKPT_BASE / "vim-s-224-10at10-dtfd-1d_s1" / "s_0_checkpoint.pt",
        "transmil": CKPT_BASE / "vim-s-224-10at10-transmil-1d_s1" / "s_0_checkpoint.pt",
    },
    "vim_t": {
        "abmil": CKPT_BASE / "vim-t-abmil-20251115_134349_s1" / "s_0_checkpoint.pt",
        "clam_sb": CKPT_BASE / "vim-t-clam_sb-20251115_134349_s1" / "s_0_checkpoint.pt",
        "dtfd": CKPT_BASE / "vim-t-dtfd-20251115_134349_s1" / "s_0_checkpoint.pt",
        "transmil": CKPT_BASE / "vim-t-transmil-20251115_134349_s1" / "s_0_checkpoint.pt",
    },
    "vit_s": {
        "abmil": CKPT_BASE / "vit-s-abmil-20251115_134349_s1" / "s_0_checkpoint.pt",
        "clam_sb": CKPT_BASE / "vit-s-clam_sb-20251115_134349_s1" / "s_0_checkpoint.pt",
        "dtfd": CKPT_BASE / "vit-s-dtfd-20251115_134349_s1" / "s_0_checkpoint.pt",
        "transmil": CKPT_BASE / "vit-s-transmil-20251115_134349_s1" / "s_0_checkpoint.pt",
    },
    "vit_t": {
        "abmil": CKPT_BASE / "vit-t-abmil-20251115_134349_s1" / "s_0_checkpoint.pt",
        "clam_sb": CKPT_BASE / "vit-t-clam_sb-20251115_134349_s1" / "s_0_checkpoint.pt",
        "dtfd": CKPT_BASE / "vit-t-dtfd-20251115_134349_s1" / "s_0_checkpoint.pt",
        "transmil": CKPT_BASE / "vit-t-transmil-20251115_134349_s1" / "s_0_checkpoint.pt",
    },
}

# ================= 2. 核心函数 =================

def get_model(backbone, method, dim, device):
    """加载模型并载入权重"""
    if method == "abmil": model = DAttention(in_dim=dim, n_classes=2, dropout=True, act="relu")
    elif method == "clam_sb": model = CLAM_SB(gate=True, size_arg="small", dropout=True, k_sample=8, n_classes=2, embedding_dim=dim)
    elif method == "dtfd": model = DTFD_MIL(in_dim=dim, n_classes=2, dropout=True)
    elif method == "transmil": model = TransMIL(in_dim=dim, n_classes=2, dropout=True, act="relu")
    else: return None

    ckpt_path = CKPT_MAP.get(backbone, {}).get(method)
    if ckpt_path and ckpt_path.exists():
        try:
            state = torch.load(ckpt_path, map_location='cpu')
            clean_state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(clean_state, strict=False)
            model.to(device)
            model.eval()
            return model
        except Exception as e:
            print(f"  [Error] 加载权重失败 {ckpt_path}: {e}")
            return None
    else:
        print(f"  [Warning] 缺失权重: {backbone} - {method}")
        return None

def infer_scores(model, method, features, device):
    """
    [关键修复] 推理获取 Attention Scores
    逻辑完全复刻热力图生成代码：
    1. DTFD: 特殊处理，no_grad
    2. 其他(ABMIL, CLAM): 尝试直接获取 Attention
    3. TransMIL: 如果 Attention 为 None，开启梯度计算 Saliency
    """
    # 确保特征是干净的叶子节点
    features = features.clone().detach().to(device)

    if method == "dtfd":
        # DTFD 不需要梯度
        with torch.no_grad():
            h = model.reducer(features)
            attn = model.attn_head.attention(h)
            scores = attn.detach().cpu().numpy().reshape(-1)
    else:
        # 其他模型：必须开启 enable_grad，为 TransMIL 的回传做准备
        with torch.enable_grad():
            features.requires_grad_(True)
            
            # 前向传播
            outs = model(features)
            
            # 解包：logits, probs, Y_hat, A (可能没有), results_dict
            logits = outs[0]
            y_hat = outs[2]
            
            attn = None
            if len(outs) > 3:
                attn = outs[3]
            
            if attn is not None:
                # Case 1: 模型直接返回了 Attention (ABMIL, CLAM)
                scores = attn.detach().cpu().numpy().reshape(-1)
            else:
                # Case 2: 模型没返回 Attention (TransMIL), 计算 Gradient Saliency
                # 1. 获取预测类别的 logit
                pred_label = int(y_hat.item())
                target_logit = logits.squeeze(dim=0)[pred_label]
                
                # 2. 反向传播
                model.zero_grad()
                target_logit.backward()
                
                # 3. 计算输入梯度的 L2 范数
                grads = features.grad.detach().cpu().numpy()
                scores = np.linalg.norm(grads, axis=1)

    # 归一化 (0-1)
    min_v, max_v = scores.min(), scores.max()
    scores = (scores - min_v) / (max_v - min_v + 1e-8)
    
    return scores

def plot_tsne(features, scores, save_path, backbone, method):
    """绘制 t-SNE 聚类图 (Top 500 vs Bottom 500)"""
    n_samples = 500
    if len(scores) < n_samples * 2:
        indices = np.arange(len(scores))
    else:
        sorted_idx = np.argsort(scores)
        idx_low = sorted_idx[:n_samples]        # Bottom (Normal)
        idx_high = sorted_idx[-n_samples:]      # Top (Tumor)
        indices = np.concatenate([idx_low, idx_high])
    
    subset_feats = features[indices]
    subset_scores = scores[indices]
    
    # 运行 t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    embedding = tsne.fit_transform(subset_feats)
    
    # 绘图
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=subset_scores, cmap='jet', alpha=0.7, s=20)
    plt.colorbar(scatter, label='Attention Score')
    plt.title(f"Feature Clustering (t-SNE)")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()

def plot_attention_distance(coords, scores, save_path, backbone, method):
    """绘制 Attention Distance 直方图 (长程依赖分析)"""
    # 筛选 Top 500 个高关注度 Patch
    top_k = 500
    if len(scores) < top_k:
        top_idx = np.arange(len(scores))
    else:
        top_idx = np.argsort(scores)[-top_k:]
    
    top_coords = coords[top_idx]
    
    # 计算成对欧几里得距离
    dists = pdist(top_coords, metric='euclidean')
    
    plt.figure(figsize=(8, 6))
    sns.histplot(dists, bins=50, kde=True, color='skyblue', edgecolor='black')
    plt.title(f"Attention Distance Distribution (Long-range)")
    plt.xlabel("Physical Distance (pixels)")
    plt.ylabel("Frequency (Pairwise count)")
    
    mean_dist = np.mean(dists)
    plt.axvline(mean_dist, color='red', linestyle='--', label=f'Mean: {mean_dist:.0f} px')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()

# ================= 3. 主流程 =================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide_id", type=str, default="tumor_104", help="Slide name without extension")
    parser.add_argument("--output_dir", type=str, default="cam16_analysis_plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.slide_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Analyzing Slide: {args.slide_id}")
    print(f"Models to run: {list(TARGET_MODELS.keys())}")

    for backbone, spec in TARGET_MODELS.items():
        # 加载 H5 特征
        h5_path = REPO_ROOT / "clam_data" / "224_10at10" / spec['path'] / "training" / "h5_files" / f"{args.slide_id}.h5"
        
        if not h5_path.exists():
            print(f"[Skip] Feature file not found: {h5_path}")
            continue
            
        try:
            with h5py.File(h5_path, 'r') as f:
                features = f['features'][:]
                coords = f['coords'][:]
            features = torch.from_numpy(features).float()
        except Exception as e:
            print(f"[Error] Failed to load {h5_path}: {e}")
            continue
            
        print(f"\n>>> Processing Backbone: {backbone}")
        
        for method in METHODS:
            print(f"  -> Method: {method}")
            
            # 加载模型
            model = get_model(backbone, method, spec['feature_dim'], device)
            if model is None: continue
            
            # 推理分数 (包含 TransMIL 的 Saliency 计算)
            try:
                scores = infer_scores(model, method, features, device)
            except Exception as e:
                print(f"     [Error] Inference failed: {e}")
                traceback.print_exc()
                continue
            
            # 绘图
            save_base = output_dir / f"{backbone}_{method}"
            
            # 1. t-SNE
            plot_tsne(features.numpy(), scores, str(save_base) + "_tsne.png", backbone, method)
            
            # 2. Distance
            plot_attention_distance(coords, scores, str(save_base) + "_distance.png", backbone, method)
            
            print(f"     [Saved] Plots generated for {backbone} + {method}")

    print("\nAll analysis plots generated successfully!")

if __name__ == "__main__":
    main()
