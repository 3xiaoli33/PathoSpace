#!/usr/bin/env python3
"""
Fully automatic MIL grid launcher.

This script runs Vim-T and Vim-S encoders across four MIL heads
(`abmil`, `transmil`, `clam_sb`, `dtfd`) using the BRACS CLAM features
that were generated with the Vim4Path DINO checkpoints. No CLI arguments
are needed: just execute the script and it will take care of everything.
"""

import datetime
import itertools
import pathlib
import subprocess

ARCH_LIST = ["vim-t", "vim-s"]
MODEL_LIST = ["abmil", "transmil", "clam_sb", "dtfd"]
IMAGE_SIZE = 224
SOURCE_LEVEL = 10
TARGET_LEVEL = 10
RESULTS_DIR = "results"


def resolve_clam_root(repo_root: pathlib.Path) -> pathlib.Path:
    default_bracs = repo_root.parent / "bracs_clam_data"
    if default_bracs.exists():
        return default_bracs.resolve()
    return (repo_root.parent / "clam_data").resolve()


def main():
    repo_root = pathlib.Path(__file__).resolve().parent
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    clam_root = resolve_clam_root(repo_root)

    base_command = [
        "python",
        "main_cam.py",
        "--image_size",
        str(IMAGE_SIZE),
        "--source_level",
        str(SOURCE_LEVEL),
        "--target_level",
        str(TARGET_LEVEL),
        "--drop_out",
        "--early_stopping",
        "--lr",
        "2e-4",
        "--k",
        "1",
        "--label_frac",
        "1",
        "--weighted_sample",
        "--bag_loss",
        "ce",
        "--inst_loss",
        "svm",
        "--task",
        "task_1_tumor_vs_normal",
        "--log_data",
        "--results_dir",
        RESULTS_DIR,
        "--clam_data_root",
        str(clam_root),
    ]

    combinations = list(itertools.product(ARCH_LIST, MODEL_LIST))
    print(f"Launching {len(combinations)} runs using clam_data at {clam_root}")
    subset_template = "{size}_{src}at{tgt}"

    for arch, model in combinations:
        subset_dir = clam_root / subset_template.format(size=IMAGE_SIZE, src=SOURCE_LEVEL, tgt=TARGET_LEVEL) / arch
        if not subset_dir.exists():
            print(f"[SKIP] Missing feature directory for arch={arch} at {subset_dir}")
            continue

        exp_code = f"{arch}-{model}-{timestamp}"
        cmd = base_command + ["--arch", arch, "--exp_code", exp_code, "--model_type", model]
        pretty = " ".join(cmd)
        print(f"\n[RUN] {pretty}")
        subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
