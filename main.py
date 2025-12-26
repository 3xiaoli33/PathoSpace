import argparse
import sys
from pathlib import Path

# Ensure dino package is importable as top-level modules inside it expect relative imports
repo_root = Path(__file__).resolve().parent
dino_dir = repo_root / 'dino'
if str(dino_dir) not in sys.path:
    sys.path.insert(0, str(dino_dir))

import main as dino_main  # dino/main.py


def main():
    parser = argparse.ArgumentParser('DINO', parents=[dino_main.get_args_parser()])
    args = parser.parse_args()
    if getattr(args, 'disable_wand', False):
        args.disable_wandb = True

    # DataAugmentationDINO in dino/main.py references module-level args
    # Set it so the imported module can resolve it.
    setattr(dino_main, 'args', args)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    dino_main.train_dino(args)


if __name__ == '__main__':
    main()
