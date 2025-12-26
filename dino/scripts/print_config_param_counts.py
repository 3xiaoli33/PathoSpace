#!/usr/bin/env python3
import argparse
import warnings
import sys
from functools import partial


def main():
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='Print param counts for all models in dino/config.py')
    parser.add_argument('--only', type=str, default=None, help='Filter by arch name (substring match)')
    args = parser.parse_args()

    # Ensure we can import dino modules with relative imports
    sys.path.insert(0, 'dino')

    from config import configurations

    def _normalize_conf(conf: dict):
        conf = dict(conf)
        conf['num_classes'] = conf.get('num_classes', 2)
        if conf.get('norm_layer') == 'nn.LayerNorm':
            eps = conf.get('eps', 1e-6)
            conf['norm_layer'] = partial(__import__('torch').nn.LayerNorm, eps=eps)
        return conf

    results = {}
    for name, cfg in configurations.items():
        if args.only and args.only not in name:
            continue
        conf = _normalize_conf(cfg)

        try:
            from vim.models_mamba import VisionMamba
            model = VisionMamba(return_features=True, **conf)

            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            results[name] = (total, trainable)
        except Exception as e:
            results[name] = (None, None, repr(e))

    # Print results (in Millions)
    for k in sorted(results.keys()):
        v = results[k]
        if len(v) == 2 and v[0] is not None:
            total_m = v[0] / 1e6
            trainable_m = v[1] / 1e6
            print(f"{k:18s}  total: {total_m:.2f}M  trainable: {trainable_m:.2f}M")
        else:
            print(f"{k:18s}  ERROR: {v[2]}")


if __name__ == '__main__':
    main()
