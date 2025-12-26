from __future__ import print_function

import argparse
import pdb
import os
import math
import sys

# internal imports
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_MIL_Dataset, Generic_Split

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np


def main(train_dataset, test_dataset, args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    all_test_f1 = []
    all_val_f1 = []
    folds = np.arange(start, end)
    full_train_dataset = train_dataset
    full_test_dataset = test_dataset
    for i in folds:
        seed_torch(args.seed)
        train_split_path = format_split_path(args.train_split_csv, i)
        val_split_path = format_split_path(args.val_split_csv, i)
        test_split_path = format_split_path(args.test_split_csv, i)

        train_split = resolve_split(train_split_path, [full_train_dataset])
        if train_split is None:
            train_split = Generic_Split(full_train_dataset.slide_data.copy(),
                                        data_dir=full_train_dataset.data_dir,
                                        num_classes=full_train_dataset.num_classes)
        val_split = resolve_split(val_split_path, [full_train_dataset, full_test_dataset]) if val_split_path else None
        test_split = resolve_split(test_split_path, [full_train_dataset, full_test_dataset])

        if val_split is None:
            val_split = test_split
        if test_split is None:
            test_split = val_split if val_split is not None else train_split

        datasets = (train_split, val_split, test_split)
        results, test_auc, val_auc, test_acc, val_acc, test_f1, val_f1 = train(datasets, i, args, embedding_dim)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        all_test_f1.append(test_f1)
        all_val_f1.append(val_f1)
        # write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 'val_auc': all_val_auc, 'val_f1': all_val_f1,
                             'test_acc': all_test_acc, 'val_acc': all_val_acc, 'test_f1': all_test_f1})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))


# Generic training settings
# Default split templates (auto-detect BRACS splits if available)
split_root = Path(__file__).resolve().parent / 'splits'
bracs_split_dir = split_root / 'bracs_k1'
if (bracs_split_dir / 'splits_0.csv').exists():
    DEFAULT_TRAIN_SPLIT = str(bracs_split_dir / 'splits_{fold}.csv')
    DEFAULT_VAL_SPLIT = str(bracs_split_dir / 'splits_{fold}_val.csv')
    DEFAULT_TEST_SPLIT = str(bracs_split_dir / 'splits_{fold}_test.csv')
else:
    DEFAULT_TRAIN_SPLIT = str(split_root / 'train' / 'splits_{fold}.csv')
    DEFAULT_VAL_SPLIT = None
    DEFAULT_TEST_SPLIT = str(split_root / 'test' / 'splits_{fold}.csv')


parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=1, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default='./splits/bracs',
                    help='manually specify the set of splits to use, '
                         + 'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enable dropout (p=0.25)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                    help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', 'abmil', 'transmil', 'dtfd', 'dsmil'], default='clam_sb',
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small',
                    help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping'])
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                    help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                    help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False,
                    help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')

parser.add_argument('--arch', type=str, help='Encoder Architecture to use. ')
parser.add_argument('--image_size', default=224, type=int, help='Image Size of global views.')
parser.add_argument("--source_level", type=str)
parser.add_argument("--target_level", type=str)
parser.add_argument('--clam_data_root', type=str, default=None,
                    help='Optional override for the clam_data directory (default: ../clam_data relative to this file).')
parser.add_argument('--train_split_csv', type=str, default=DEFAULT_TRAIN_SPLIT,
                    help='Path template to the training split CSV (use {fold} placeholder for fold index).')
parser.add_argument('--val_split_csv', type=str, default=DEFAULT_VAL_SPLIT,
                    help='Path template to the validation split CSV (optional).')
parser.add_argument('--test_split_csv', type=str, default=DEFAULT_TEST_SPLIT,
                    help='Path template to the test split CSV (use {fold} placeholder for fold index).')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(args.seed)

settings = {'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

if args.model_type in ['clam_sb', 'clam_mb']:
    settings.update({'bag_weight': args.bag_weight,
                     'inst_loss': args.inst_loss,
                     'B': args.B})

# embedding_dim = 1024
embedding_map = {'vit-t': 768,
                 'vim-t': 192,
                 'vim-t-plus': 384,
                 'vit-s': 1536,
                 'vim-s': 384
                 }
embedding_dim = embedding_map[args.arch]

default_clam_root = Path(__file__).resolve().parent.parent / 'clam_data'
clam_data_root = Path(args.clam_data_root).expanduser().resolve() if args.clam_data_root else default_clam_root
clam_subset = clam_data_root / f'{args.image_size}_{args.source_level}at{args.target_level}' / args.arch
train_csv = clam_subset / 'training' / 'tumor_vs_normal.csv'
test_csv = clam_subset / 'testing' / 'tumor_vs_normal.csv'

def infer_label_dict(csv_paths):
    norm_to_idx = {}
    mapping = {}
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if 'label' not in df.columns:
            raise ValueError(f"'label' column missing from {csv_path}")
        for raw_label in df['label'].dropna().unique():
            label_str = str(raw_label).strip()
            if not label_str:
                continue
            norm = label_str.upper()
            if norm not in norm_to_idx:
                norm_to_idx[norm] = len(norm_to_idx)
            idx = norm_to_idx[norm]
            mapping[label_str] = idx
            mapping[label_str.lower()] = idx
            mapping[norm] = idx
    return mapping, len(norm_to_idx)

label_dict, num_label_classes = infer_label_dict([train_csv, test_csv])
args.n_classes = num_label_classes

def format_split_path(template, fold_idx):
    if not template:
        return None
    return template.format(fold=fold_idx)

def load_ids_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    candidate_cols = [c for c in df.columns if c not in ('Unnamed: 0',)]
    priority = ['train', 'val', 'test', 'slide_id']
    col = None
    for key in priority:
        if key in df.columns:
            col = key
            break
    if col is None and candidate_cols:
        col = candidate_cols[0]
    if col is None:
        return []
    return df[col].dropna().astype(str).tolist()

def create_split_from_ids(dataset, slide_ids):
    if not slide_ids:
        return None
    mask = dataset.slide_data['slide_id'].astype(str).isin(set(slide_ids))
    df_slice = dataset.slide_data[mask].reset_index(drop=True)
    if df_slice.empty:
        return None
    return Generic_Split(df_slice, data_dir=dataset.data_dir, num_classes=dataset.num_classes)

def resolve_split(csv_path, dataset_candidates):
    if not csv_path:
        return None
    csv_file = Path(csv_path)
    if not csv_file.is_file():
        print(f'Warning: split file {csv_file} not found, skipping.')
        return None
    slide_ids = load_ids_from_csv(csv_file)
    for dataset in dataset_candidates:
        split = create_split_from_ids(dataset, slide_ids)
        if split is not None and len(split) > 0:
            return split
    raise ValueError(f'Unable to match any slide ids from {csv_file} to available datasets.')

train_dataset = Generic_MIL_Dataset(csv_path=str(train_csv),
                              data_dir=str(clam_subset / 'training'),
                              shuffle=True,
                              seed=args.seed,
                              print_info=True,
                              label_dict=label_dict,
                              patient_strat=False,
                              ignore=[])

test_dataset = Generic_MIL_Dataset(csv_path=str(test_csv),
                              data_dir=str(clam_subset / 'testing'),
                              shuffle=True,
                              seed=args.seed,
                              print_info=True,
                              label_dict=label_dict,
                              patient_strat=False,
                              ignore=[])

if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))

if __name__ == "__main__":
    results = main(train_dataset, test_dataset, args)
    print("finished!")
    print("end script")
