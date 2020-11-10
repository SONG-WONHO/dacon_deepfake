import os
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import cv2
from torch.utils.data import Dataset

import matplotlib.pyplot as plt


def load_data(config):
    # train dataframe
    tr_df = []
    tr_path = os.path.join(config.root_path, "train_face_margin")
    for root, dirs, files in os.walk(tr_path):
        if files:
            for file in files:
                img_path = os.path.join(root, file)
                item = img_path.split("/")[3:]
                if item[0] == "real":
                    item.insert(4, "None")
                item.append(img_path)
                tr_df.append(item)
    tr_df = pd.DataFrame(tr_df, columns=["target", "c1", "date", "c2",
                                         "algorithm", "c3", "image_name",
                                         "image_path"])

    # test dataframe
    te_path = os.path.join(config.root_path, "test_face_margin", "leaderboard")
    te_df = [os.path.join(te_path, f) for f in os.listdir(te_path)]
    te_df = pd.DataFrame(te_df, columns=["image_path"])
    te_df['target'] = np.nan
    te_df['sub_target'] = np.nan

    # sample submission
    ss_df = pd.read_csv(
        os.path.join(config.root_path, "test_face", "sample_submission.csv"))

    print(f"... Train Shape: {tr_df.shape} Test Shape: {te_df.shape}")
    return tr_df, te_df, ss_df


def preprocess_data(config, df):
    df['target'] = df['target'].map({"fake": 1, "real": 0})
    df['sub_target'] = df['algorithm'].map({
        "None": 0, "fsgan": 1, "dffs": 2, "fo": 3, "dfl": 4})
    return df


def stratified_group_k_fold(X, y, groups, k, seed=None):
    """ src: https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
    """
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std(
                [y_counts_per_fold[i][label] / y_distr[label] for i in
                 range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


def split_data(config, df):
    df['group'] = df['target'].astype(str) + "_" + df['c2']
    for fold, (tr_idx, vl_idx) in enumerate(stratified_group_k_fold(
                df, df['target'], df['group'], config.n_folds, config.seed)):

        train_df, valid_df = df.iloc[tr_idx], df.iloc[vl_idx]
        if fold == config.val_fold:
            break
    print(f"... Train Shape: {train_df.shape} Valid Shape: {valid_df.shape}")
    print(f"... Validation fold: {fold}")
    print(f"    ... Train Info: ")
    print(f"        ... Number of groups: {train_df['group'].nunique()}")
    print(f"        ... Target ratio(real/fake): "
          f"""{'/'.join(train_df['target'].
                        value_counts(normalize=True)[[0, 1]].
                        round(2).astype(str).values)}""")

    print(f"    ... Valid Info: ")
    print(f"        ... Number of groups: {valid_df['group'].nunique()}")
    print(f"        ... Target ratio(real/fake): "
          f"""{'/'.join(valid_df['target'].
                        value_counts(normalize=True)[[0, 1]].
                        round(2).astype(str).values)}""")

    return train_df, valid_df


class DFDDataset(Dataset):
    def __init__(self, config, df, transforms=None):
        self.config = config
        self.df = df[['image_path', 'target']].values
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fn, label = self.df[idx]
        im = np.load(fn)

        # Apply transformations
        if self.transforms:
            im = self.transforms(image=im)['image']

        return im, label

