import os
import numpy as np
import pandas as pd


def load_data(config):
    tr_df = []
    tr_path = os.path.join(config.root_path, "train")
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
    tr_df['target'] = tr_df['target'].map({"fake": 1, "real": 0})

    te_path = os.path.join(config.root_path, "test", "leaderboard")
    te_df = [os.path.join(te_path, f) for f in os.listdir(te_path)]
    te_df = pd.DataFrame(te_df, columns=["image_path"])

    ss_df = pd.read_csv(
        os.path.join(config.root_path, "test", "sample_submission.csv"))

    print(f"- Train Shape: {tr_df.shape} Test Shape: {te_df.shape}")
    return tr_df, te_df, ss_df


