import os, sys, argparse, json
from pprint import pprint
import warnings

import pandas as pd

import torch
import torch.nn as nn

from data import *
from transform import get_transform
from model import get_model
from learner import Learner
from utils import *

warnings.filterwarnings("ignore")


class CFG:
    # path
    root_path = "./input/"
    save_path = './submission/'
    sub_name = 'submission.csv'

    # learning
    batch_size = 64
    workers = 0
    seed = 42


def main():
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--root-path', default=CFG.root_path,
                        help="root path")
    parser.add_argument('--save-path', default=CFG.save_path,
                        help="save path")
    parser.add_argument('--sub-name', default=CFG.sub_name,
                        help="submission name")

    # learning
    parser.add_argument('--batch-size', default=CFG.batch_size, type=int,
                        help=f"batch size({CFG.batch_size})")
    parser.add_argument("--workers", default=CFG.workers, type=int,
                        help=f"number of workers({CFG.workers})")
    parser.add_argument("--seed", default=CFG.seed, type=int,
                        help=f"seed({CFG.seed})")

    # version
    parser.add_argument('--version', type=int)
    parser.add_argument('--exp-id', type=int)

    # etc
    parser.add_argument('--tta', action='store_true', default=False)

    args = parser.parse_args()

    CFG.root_path = args.root_path
    CFG.save_path = args.save_path
    CFG.sub_name = args.sub_name

    CFG.batch_size = args.batch_size
    CFG.workers = args.workers
    CFG.seed = args.seed

    CFG.model_path = f"./model/v{args.version}/exp_{args.exp_id}/"
    CFG.log_path = f"./log/v{args.version}/exp_{args.exp_id}/"

    CFG.tta = args.tta

    # get device
    CFG.device = get_device()

    # load train environment
    env = json.load(open(os.path.join(CFG.log_path, 'CFG.json'), 'r'))
    for k, v in env.items(): setattr(CFG, k, v)

    loss, metric = 0, 0
    for fold in range(CFG.n_folds):
        fn = os.path.join(CFG.log_path, f"log.fold_{fold}.csv")
        score = pd.read_csv(fn).sort_values("val_metric", ascending=False).iloc[0]
        loss += score['val_loss'] #/ CFG.n_folds
        metric += score['val_metric'] #/ CFG.n_folds
        break

    CFG.sub_name = f"submission." \
                   f"ver_{args.version}." \
                   f"exp_{args.exp_id}." \
                   f"loss_{loss:.4f}." \
                   f"metric_{metric:.4f}.csv"

    if CFG.tta:
        CFG.sub_name = "tta." + CFG.sub_name

    pprint({k: v for k, v in dict(CFG.__dict__).items() if '__' not in k})
    print()

    ### seed all
    seed_everything(CFG.seed)

    ### Data related logic
    # load data
    print("Load Raw Data")
    _, test_df, ss_df = load_data(CFG)
    print()

    # get transform
    print("Get Transform")
    _, test_transforms = get_transform(CFG)
    print()

    # dataset
    tst_data = DFDDataset(CFG, test_df, test_transforms)

    # if tta
    if CFG.tta:
        test_transforms = get_tta_transform(CFG)
        tst_data = DFDDataset(CFG, test_df, test_transforms)

    final_preds = np.zeros(test_df.shape[0])

    # folds
    for fold in range(CFG.n_folds):
        print(f"========== Fold: {fold} ==========")
        # load learner
        print("Load Model")
        model_name = f'model.fold_{fold}.best.pt'
        learner = Learner(CFG)
        learner.load(
            os.path.join(CFG.model_path, model_name), f"model_state_dict")

        # prediction
        if not CFG.tta:
            test_preds = torch.sigmoid(
                learner.predict(tst_data).view(-1)).numpy()

        else:
            test_preds = np.zeros(test_df.shape[0])
            for _ in range(4):
                test_preds += torch.sigmoid(
                    learner.predict(tta_data).view(-1)).numpy() / 4

        final_preds += test_preds #/ CFG.n_folds
        print()
        break

    image_path = [path.replace(".npy", "").replace("./input/test_face_margin/", "")
                  for path in test_df['image_path'].values]
    print(final_preds.max(), final_preds.min())
    test_df = pd.DataFrame({"path": image_path, "y": np.round(final_preds)})
    test_df = test_df.set_index("path")
    ss_df = test_df.loc[ss_df['path']].reset_index()[['path', 'y']]
    ss_df.to_csv(os.path.join(CFG.save_path, f"{CFG.sub_name}"), index=False)

    print(ss_df.head())


if __name__ == '__main__':
    main()
