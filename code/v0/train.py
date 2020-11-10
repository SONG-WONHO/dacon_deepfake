import os
import sys
import json
import warnings
import argparse
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim

from utils import *
from data import *
from transform import get_transform
from model import get_model
from learner import Learner

warnings.filterwarnings("ignore")


class CFG:
    # path
    root_path = "./input/"
    log_path = './log/'
    model_path = './model/'

    # model
    model_name = "BaseModel"
    backbone_name = "efficientnet-b0"

    # train
    batch_size = 64
    learning_rate = 5e-4
    num_epochs = 40

    # etc
    seed = 42
    workers = 1
    num_targets = 1
    debug = False
    n_folds = 5
    val_fold = 0


def main():
    """ main function
        """

    ### header
    parser = argparse.ArgumentParser()

    # path
    parser.add_argument('--root-path', default=CFG.root_path,
                        help="root path")
    parser.add_argument('--log-path', default=CFG.log_path,
                        help="log path")
    parser.add_argument('--model-path', default=CFG.model_path,
                        help="model path")
    parser.add_argument('--pretrained-path',
                        help='pretrained path')

    # image
    parser.add_argument('--transform-version', default=0, type=int,
                        help="image transform version ex) 0, 1, 2 ...")
    parser.add_argument('--image-size', default=64, type=int,
                        help="image size(64)")

    # model
    parser.add_argument('--model-name', default=CFG.model_name,
                        help=f"model name({CFG.model_name})")
    parser.add_argument('--backbone-name', default=CFG.backbone_name,
                        help=f"backbone name({CFG.backbone_name})")

    # learning
    parser.add_argument('--batch-size', default=CFG.batch_size, type=int,
                        help=f"batch size({CFG.batch_size})")
    parser.add_argument('--learning-rate', default=CFG.learning_rate,
                        type=float,
                        help=f"learning rate({CFG.learning_rate})")
    parser.add_argument('--num-epochs', default=CFG.num_epochs, type=int,
                        help=f"number of epochs({CFG.num_epochs})")

    # etc
    parser.add_argument("--seed", default=CFG.seed, type=int,
                        help=f"seed({CFG.seed})")
    parser.add_argument("--workers", default=CFG.workers, type=int,
                        help=f"number of workers({CFG.workers})")
    parser.add_argument("--debug", action="store_true",
                        help="debug mode")
    parser.add_argument("--val-fold", default=CFG.val_fold,
                        choices=[list(range(0, CFG.n_folds))],
                        help=f"fold number for validation({CFG.val_fold})")

    args = parser.parse_args()

    # path
    CFG.root_path = args.root_path
    CFG.model_path = args.model_path
    CFG.log_path = args.log_path
    CFG.pretrained_path = args.pretrained_path

    # image
    CFG.transform_version = args.transform_version
    CFG.image_size = args.image_size

    # model
    CFG.model_name = args.model_name
    CFG.backbone_name = args.backbone_name

    # learning
    CFG.batch_size = args.batch_size
    CFG.learning_rate = args.learning_rate
    CFG.num_epochs = args.num_epochs

    # etc
    CFG.seed = args.seed
    CFG.workers = args.workers
    CFG.debug = args.debug

    # get device
    CFG.device = get_device()

    # get version
    _, version, _ = sys.argv[0].split('/')
    CFG.version = version

    # update log path
    if not CFG.debug:
        CFG.log_path = os.path.join(CFG.log_path, CFG.version)
        os.makedirs(CFG.log_path, exist_ok=True)
        CFG.log_path = os.path.join(CFG.log_path,
                                    f'exp_{get_exp_id(CFG.log_path, prefix="exp_")}')
        os.makedirs(CFG.log_path, exist_ok=True)
    else:
        CFG.log_path = os.path.join(CFG.log_path, "debug")
        os.makedirs(CFG.log_path, exist_ok=True)
        CFG.log_path = os.path.join(CFG.log_path, "debug")
        os.makedirs(CFG.log_path, exist_ok=True)

    # update model path
    if not CFG.debug:
        CFG.model_path = os.path.join(CFG.model_path, version)
        os.makedirs(CFG.model_path, exist_ok=True)
        CFG.model_path = os.path.join(CFG.model_path,
                                      f'exp_{get_exp_id(CFG.model_path, prefix="exp_")}')
        os.makedirs(CFG.model_path, exist_ok=True)
    else:
        CFG.model_path = os.path.join(CFG.model_path, "debug")
        os.makedirs(CFG.model_path, exist_ok=True)
        CFG.model_path = os.path.join(CFG.model_path, "debug")
        os.makedirs(CFG.model_path, exist_ok=True)

    pprint({k: v for k, v in dict(CFG.__dict__).items() if '__' not in k})
    json.dump(
        {k: v for k, v in dict(CFG.__dict__).items() if '__' not in k},
        open(os.path.join(CFG.log_path, 'CFG.json'), "w"))
    print()

    ### Seed all
    seed_everything(CFG.seed)

    ### Data Related
    # load raw data
    print("Load Raw Data")
    train_df, test_df, ss_df = load_data(CFG)

    # preprocess data
    print("Preprocess Data")
    train_df = preprocess_data(CFG, train_df)

    # split data
    print("Split Data")
    train_df, valid_df = split_data(CFG, train_df)
    train_df = train_df.sample(10000)

    # get transform
    print("Get Transforms")
    train_transforms, test_transforms = get_transform(CFG)

    # get dataset
    print("Get Dataset")
    trn_data = DFDDataset(CFG, train_df, train_transforms)
    val_data = DFDDataset(CFG, valid_df, test_transforms)

    ### Model related
    # get learner
    learner = Learner(CFG)
    learner.name = f"model.fold_{CFG.val_fold}"
    if CFG.pretrained_path:
        print("Load Pretrained Model")
        print(f"... Pretrained Info - {CFG.pretrained_path}")
        learner.load(CFG.pretrained_path, f"model_state_dict")
        model = learner.best_model.to(CFG.deivce)
    else:
        print(f"Load Model")
        model = get_model(CFG)
        model = model.to(CFG.device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # get optimizer
    optimizer = optim.Adam(model.parameters(), lr=CFG.learning_rate)

    # get scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, verbose=False, factor=0.5)

    ### train related
    # train model
    learner.train(trn_data, val_data, model, optimizer, scheduler)


if __name__ == "__main__":
    main()
