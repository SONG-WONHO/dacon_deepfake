""" Commonly used functions
"""
import os
import random
import torch
import numpy as np
import pandas as pd


def get_exp_id(path="./model/", prefix="exp_"):
    """Get new experiement ID

    Args:
        path: Where the "model/" or "log/" used in the project is stored.
        prefix: Experiment ID ahead.

    Returns:
        Experiment ID
    """
    files = set([int(d.replace(prefix, "")) for d in os.listdir(path) if prefix in d])
    if len(files):
        return min(set(range(0, max(files) + 2)) - files)
    else:
        return 0


def seed_everything(seed=42):
    """Seed All

    Args:
        seed: seed number
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_device():
    """Get device type

    Returns: device, "cpu" if cuda is available else "cuda"
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def count_parameters(model):
    """ Count model paramters

    Args:
        model: model object

    Returns: number of model paramters required gradient
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)