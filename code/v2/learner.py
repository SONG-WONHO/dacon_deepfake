import os
import gc
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from model import get_model


# average meter
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# loss function
def loss_func(pred, target):
    return nn.BCEWithLogitsLoss()(pred, target)


class Learner(object):
    def __init__(self, config):
        self.config = config
        self.best_model = None
        self.logger = None
        self.name = "model"

    def train(self, trn_data, val_data, model, optimizer, scheduler, attack):

        # dataloader
        train_loader = DataLoader(
            trn_data,
            batch_size=self.config.batch_size, shuffle=True,
            num_workers=self.config.workers, pin_memory=True,
        )

        valid_loader = DataLoader(
            val_data,
            batch_size=self.config.batch_size * 2, shuffle=False,
            num_workers=self.config.workers, pin_memory=True,
        )

        # loger
        logger = self._create_logger()

        # training
        best_metric = 1e-8
        for epoch in range(self.config.num_epochs):
            # train one epoch
            tr_loss = self._train_one_epoch(
                train_loader, model, optimizer, scheduler, attack)

            # validate one epoch
            vl_loss, vl_metric, adv_metric = self._valid_one_epoch(
                valid_loader, model, attack)

            # logging
            logger.loc[epoch] = [
                np.round(tr_loss, 4),
                np.round(vl_loss, 4),
                np.round(vl_metric, 4),
                np.round(adv_metric, 4),
                optimizer.param_groups[0]['lr']]

            logger.to_csv(os.path.join(
                self.config.log_path, f'log.{self.name.split(".")[-1]}.csv'))

            # save model
            if best_metric < logger.loc[epoch, 'val_metric']:
                print(f"... From {best_metric:.4f} To {logger.loc[epoch, 'val_metric']:.4f}")
                best_metric = logger.loc[epoch, 'val_metric']
                self.best_model = copy.deepcopy(model)
                name = self.name
                self.name = f"{name}.epoch_{epoch}"
                self.save()
                self.name = f"{name}.best"
                self.save()
                self.name = name

            scheduler.step(metrics=vl_loss)

        self.logger = logger

    def predict(self, tst_data):
        model = self.best_model

        test_loader = DataLoader(
            tst_data,
            batch_size=self.config.batch_size * 2, shuffle=False,
            num_workers=0, pin_memory=False
        )

        pred_final = []

        model.eval()

        test_loader = tqdm(test_loader, leave=False)

        for X_batch, _ in test_loader:
            X_batch = X_batch.to(self.config.device)

            with torch.no_grad():
                preds = model(X_batch)

            preds = preds.cpu().detach()

            pred_final.append(preds)

        pred_final = torch.cat(pred_final, dim=0)

        return pred_final

    def save(self):
        if self.best_model is None:
            print("Must Train before save !")
            return

        torch.save({
            "logger": self.logger,
            "model_state_dict": self.best_model.cpu().state_dict(),
        }, f"{os.path.join(self.config.model_path, self.name)}.pt")

    def load(self, path, name=None):
        ckpt = torch.load(path)
        self.logger = ckpt['logger']
        model_state_dict = ckpt[name]
        model = get_model(self.config)
        try:
            model.load_state_dict(model_state_dict)
            print("... Single GPU (Train)")
        except:
            def strip_module_str(v):
                if v.startswith('module.'):
                    return v[len('module.'):]

            model_state_dict = {strip_module_str(k): v for k, v in model_state_dict.items()}
            model.load_state_dict(model_state_dict)
            print("... Multi GPU (Train)")

        self.best_model = model.to(self.config.device)
        print("... Model Loaded!")

    def _train_one_epoch(self, train_loader, model, optimizer, scheduler, attack):
        losses = AverageMeter()

        model.train()

        train_iterator = tqdm(train_loader, leave=False)
        for X_batch, y_batch in train_iterator:
            X_batch = X_batch.to(self.config.device)
            y_batch = y_batch.to(self.config.device).type(torch.float32)
            X_batch = attack.perturb(X_batch, y_batch, 'mean', True)

            batch_size = X_batch.size(0)

            preds = model(X_batch)

            loss = loss_func(preds.view(-1), y_batch.view(-1))
            losses.update(loss.item(), batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_iterator.set_description(
                f"train bce:{losses.avg:.4f}, lr:{optimizer.param_groups[0]['lr']:.6f}")

        return losses.avg

    def _valid_one_epoch(self, valid_loader, model, attack):
        losses = AverageMeter()
        true_final, pred_final, adv_final = [], [], []

        model.eval()

        valid_loader = tqdm(valid_loader, leave=False)
        for i, (X_batch, y_batch) in enumerate(valid_loader):
            X_batch = X_batch.to(self.config.device)
            y_batch = y_batch.to(self.config.device).type(torch.float32)
            X_adv = attack.perturb(X_batch, y_batch, 'mean', False)

            batch_size = X_batch.size(0)

            with torch.no_grad():
                preds = model(X_batch)
                preds_adv = model(X_adv)
                loss = loss_func(preds.view(-1), y_batch.view(-1))
                losses.update(loss.item(), batch_size)

            true_final.append(y_batch.cpu())
            pred_final.append(preds.detach().cpu())
            adv_final.append(preds_adv.detach().cpu())

            losses.update(loss.item(), batch_size)

            valid_loader.set_description(f"valid ce:{losses.avg:.4f}")

        true_final = torch.cat(true_final, dim=0)
        pred_final = torch.cat(pred_final, dim=0).view(-1)
        pred_final = torch.sigmoid(pred_final)
        adv_final = torch.cat(adv_final, dim=0).view(-1)
        adv_final = torch.sigmoid(adv_final)

        vl_score = accuracy_score(
            true_final.cpu().numpy(), np.round(pred_final.cpu().numpy()))
        adv_score = accuracy_score(
            true_final.cpu().numpy(), np.round(adv_final.cpu().numpy())
        )

        return losses.avg, vl_score, adv_score

    def _create_logger(self):
        log_cols = ['tr_loss', 'val_loss', 'val_metric', 'adv_metric', 'lr']
        return pd.DataFrame(
            index=range(self.config.num_epochs), columns=log_cols)
