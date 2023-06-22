import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F

import pytorch_lightning as pl
from einops import rearrange, repeat

from models.utils import build_mlp

import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, max_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.max_lr = max_lr
        super(WarmupCosineDecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = self.last_epoch / self.warmup_epochs
            return [base_lr + (self.max_lr - base_lr) * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine decay
            t = self.last_epoch - self.warmup_epochs
            T = self.total_epochs - self.warmup_epochs
            decay_factor = 0.5 * (1 + math.cos(math.pi * t / T))
            return [self.max_lr * decay_factor for _ in self.base_lrs]

class DeepSet(nn.Module):
    def __init__(self, n_in=1, n_out=2, n_embedding=512, seq_length=100, sum_aggregations=False):
        super(DeepSet, self).__init__()

        self.enc = build_mlp(input_dim=n_in, hidden_dim=n_embedding, output_dim=n_embedding, layers=3)  # Per-event embedding network
        self.dec = build_mlp(input_dim=n_embedding, hidden_dim=int(2 * n_embedding), output_dim=n_out, layers=3)

        self.seq_length = seq_length
        self.sum_aggregations = sum_aggregations

    def forward(self, x):
        n_batch = x.shape[0]

        lens = torch.randint(low=1, high=self.seq_length + 1, size=(n_batch,), dtype=torch.float)
        mask = (torch.arange(self.seq_length).expand(len(lens), self.seq_length) < torch.Tensor(lens)[:, None]).to(x.device)

        x = rearrange(x, "batch n_set n_feat -> (batch n_set)  n_feat", n_set=self.seq_length)
        x = self.enc(x)

        x = rearrange(x, "(batch n_set) n_out -> batch n_set n_out", n_set=self.seq_length)
        x = x * mask[:, :, None]

        x = x.sum(-2)

        # If using mean aggregation, divide by number of elements in set and append cardinality
        if not self.sum_aggregations:
            x /= mask.sum(1)[:, None]
            x = torch.cat([x, lens[:, None].to(x.device)], -1)  # Add cardinality for rho network

        x = self.dec(x)

        return x


class DeepSetPL(pl.LightningModule):
    def __init__(
        self,
        optimizer=torch.optim.AdamW,
        optimizer_kwargs={"weight_decay": 1e-5},
        n_in=1,
        n_out=2,
        n_embedding=256,
        seq_length=100,
        lr=3e-4,
        max_epochs=50,
        eps=1e-6,
        sum_aggregations=False,
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
        # scheduler=WarmupCosineDecayScheduler,
    ):
        super().__init__()

        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler = scheduler
        self.scheduler_kwargs = {"T_max":max_epochs}
        # self.scheduler_kwargs = {"warmup_epochs":int(0.1 * max_epochs), "total_epochs": max_epochs, "max_lr":lr}
        self.lr = lr
        self.seq_length = seq_length
        self.eps = eps

        self.net = DeepSet(n_in=n_in, n_out=n_out, seq_length=seq_length, sum_aggregations=sum_aggregations, n_embedding=n_embedding)

    def forward(self, x, y):
        x = self.net(x)
        pred_mu, pred_log_sigma = torch.chunk(x, 2, -1)
        pred_sigma = pred_log_sigma.exp() + self.eps  # Add epsilon to avoid numerical issues
        log_prob = dist.Normal(pred_mu, pred_sigma).log_prob(y).mean()
        return log_prob

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr, **self.optimizer_kwargs)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": self.scheduler(optimizer, **self.scheduler_kwargs), "interval": "epoch", "monitor": "val_loss", "frequency": 1}}

    def training_step(self, batch, batch_idx):
        x, y = batch
        log_prob = self(x, y)
        loss = -log_prob
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        log_prob = self(x, y)
        loss = -log_prob
        self.log("val_loss", loss, on_epoch=True)
        return loss
