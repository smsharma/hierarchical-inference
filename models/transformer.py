import torch
import torch.nn as nn
import torch.distributions as dist

import pytorch_lightning as pl
from einops import repeat, rearrange
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

def create_causal_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask==1, float('-inf'))
    return mask

class TransformerUpdateNet(nn.Module):
    def __init__(self, n_in=1, seq_length=100, n_embedding=128, n_out=2, dropout=0.0):
        super(TransformerUpdateNet, self).__init__()

        self.embedder = build_mlp(n_in, int(2 * n_embedding), n_embedding, 2)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=n_embedding, nhead=4, dim_feedforward=int(4 * n_embedding), norm_first=False, batch_first=True, activation="relu", dropout=dropout), num_layers=6)
        self.predicter = build_mlp(n_embedding, int(2 * n_embedding), n_out, 3)
        self.mask = create_causal_mask(seq_length)

    def forward(self, x):
        embedding = self.embedder(x)
        x = self.transformer(embedding, mask=self.mask.to(x.device))
        x = self.predicter(x)
        return x


class TransformerUpdateNetPL(pl.LightningModule):
    def __init__(
        self,
        optimizer=torch.optim.AdamW,
        dim=3,
        optimizer_kwargs={"weight_decay": 1e-5},
        n_in=1,
        seq_length=100,
        n_embedding=256,
        n_out=2,
        lr=3e-4,
        max_epochs=50,
        # scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
        scheduler=WarmupCosineDecayScheduler,
    ):
        super().__init__()

        self.dim = dim

        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler = scheduler
        # self.scheduler_kwargs = {"T_max":max_epochs}
        self.scheduler_kwargs = {"warmup_epochs":int(0.1 * max_epochs), "total_epochs": max_epochs, "max_lr":lr}
        self.lr = lr
        self.seq_length = seq_length

        self.net = TransformerUpdateNet(n_in=n_in, n_out=n_out, n_embedding=n_embedding, seq_length=seq_length)

    def forward(self, x):
        x = self.net(x)
        return x

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr, **self.optimizer_kwargs)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": self.scheduler(optimizer, **self.scheduler_kwargs), "interval": "epoch", "monitor": "val_loss", "frequency": 1}}

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = repeat(y, "batch n_dim -> batch n_seq n_dim", n_seq=self.seq_length)
        x = self(x)
        pred_mu, pred_log_sigma = torch.chunk(x, 2, -1)
        pred_sigma = pred_log_sigma.exp()
        loss = -(dist.Normal(pred_mu, pred_sigma).log_prob(y)).mean()
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = repeat(y, "batch n_dim -> batch n_seq n_dim", n_seq=self.seq_length)
        x = self(x)
        pred_mu, pred_log_sigma = torch.chunk(x, 2, -1)
        pred_sigma = pred_log_sigma.exp()
        loss = -(dist.Normal(pred_mu, pred_sigma).log_prob(y)).mean()
        self.log("val_loss", loss, on_epoch=True)
        return loss
