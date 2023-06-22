import torch
import torch.nn as nn
import torch.distributions as dist

import numpy as np
import scipy.stats as sps

import pytorch_lightning as pl
from einops import rearrange


from models.utils import build_mlp
from models.utils import ResidualMLP


class DynamicalBayes(nn.Module):
    def __init__(self, n_in=1, n_embedding=128, seq_length=10):
        super(DynamicalBayes, self).__init__()

        self.enc = build_mlp(input_dim=n_in, hidden_dim=n_embedding, output_dim=n_embedding, layers=2)  # Per-event embedding network
        self.resnet = ResidualMLP(input_size=2, hidden_size=int(2 * n_embedding), num_cond_features=n_embedding, num_layers=2)
        self.seq_length = seq_length
        self.prior = torch.Tensor([0.0, np.log(3.0)])  # mu and log-sigma of prior

    def forward(self, x):
        n_batch = x.shape[0]

        x = rearrange(x, "batch n_set n_feat -> (batch n_set) n_feat", batch=n_batch)
        x = self.enc(x)
        x = rearrange(x, "(batch n_set) n_out -> batch n_set n_out", batch=n_batch)

        seq_length = x.shape[1]

        if self.train:
            idx_setperm = torch.randperm(seq_length)  # Permutation indices
            x = x[:, idx_setperm, :]  # Permute set elements

        # Set prior
        pred = torch.ones(x.shape[0], 2).to(x)
        pred[:, :] = torch.Tensor(self.prior)[None, :]

        preds = torch.zeros(seq_length, x.shape[0], 2).to(x)

        for i_set in range(seq_length):
            pred = self.resnet(pred, x[:, i_set, :])
            preds[i_set, :, :] = pred

        return torch.Tensor(preds).permute(1, 0, 2)  # Batch dim first


class DynamicalBayesPL(pl.LightningModule):
    def __init__(
        self,
        optimizer=torch.optim.AdamW,
        optimizer_kwargs={"weight_decay": 1e-4},
        n_in=1,
        n_embedding=128,
        seq_length=100,
        lr=1e-3,
        max_epochs=50,
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
    ):
        super().__init__()

        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler = scheduler
        self.scheduler_kwargs = {"T_max": max_epochs}
        self.lr = lr
        self.seq_length = seq_length

        self.net = DynamicalBayes(n_in=n_in, n_embedding=n_embedding, seq_length=seq_length)

    def forward(self, x, y):
        x = self.net(x)
        pred_mu, pred_log_sigma = x[:, :, 0], x[:, :, 1]
        pred_sigma = pred_log_sigma.exp().clip(1e-6, 100)
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
