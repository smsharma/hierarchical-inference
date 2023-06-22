import torch
import torch.nn as nn
import torch.distributions as dist

import pytorch_lightning as pl
from einops import rearrange, repeat

from models.utils import build_mlp


class RandomMaskingLSTM(nn.Module):
    def __init__(self, n_in=1, n_out=2, n_embedding=256, seq_length=100):
        super(RandomMaskingLSTM, self).__init__()

        self.enc = build_mlp(input_dim=n_in, hidden_dim=n_embedding, output_dim=n_embedding // 2, layers=4)  # Per-event embedding network
        self.dec = nn.LSTM(input_size=n_embedding // 2, hidden_size=n_embedding, num_layers=1, proj_size=n_embedding // 2, batch_first=True, bidirectional=False)
        
        self.predicter = build_mlp(n_embedding // 2, n_embedding, n_out, 3)

        self.seq_length = seq_length

    def forward(self, x):
        x = rearrange(x, "batch n_set n_feat -> (batch n_set)  n_feat", n_set=self.seq_length)
        x = self.enc(x)
        x = rearrange(x, "(batch n_set) n_out -> batch n_set n_out", n_set=self.seq_length)

        x, (h, c) = self.dec(x)
        
        # Element-wise predicter
        x = self.predicter(x)

        return x


class LSTMRandomMaskingPL(pl.LightningModule):
    def __init__(
        self,
        optimizer=torch.optim.AdamW,
        optimizer_kwargs={"weight_decay": 1e-4},
        n_in=1,
        n_out=2,
        seq_length=100,
        lr=3e-4,
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

        self.net = RandomMaskingLSTM(n_in=n_in, n_out=n_out, seq_length=seq_length)

    def forward(self, x, y):
                
        x = self.net(x)
        y = repeat(y, "batch n_dim -> batch n_seq n_dim", n_seq=self.seq_length)
        pred_mu, pred_log_sigma = torch.chunk(x, 2, -1)
        # pred_sigma = pred_log_sigma.exp()
        pred_sigma = nn.Softplus()(pred_log_sigma)
        log_prob = dist.Normal(pred_mu, pred_sigma).log_prob(y).mean()
        return log_prob

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr, **self.optimizer_kwargs)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": self.scheduler(optimizer, **self.scheduler_kwargs), "interval": "epoch", "monitor": "val_loss", "frequency": 1}}

    def training_step(self, batch, batch_idx):
        
        x, y = batch
        
        # Randomly permute sequence while training
        idx_permute = torch.randperm(x.shape[1])
        x = x[:, idx_permute, ...]

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
