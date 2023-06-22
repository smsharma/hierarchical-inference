import sys, os, yaml, pickle, uuid
sys.path.append(os.environ["ROOTDIR"])
from simulators.extended_particle_model import generate_mixture_model

from argparse import ArgumentParser
import torch
import numpy as np
from models.utils import build_mlp
import pytorch_lightning as pl
import torch.distributions as dist
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# model parameters: need to obey the hierarchy: sig_sigma_truth < bkg_sigma_truth < sig_mean_prior_sigma
sig_sigma_truth = 0.1
bkg_mean_truth, bkg_sigma_truth = 0.0, 1.0
sig_mean_prior_mu, sig_mean_prior_sigma = 1.0, 2.0
num_evts = 100

def make_targets_data(dset_length, num_dsets):

    # model parameters with some prior
    sigfracs_truth = np.random.uniform(0, 1, size = num_dsets)
    sig_means_truth = np.random.normal(sig_mean_prior_mu, sig_mean_prior_sigma, size = num_dsets)

    data = [np.expand_dims(generate_mixture_model(dset_length, sigfrac = cur_sigfrac, sigdist = (cur_sig_mean, sig_sigma_truth), 
                                                    bkgdist = (bkg_mean_truth, bkg_sigma_truth)), -1) \
                            for cur_sig_mean, cur_sigfrac in zip(sig_means_truth, sigfracs_truth)]

    data = torch.Tensor(data)
    targets = torch.Tensor(sigfracs_truth).unsqueeze(-1)

    return targets, data

class DeepSet(pl.LightningModule):

    def __init__(self, optimizer = torch.optim.AdamW, optimizer_kwargs = {"weight_decay": 1e-2}, lr = 1e-3,
                 max_epochs = 10, input_dim = 1, embedding_dim = 64, hidden_dim = 128, layers = 2):
        super().__init__()

        self.max_epochs = max_epochs
        self.optimizer = optimizer
        self.lr = lr
        self.optimizer_kwargs = optimizer_kwargs

        self.enc = build_mlp(input_dim = input_dim, output_dim = embedding_dim, 
                                hidden_dim = hidden_dim, layers = layers)
        self.dec = build_mlp(input_dim = embedding_dim, output_dim = 2,
                                hidden_dim = hidden_dim, layers = layers)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr = self.lr, **self.optimizer_kwargs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = self.max_epochs)
        return {"optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "val_loss",
                    "frequency": 1
                    }
                }

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred_mu, pred_sigma = self.forward(x)
        loss = -dist.Normal(pred_mu, pred_sigma).log_prob(y).mean()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred_mu, pred_sigma = self.forward(x)
        loss = -dist.Normal(pred_mu, pred_sigma).log_prob(y).mean()
        self.log("val_loss", loss)
        return loss

    def forward(self, x):
        embedding = self.enc.forward(x).mean(-2)
        pred = self.dec(embedding)
        pred_mu, pred_log_sigma = torch.chunk(pred, 2, -1)
        pred_sigma = pred_log_sigma.exp()
        return pred_mu, pred_sigma

def make_datasets_x(dset_length, num_dsets, sigfrac, sig_mean):
    data = [np.expand_dims(generate_mixture_model(dset_length, sigfrac = sigfrac, sigdist = (sig_mean, sig_sigma_truth), 
                                                  bkgdist = (bkg_mean_truth, bkg_sigma_truth)), -1) for cur in range(num_dsets)]
    return data

def run_training(model_outpath):
    targets_train, data_train = make_targets_data(num_evts, 500000)
    dataset_train = TensorDataset(data_train, targets_train)
    train_loader = DataLoader(dataset_train, batch_size = 256, num_workers = 8, pin_memory = True, shuffle = True)

    targets_val, data_val = make_targets_data(num_evts, 50000)
    dataset_val = TensorDataset(data_val, targets_val)
    val_loader = DataLoader(dataset_val, batch_size = 256, num_workers = 8, pin_memory = True, shuffle = False)

    max_epochs = 300
    model_ds = DeepSet(max_epochs = max_epochs)

    early_stop_callback = EarlyStopping(monitor = 'val_loss', patience = max_epochs)
    checkpoint_callback = ModelCheckpoint(monitor = "val_loss", filename = "{epoch:02d}-{val_loss:.2f}", 
                                          every_n_epochs = 1, save_top_k = max_epochs, dirpath = model_outpath)
    trainer = pl.Trainer(max_epochs = max_epochs, accelerator = 'gpu', devices = 1, callbacks = [early_stop_callback, checkpoint_callback])
    trainer.fit(model = model_ds, train_dataloaders = train_loader, val_dataloaders = val_loader)
    
    model_ds = DeepSet.load_from_checkpoint(checkpoint_callback.best_model_path)

    return model_ds

def run_evaluation(model_ds, campaigndir, sigfrac, sig_mean, num_dsets = 1000):

    if not os.path.exists(campaigndir):
        os.makedirs(campaigndir)

    configpath = os.path.join(campaigndir, "config.yaml")
    with open(configpath, 'w') as configfile:
        configfile.write(yaml.dump({"sigfrac": sigfrac, 
                                    "sig_mean": sig_mean, 
                                    "dset_length": num_evts}
                               ))

    outdir = os.path.join(campaigndir, "output")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    eval_data = torch.Tensor(make_datasets_x(dset_length = num_evts, num_dsets = num_dsets, 
                                             sigfrac = sigfrac, sig_mean = sig_mean))
    pred_mu, pred_sigma = model_ds.forward(eval_data)    

    data = {
        "means_sigfrac": pred_mu,
        "sigmas_sigfrac": pred_sigma
    }

    out_id = os.path.join(outdir, str(uuid.uuid4()))
    with open(out_id + ".pkl", 'wb') as outfile:
        pickle.dump(data, outfile, protocol = pickle.DEFAULT_PROTOCOL)        

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--outdir", action = "store", dest = "outdir")
    parser.add_argument("--run_evaluation", action = "store_true", default = False, dest = "run_evaluation")
    args = vars(parser.parse_args())

    sigfrac_truth = 0.2
    sig_means = np.linspace(0, 2, 100)
    
    model_outpath = os.path.join(args["outdir"], "model")
    model_ds = run_training(model_outpath = model_outpath)

    if args["run_evaluation"]:
        for sig_mean in sig_means:
            run_evaluation(model_ds, args["outdir"], sigfrac_truth, sig_mean)
