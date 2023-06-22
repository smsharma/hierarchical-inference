from argparse import ArgumentParser
import numpy as np
import os, uuid, yaml, pickle
import pymc_utils

def RunMCMC(outdir, number_runs, configpath, store_trace = False):

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with open(configpath, 'r') as configfile:
        config = yaml.safe_load(configfile)

    dset_length = config["dset_length"]
    sigfrac = config["sigfrac"]
    sig_mean = config["sig_mean"]

    models = ["MCMC_x", "MCMC_clf_nom", "MCMC_clf_marg"]
    observables = [lambda x: x, pymc_utils.discriminant_nominal_signal, pymc_utils.discriminant_marg_signal]
    likelihoods = [pymc_utils.logp_x, pymc_utils.logp_disc_nominal_signal, pymc_utils.logp_disc_marg_signal]

    means_sigfrac = {model: [] for model in models}
    sigmas_sigfrac = {model: [] for model in models}
    means_mu_S = {model: [] for model in models}
    sigmas_mu_S = {model: [] for model in models}

    out_id = os.path.join(outdir, str(uuid.uuid4()))

    datasets_x = pymc_utils.make_datasets_x(dset_length = dset_length, num_dsets = number_runs, sigfrac = sigfrac, sig_mean = sig_mean)
    for ind, dataset_x in enumerate(datasets_x):
        for model, observable, likelihood in zip(models, observables, likelihoods):
            
            trace = pymc_utils.run_inference(observable(dataset_x), likelihood)

            print("----")
            print(model)
            print(np.mean(trace[("posterior", "sigfrac")]))
            print(np.std(trace[("posterior", "sigfrac")]))
            print("----")

            means_sigfrac[model].append(np.mean(trace[("posterior", "sigfrac")]))
            sigmas_sigfrac[model].append(np.std(trace[("posterior", "sigfrac")]))

            means_mu_S[model].append(np.mean(trace[("posterior", "mu_S")]))
            sigmas_mu_S[model].append(np.std(trace[("posterior", "mu_S")]))
        
            if store_trace:
                trace.to_hdf(f"{out_id}_trace_{ind}.h5", key = "trace", mode = 'w')

    data = {
        "means_sigfrac": means_sigfrac,
        "sigmas_sigfrac": sigmas_sigfrac,
        "means_mu_S": means_mu_S,
        "sigmas_mu_S": sigmas_mu_S
    }

    with open(out_id + ".pkl", 'wb') as outfile:
        pickle.dump(data, outfile, protocol = pickle.DEFAULT_PROTOCOL)

    print("done")

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--outdir", action = "store", dest = "outdir")
    parser.add_argument("--config", action = "store", dest = "configpath")
    parser.add_argument("--number_runs", action = "store", type = int, dest = "number_runs")
    args = vars(parser.parse_args())

    RunMCMC(**args)
