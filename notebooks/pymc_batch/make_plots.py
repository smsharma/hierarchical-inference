from argparse import ArgumentParser
import pickle, yaml, os, glob
import numpy as np

def load_data(campaigndir, POI_name = "sigfrac", models = ["MCMC_x", "MCMC_clf_nom", "MCMC_clf_marg"]):
    configpath = os.path.join(campaigndir, "config.yaml")

    with open(configpath, 'r') as configfile:
        config = yaml.safe_load(configfile)

    POI_means = {model: [] for model in models}
    POI_sigmas = {model: [] for model in models}

    for resfile_path in glob.glob(os.path.join(campaigndir, "output", "*.pkl")):
        with open(resfile_path, 'rb') as resfile:
            data = pickle.load(resfile)
            for model in models:
                POI_means[model] += data[f"means_{POI_name}"][model]
                POI_sigmas[model] += data[f"sigmas_{POI_name}"][model]
            
    retdict = {"POI_val_truth": config[POI_name],
               "sig_mean": config["sig_mean"]
    }

    retdict["POI_mu_median"] = {model: np.mean(POI_means[model]) for model in models}
    retdict["POI_sigma_median"] = {model: np.mean(POI_sigmas[model]) for model in models}

    return retdict

def MakePlots(campaigndirs, plotdir):

    all_data = []
    for campaigndir in campaigndirs:
        all_data.append(load_data(campaigndir))

    print(all_data)

    # separate according to model name etc.

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--campaigndirs", action = "store", nargs = '+', dest = "campaigndirs")
    parser.add_argument("--plotdir", action = "store", dest = "plotdir")
    args = vars(parser.parse_args())

    MakePlots(**args)
