import pymc as pm
import numpy as np
import sys, os
sys.path.append(os.environ["ROOTDIR"])
from simulators.extended_particle_model import generate_mixture_model

# model parameters without priors
sig_sigma_truth = 0.1
bkg_mean_truth, bkg_sigma_truth = 0.0, 1.0
sig_mean_prior_mu, sig_mean_prior_sigma = 1.0, 2.0

def make_datasets_x(dset_length, num_dsets, sigfrac, sig_mean):
    data = [np.expand_dims(generate_mixture_model(dset_length, sigfrac = sigfrac, sigdist = (sig_mean, sig_sigma_truth), 
                                                  bkgdist = (bkg_mean_truth, bkg_sigma_truth)), -1) for cur in range(num_dsets)]
    return data

def discriminant_nominal_signal(x):
    return np.square(x / bkg_sigma_truth) - np.square((x - sig_mean_prior_mu) / sig_sigma_truth)

def make_datasets_disc_nominal_signal(dset_length, num_dsets, sigfrac, sig_mean):
    return map(discriminant_nominal_signal, make_datasets_x(dset_length, num_dsets, sigfrac, sig_mean))

def discriminant_marg_signal(x):
    return np.square(x) / (bkg_sigma_truth ** 2) - np.square(x - sig_mean_prior_mu) / (sig_sigma_truth ** 2 + sig_mean_prior_sigma ** 2)

def make_datasets_disc_marg_signal(dset_length, num_dsets, sigfrac, sig_mean):
    return map(discriminant_marg_signal, make_datasets_x(dset_length, num_dsets, sigfrac, sig_mean))    

def gauss(x, mu, sigma):
    return 1.0 / (pm.math.sqrt(2 * np.pi) * sigma) * pm.math.exp(-0.5 * pm.math.sqr((x - mu) / sigma))

def logp_x(sigfrac, mu_S, data):
    logL = pm.math.sum(pm.math.log(sigfrac * gauss(x = data, mu = mu_S, sigma = sig_sigma_truth) + \
                        (1 - sigfrac) * gauss(x = data, mu = bkg_mean_truth, sigma = bkg_sigma_truth)))
    
    return logL

def logp_disc_nominal_signal(sigfrac, mu_S, x):
    sqrtexpr = sig_mean_prior_mu**2 - x * bkg_sigma_truth**2 + x * sig_sigma_truth**2
    x1 = bkg_sigma_truth * (sig_mean_prior_mu * bkg_sigma_truth + sig_sigma_truth * pm.math.sqrt(sqrtexpr)) / (bkg_sigma_truth**2 - sig_sigma_truth**2)
    x2 = bkg_sigma_truth * (sig_mean_prior_mu * bkg_sigma_truth - sig_sigma_truth * pm.math.sqrt(sqrtexpr)) / (bkg_sigma_truth**2 - sig_sigma_truth**2)
    A = 1.0 / sig_sigma_truth**2 - 1.0 / bkg_sigma_truth**2

    psig = 1.0 / (A * pm.math.abs(x2 - x1)) * (gauss(x1, mu_S, sig_sigma_truth) + gauss(x2, mu_S, sig_sigma_truth))
    pbkg = 1.0 / (A * pm.math.abs(x2 - x1)) * (gauss(x1, bkg_mean_truth, bkg_sigma_truth) + gauss(x2, bkg_mean_truth, bkg_sigma_truth))

    logL = pm.math.sum(pm.math.log(sigfrac * psig + (1 - sigfrac) * pbkg))
    return logL

def logp_disc_marg_signal(sigfrac, mu_S, x):
    sqrtexpr = (sig_sigma_truth**2 + sig_mean_prior_sigma**2) * (sig_mean_prior_mu**2 + x * (-bkg_sigma_truth**2 + sig_sigma_truth**2 + sig_mean_prior_sigma**2))
    x1 = bkg_sigma_truth * (sig_mean_prior_mu * bkg_sigma_truth + pm.math.sqrt(sqrtexpr)) / (bkg_sigma_truth**2 - sig_sigma_truth**2 - sig_mean_prior_sigma**2)
    x2 = bkg_sigma_truth * (sig_mean_prior_mu * bkg_sigma_truth - pm.math.sqrt(sqrtexpr)) / (bkg_sigma_truth**2 - sig_sigma_truth**2 - sig_mean_prior_sigma**2)
    A = 1.0 / bkg_sigma_truth**2 - 1.0 / (sig_sigma_truth**2 + sig_mean_prior_sigma**2)

    psig = 1.0 / (A * pm.math.abs(x2 - x1)) * (gauss(x1, mu_S, sig_sigma_truth) + gauss(x2, mu_S, sig_sigma_truth))
    pbkg = 1.0 / (A * pm.math.abs(x2 - x1)) * (gauss(x1, bkg_mean_truth, bkg_sigma_truth) + gauss(x2, bkg_mean_truth, bkg_sigma_truth))

    logL = pm.math.sum(pm.math.log(sigfrac * psig + (1 - sigfrac) * pbkg))
    return logL

def run_inference(inference_data, logp):
    pm_model = pm.Model()

    with pm_model:
        inference_data = pm.ConstantData("data", inference_data)

        mu_S = pm.Normal("mu_S", mu = sig_mean_prior_mu, sigma = sig_mean_prior_sigma) # prior for signal position
        sigfrac = pm.Uniform("sigfrac", lower = 0.0, upper = 1.0)

        logL = pm.Potential("logL", logp(sigfrac, mu_S, inference_data))

    with pm_model:
        step = pm.NUTS(vars = (sigfrac, mu_S))
        pm.init_nuts(init = "jitter+adapt_diag", chains = 10)
        trace = pm.sample(step = step, draws = 50000, tune = 12000, progressbar = True, chains = 2,
                            compute_convergence_checks = True, return_inferencedata = True,
                            discard_tuned_samples = True)
        trace_df = trace.to_dataframe()

    return trace_df
