# HEP like simulators

import numpy as np
import scipy.stats as sps

def compute_from_generic(pars):
    '''
    compute some useful derived quantities based on a set of generic parameters
    '''
    
    extended, sigfrac, sigdist, bkgdist = pars
    bkgfrac = 1-sigfrac
    return extended, sigfrac, bkgfrac, sigdist, bkgdist


def digitize(samples, bins):
    return np.eye(len(bins)-1)[np.digitize(samples,bins)-1]

def generate_generic(pars, bins = None):
    '''
    pars: extended, sigfrac, (sig mean, sig std), (bkg mean, bkg std)
    '''
    extended, sigfrac, bkgfrac, sigdist, bkgdist = compute_from_generic(pars)
    # extended part... 
    N = np.random.poisson(extended)
    return generate_mixture_model(N, sigfrac, sigdist, bkgdist)
    
def generate_mixture_model(nevts, sigfrac, sigdist, bkgdist, bins = None):
    n_signal = np.random.binomial(nevts,sigfrac)
    n_backgr = nevts-n_signal
    sig_samples = np.random.normal(*sigdist,size = n_signal)
    bkg_samples = np.random.normal(*bkgdist,size = n_backgr)

    samples = np.concatenate([sig_samples,bkg_samples])
    np.random.shuffle(samples)

    if bins is not None:
        return digitize(samples,bins)
    return samples

def get_of_off_simulator(hp):
    def generate_on_off(pars, bins = None):
        mu,nu = pars
        generic_pars = on_off_reparam(mu,nu, hyper_params = hp)
        return generate_generic(generic_pars)
    return generate_on_off

def on_off_hpars(lumi = 1., s0 = 15., b0 = 70., tau = 1.0):
    s,b,tau = s0*lumi, b0*lumi, tau
    return s,b,tau

def on_off_reparam(mu,nu,hyper_params, bins = None):
    s,b,tau = hyper_params
    cut = sps.norm.ppf(1/(1+tau))

    
    b1 = nu*b
    b2 = nu*tau*b
    
    s1 = mu*s
    s2 = 0

    total_s = s1+s2
    total_b = b1+b2
    total = total_s + total_b

    bmean = 0-cut
    bsigma = 1
    smean = -5
    ssigma = 1
    
    
    #Component Parameters
    sigdist = (smean,ssigma) 
    bkgdist = (bmean,bsigma)

    
    #Component Weights
    sigfrac = total_s/total
    bkgfrac = total_b/total

    #Overall Rate
    extended = total
    
        
    return extended, sigfrac, sigdist, bkgdist

def plot_densities(axs,pars):
    extended, sigfrac, bkgfrac, sigdist, bkgdist = compute_from_generic(pars)
    xi = np.linspace(-10,10,1001)
    yi_sig = sps.norm(*sigdist).pdf(xi)
    yi_bkg = sps.norm(*bkgdist).pdf(xi)
    ax = axs[0]
    ax.plot(xi,yi_sig, label = 'signal')
    ax.plot(xi,yi_bkg, label = 'background')

    yi_overall = sigfrac * sps.norm(*sigdist).pdf(xi) + bkgfrac * sps.norm(*bkgdist).pdf(xi)
    ax.plot(xi,yi_overall, label = 'mixture')
    ax.vlines(0,0,0.5, colors = 'k', linestyle = 'dashed')
    ax.legend()

    ax = axs[1]
    counts = np.arange(20,150)
    probs = sps.poisson(extended).pmf(counts)
    ax.bar(counts,probs, width = 1)
    ax.set_title(fr'$\lambda$: {extended:.2f}')

    ax = axs[2]
    yi_overall = extended*(sigfrac * sps.norm(*sigdist).pdf(xi) + bkgfrac * sps.norm(*bkgdist).pdf(xi))
    ax.plot(xi,yi_overall, label = 'mixture * extended')
    ax.set_ylim(0,30)
    ax.vlines(0,0,30, colors = 'k', linestyle = 'dashed')
