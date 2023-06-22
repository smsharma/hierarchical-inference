# HEP like simulators

import numpy as np
import scipy.stats as sps
import scipy.optimize as sopt

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

    n_signal = np.random.binomial(N,sigfrac)
    n_backgr = N-n_signal
    sig_samples = np.random.normal(*sigdist,size = n_signal)
    bkg_samples = np.random.normal(*bkgdist,size = n_backgr)

    samples = np.concatenate([sig_samples,bkg_samples])
    np.random.shuffle(samples)

    if bins is not None:
        samples =  digitize(samples,bins)
        sig_samples =  digitize(sig_samples,bins)
        bkg_samples =  digitize(bkg_samples,bins)
    return samples, (sig_samples, bkg_samples)
    

def get_reparam_simulator(reparam, hp):
    def generate_on_off(pars, bins = None):
        generic_pars = reparam(pars, hyper_params = hp)
        return generate_generic(generic_pars, bins = bins)
    return generate_on_off

def get_reparam_lrtfunc(reparam, hp):
    logprob_fn = logprob_unbinned
    reparam_lhood = lambda d,p: logprob_fn(d,reparam(p,hyper_params = hp))

    def lrt_func(data, poi):
        return get_lrt(data,poi,reparam_lhood)

    return lrt_func

def logprob_unbinned(data,pars):
    '''
    compute the lob probability of data in the extended particle model
    under the given parameters
    '''
    extended, sigfrac, bkgfrac, sigdist, bkgdist = compute_from_generic(pars)
    
    N_obs = len(data)
    log_extended = sps.poisson(extended).logpmf(N_obs)
    
    probs_s = sps.norm(*sigdist).pdf(data)
    probs_b = sps.norm(*bkgdist).pdf(data)
    log_probs_mixture = np.log((sigfrac*probs_s + bkgfrac*probs_b))
    log_samples = log_probs_mixture.sum()
    log_prob = log_samples + log_extended
    return log_prob

def get_conditional_mle(data,poi, logprob_fn):
    '''
    find the best fit nuisance parameters for fixed POI
    '''
    mle = sopt.minimize(lambda nuis: -logprob_fn(data,np.asarray([poi,nuis[0]])),np.array([1.5]))
    return np.asarray([poi,mle.x[0],mle.fun])

def get_global_mle(data, logprob_fn):
    '''
    find the global best fit parameters
    '''
    mle = sopt.minimize(lambda pars: -logprob_fn(data,pars),np.array([3,1.5]))
    return np.asarray([mle.x[0],mle.x[1],mle.fun])

def get_lrt(data,poi,logprob_fn):
    '''
    compute the profile likelihood ratio
    '''
    den = get_global_mle(data,logprob_fn)
    num = get_conditional_mle(data,poi,logprob_fn)
    val = 2*(num[-1]-den[-1])
    return np.asarray([num[0],num[1],val])

def plot_densities(axs,pars):
    extended, sigfrac, bkgfrac, sigdist, bkgdist = compute_from_generic(pars)
    xi = np.linspace(-10,10,1001)
    yi_sig = sps.norm(*sigdist).pdf(xi)
    yi_bkg = sps.norm(*bkgdist).pdf(xi)
    ax = axs[0]
    ax.plot(xi,yi_sig, label = 'signal')
    ax.plot(xi,yi_bkg, label = 'background')
    ax.set_title('per-event pdf')

    yi_overall = sigfrac * sps.norm(*sigdist).pdf(xi) + bkgfrac * sps.norm(*bkgdist).pdf(xi)
    ax.plot(xi,yi_overall, label = 'mixture')
    ax.vlines(0,0,0.5, colors = 'k', linestyle = 'dashed')
    ax.legend()

    ax = axs[1]
    counts = np.arange(0,150)
    probs = sps.poisson(extended).pmf(counts)
    ax.bar(counts,probs, width = 1)
    ax.set_title(fr'cardinality $\lambda$: {extended:.2f}')

    ax = axs[2]
    yi_overall = extended*(sigfrac * sps.norm(*sigdist).pdf(xi) + bkgfrac * sps.norm(*bkgdist).pdf(xi))
    ax.plot(xi,yi_overall, label = 'mixture * extended')
    ax.set_ylim(0,30)
    ax.vlines(0,0,30, colors = 'k', linestyle = 'dashed')
    ax.set_title('count density')