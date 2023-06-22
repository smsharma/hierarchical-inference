import simulators.extended_particle_model as esim
import scipy.stats as sps

def on_off_reparam(pars, hyper_params, bins = None):
    mu, nu = pars
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
    bsigma = 3
    smean = -7
    ssigma = 2
    
    #Component Parameters
    sigdist = (smean,ssigma) 
    bkgdist = (bmean,bsigma)

    #Component Weights
    sigfrac = total_s/total

    #Overall Rate
    extended = total
    
    return extended, sigfrac, sigdist, bkgdist

def on_off_hpars(lumi = 1., s0 = 3., b0 = 40., tau = 1.0):
    s,b,tau = s0*lumi, b0*lumi, tau
    return s,b,tau