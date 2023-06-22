
import torch
import numpy as np
from models.deep_set_freq import _preprocess_ragged_data, mask_data
import matplotlib.pyplot as plt
import simulators.on_off_sim as oosim
import scipy.stats as sps

def plot(simulator, trained_model, poi, hp, lrt_func, bins = None, Nbatch = 100):
    trained_model.eval()
    poi_scan = np.linspace(2,10,101)
    
    nuis = np.random.uniform(0.5,1.5)

    data = [simulator([poi,nuis], bins = bins) for i in range(Nbatch)]
    binned = bins is not None
    X,S,B = _preprocess_ragged_data(data,binned)

    sX,sS,sB = X[0],S[0],B[0]
    sX = torch.FloatTensor(sX.reshape(1,-1,1))
    sS = torch.FloatTensor(sS.reshape(1,-1,1))
    sB = torch.FloatTensor(sB.reshape(1,-1,1))

    m = mask_data(X)
    f,axarr = plt.subplots(1,5)
    f.set_facecolor('w')
    f.set_size_inches(20,4)

    scans = torch.cat([trained_model(X,p).detach() for p in poi_scan],dim=-1)
    
    ax = axarr[0]
    ax.plot(poi_scan,scans.T, c = 'k', alpha = 20/Nbatch);
    ax.set_ylim(0,1.2)

    ax = axarr[1]

    mle_pois = poi_scan[scans.argmin(dim=-1)]
    ax.hist(mle_pois, bins = np.linspace(2,10,31), density=True)
    ax.vlines(poi,0,1, colors = 'k')
    ax.set_ylim(0,1.2)

    ax = axarr[2]
    bins = np.linspace(-10,10,31)
    ax.hist(sX[:,:,0][mask_data(sX)], bins = bins, facecolor = 'k', alpha = 0.2, edgecolor = 'k');
    ax.hist(sS[:,:,0][mask_data(sS)], bins = bins, histtype = 'step', edgecolor = 'r');
    ax.hist(sB[:,:,0][mask_data(sB)], bins = bins, histtype = 'step', edgecolor = 'b', linestyle = 'dashed');

    xi = np.linspace(-10,10,1001)
    pp = oosim.on_off_reparam([poi, nuis], hp)
    bw = np.diff(bins)[0]

    _sigpdf = pp[1]*sps.norm(*pp[-2]).pdf(xi)
    _bkgpdf = (1-pp[1])*sps.norm(*pp[-1]).pdf(xi)
    ax.plot(xi,pp[0]*bw * _sigpdf)
    ax.plot(xi,pp[0]*bw * _bkgpdf)
    ax.plot(xi, pp[0]*bw * (_sigpdf + _bkgpdf))


    ax.set_ylim(0,60)
    ax.set_title(f'poi: {poi:.2f}')

    ax = axarr[3]

    _true = np.array([lrt_func(_d[0],poi)[-1] for _d in data])
    _ml = trained_model(_preprocess_ragged_data(data,False)[0],poi)[:,0].detach().numpy()
    ax.scatter(_ml,_true, alpha = 0.2, c = mle_pois - poi, cmap = 'coolwarm')
    ax.set_ylim(0,10)
    ax.set_xlim(_ml.min()-0.1, _ml.max() + 0.1)

    ax = axarr[4]
    xi = np.linspace(0,15)
    pi = sps.chi2(df = 1).pdf(xi)
    ax.hist(_true, bins = xi, density=True)
    ax.plot(xi,pi)

    f.set_tight_layout(True)
