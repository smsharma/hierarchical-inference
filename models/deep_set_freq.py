import torch
import numpy as np
import scipy.stats as sps

def mask_data(X, pad_value = -10):
    return (X[:,:,0] != pad_value)


class Embed(torch.nn.Module):
    def __init__(self, insize = 1, hsize = 256):
        super().__init__()
        self.insize = insize
        self.hsize = hsize
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1,self.hsize),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hsize,self.hsize),
        )

    def forward(self, X):
        return self.net(X)        

class Transformer(torch.nn.Module):
    def __init__(self, hsize = 256, out_size = 10):
        super().__init__()
        self.hsize = hsize
        self.out_size = out_size

        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.hsize, nhead=1, dim_feedforward = 64)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.project = torch.nn.Linear(self.hsize, self.out_size)

    def forward(self, X):
        X = self.transformer_encoder(X)
        X = self.project(X)
        return X


class ElementWise(torch.nn.Module):
    def __init__(self, hsize = 256, out_size = 10):
        super().__init__()
        self.hsize = hsize
        self.out_size = out_size
        self._per_elem = torch.nn.Sequential(
            torch.nn.Linear(self.hsize,self.hsize),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hsize,self.hsize),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hsize,out_size),
        )

    def forward(self, X):
        return self._per_elem(X)

class BinnedModel(torch.nn.Module):
    def __init__(self, hardscale = 1, n_elem_feats = 100, set_encoder = 'ele'):
        super().__init__()

        self.hsize = 256
        self.out_size = n_elem_feats

        self._embed_elem = Embed(1,self.hsize)
        if set_encoder == 'trf':
            self._per_elem = Transformer(self.hsize, self.out_size)
        elif set_encoder == 'ele':
            self._per_elem = ElementWise(self.hsize, self.out_size)
        
        self.onset = torch.nn.Sequential(
            torch.nn.Linear(n_elem_feats+1,self.hsize),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hsize,self.hsize),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hsize,1),
            torch.nn.Sigmoid()
        )
        
        self.softmax = torch.nn.Softmax(dim=-1)
        self.hardscale = hardscale

    def per_elem(self,X):
        elem_embeds = self._embed_elem(X)
        logits = self._per_elem(elem_embeds)
        softmaxxed = self.softmax(logits*self.hardscale)
        return softmaxxed

    def unbinned2embed(self, X):
        '''run seq2seq and pool with mask'''
        m = mask_data(X)
        softmaxxed = self.per_elem(X)
        softmaxxed = softmaxxed*m[:,:,None]
        counts = (softmaxxed).sum(dim=1)
        return counts

    def test_embed(self, embed, poi):
        poi_col = poi*torch.ones(embed.shape[0])[:,None]
        Xpar = torch.cat([embed,poi_col],dim=-1)
        p = self.onset(Xpar)
        return p


    def forward(self,X,poi):
        embed = self.unbinned2embed(X)
        test_stat = self.test_embed(embed, poi)
        return test_stat


#### Utils

def pad_data(data, Nbatch, pad_value = -10):
    lens = [len(d) for d in data]
    padded = torch.ones((Nbatch,max(lens)))*pad_value
    for i,d in enumerate(data):
        padded[i][:len(d)][:] = torch.FloatTensor(d)
    X =  padded.reshape(Nbatch,-1,1)
    return X

def transform(x_sequence, binned = True):
    if binned:
        bin = x_sequence.argmax(axis=-1)
        return 2*(bin-0.5)
    else:
        return x_sequence

def _preprocess_ragged_data(data,binned):
    Nbatch = len(data)
    X = pad_data([transform(d[0], binned = binned) for d in data], Nbatch)
    S = pad_data([transform(d[1][0], binned = binned) for d in data], Nbatch)
    B = pad_data([transform(d[1][1], binned = binned) for d in data], Nbatch)
    return X,S,B

def generate(simulator,pars, Nbatch, pad_value = -10, bins = None):
    data = [simulator(pars, bins = bins) for i in range(Nbatch)]
    binned = bins is not None
    return _preprocess_ragged_data(data,binned)

def generate_for_wald(simulator, poi,N=100, bins = None):
    #generate a "surface" by
    #1. generating a random equidistant surface
    #2. generating a random nuisances for left/right/central

    delta = np.random.uniform(0.5,2.0)
    n1,n2,n3 = np.random.uniform(0.5,2.0, size = (3,))

    Xnull,_,_ = generate(simulator,[poi,n1], N, bins = bins)
    Xa,_,_ = generate(simulator,[poi+delta,n2],N//2, bins = bins)
    Xb,_,_ = generate(simulator,[poi-delta,n3],N//2, bins = bins)

    ynull = torch.zeros(N).reshape(-1,1)
    ya = torch.ones(N//2).reshape(-1,1)
    yb = torch.ones(N//2).reshape(-1,1)
    return (Xnull,Xa,Xb),(ynull,ya,yb),poi

