"""Baseline model definitions used in the four-model comparison."""
import math
import numpy as np
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_features=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features,64), nn.BatchNorm1d(64), nn.SiLU(), nn.Dropout(.20),
            nn.Linear(64,32), nn.SiLU(), nn.Dropout(.10), nn.Linear(32,1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

class GNNWLR(nn.Module):
    """Spatially varying-coefficient neural logistic regression."""
    def __init__(self, n_features=9):
        super().__init__()
        self.swnn = nn.Sequential(nn.Linear(2,96), nn.PReLU(), nn.Dropout(.15),
                                  nn.Linear(96,96), nn.PReLU(),
                                  nn.Linear(96,n_features+1))
        self.global_beta = nn.Parameter(torch.zeros(n_features+1))
    def forward(self, x, coordinates):
        beta = self.global_beta + self.swnn(coordinates)
        return beta[:,0] + (beta[:,1:] * x).sum(1)

def _local_logistic(x, y, weights, ridge=1e-3, steps=30):
    xa=np.column_stack([np.ones(len(x)),x]); beta=np.zeros(xa.shape[1])
    for _ in range(steps):
        eta=np.clip(xa@beta,-25,25); p=1/(1+np.exp(-eta)); v=np.maximum(p*(1-p),1e-6)
        h=xa.T@((weights*v)[:,None]*xa); h.flat[::h.shape[0]+1]+=ridge
        grad=xa.T@(weights*(y-p))-ridge*beta
        try: delta=np.linalg.solve(h,grad)
        except np.linalg.LinAlgError: delta=np.linalg.pinv(h)@grad
        beta+=delta
        if np.max(np.abs(delta))<1e-6: break
    return beta

def gwr_predict(x_train, xy_train, y_train, x_test, xy_test, neighbors=220):
    """Adaptive-bisquare geographically weighted logistic regression."""
    out=np.empty(len(x_test)); k=min(neighbors,len(x_train)-1)
    for j,c in enumerate(xy_test):
        d=np.sqrt(((xy_train-c)**2).sum(1)); bw=np.partition(d,k)[k]+1e-8
        w=np.where(d<bw,(1-(d/bw)**2)**2,0.); beta=_local_logistic(x_train,y_train,w)
        eta=float(np.clip(np.r_[1.,x_test[j]]@beta,-25,25)); out[j]=1/(1+math.exp(-eta))
    return out
