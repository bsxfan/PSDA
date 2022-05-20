import numpy as np

from toroidal.psdamodel import ToroidalPSDA, train_ml, train_map, \
                               KappaPrior_KL, Scoring


import matplotlib.pyplot as plt
from subsphere.pca import Globe

from nlib.phonexia.embeddings import one_hot

from pyllr.quick_eval import tarnon_2_eer, tarnon_2_eer_cllr_mincllr




def umodel( D, d, m, w=None, kappa=None):
    n = len(d)
    assert d.sum() <= D
    assert 1 <= m <= n
    
    gamma = np.zeros(n)
    model =  ToroidalPSDA.random(D, d, m, w, kappa, gamma)
    return model

def sample_tr(model:ToroidalPSDA, ns=100, nr=10):
    print('sampling training data')
    labels = np.arange(ns).repeat(nr)
    X, Y, Z, Mu, labels = model.sample(labels)
    return X, labels

def sample_te(model:ToroidalPSDA, ns=100, ne=1, nt=10):
    print('sampling test data')
    elabels = np.arange(ns).repeat(ne)
    E, Y, Z, Mu, elabels = model.sample(elabels)
    tlabels = np.arange(ns).repeat(nt)
    t = len(tlabels)
    T = model.sample_data(Z,tlabels,model.sample_channels(t))
    return E, elabels, T, tlabels

def utrain(X,labels,d,m,niters=50,quiet=True):
    print('training model')
    model, obj = train_ml(d,m,niters,X,labels,uniformz=True,uniformy=True,
                          quiet=quiet)
    return model, obj


def marg_llh(model:ToroidalPSDA, X, labels):
    return model.margloglh(X,labels=labels)


def test(model:ToroidalPSDA, E, elabels, T, tlabels):
    sc = Scoring(model)
    left = sc.side(E)
    right = sc.side(T)
    llr = left.llrMatrix(right)
    tar = elabels.reshape(-1,1) == tlabels
    non = np.logical_not(tar)
    eer, cllr, mincllr = tarnon_2_eer_cllr_mincllr(llr[tar],llr[non])
    return eer, cllr, mincllr
    

def train_test(m, d, Xtr, labels_tr, E, elabels, T, tlabels):
    model, obj = utrain(Xtr,labels_tr,d,m)
    eer, cllr, mincllr = test(model, E, elabels, T, tlabels)
    marg = marg_llh(model,T,tlabels)
    print(f"w={model.E.w}, kappa={model.kappa}")
    print('eer, Cllr, minCllr:',eer,cllr,mincllr)
    print('marg_llh:',marg)    
    
    
    
D = 256
m = 3
d = np.array([20,20,20,50])
w = np.array([2,4,8,16])
model0 = umodel(D,d,m,w,kappa=200)    
E, elabels, T, tlabels = sample_te(model0,100,1,10)    
eer, cllr, mincllr = test(model0, E, elabels, T, tlabels)
marg = marg_llh(model0,T,tlabels)
print('oracle test:')
print('eer, Cllr, minCllr:',eer,cllr,mincllr)
print('marg_llh:',marg)    

print('\n')
Xtr, labels_tr = sample_tr(model0,5000,10)    

print('\ntraining with oracle structure (m=5)')
train_test(m, d, Xtr, labels_tr, E, elabels, T, tlabels)

m = 1
d = np.array([60,50])
print('\ntraining with m=1')
train_test(m, d, Xtr, labels_tr, E, elabels, T, tlabels)

m = 6
d = np.array([10,10,10,10,10,10,50])
print('\ntraining with m=6')
train_test(m, d, Xtr, labels_tr, E, elabels, T, tlabels)


# m = 4
# d = np.array([25,25,25,25,25,25])
# print('\ntraining with m=4')
# train_test(m, d, Xtr, labels_tr, E, elabels, T, tlabels)


# m = 1
# d = np.array([100])
# print('\ntraining with m=1, no channels')
# train_test(m, d, Xtr, labels_tr, E, elabels, T, tlabels)

