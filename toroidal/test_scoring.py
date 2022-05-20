import numpy as np

from toroidal.psdamodel import ToroidalPSDA, train_ml, train_map, \
                               KappaPrior_KL, Scoring


import matplotlib.pyplot as plt
from subsphere.pca import Globe

from nlib.phonexia.embeddings import one_hot

from pyllr.quick_eval import tarnon_2_eer, tarnon_2_eer_cllr_mincllr

D = 256
m = 2
d = np.array([50,50,30])        
#d = np.full((128),2)
#m = len(d) // 2
n = len(d)
assert d.sum() <= D
assert 1 <= m <= n
uniform_z, uniform_y = True, True
kappa = 150
snr = 1

gamma_z = np.zeros(m) if uniform_z else np.full(m,5.0)
gamma_y = np.zeros(n-m) if uniform_y else np.full(n-m,1)
gamma = np.hstack([gamma_z,gamma_y])
w0 = np.ones(n)
w0[:m] = np.sqrt(snr)
model0 = ToroidalPSDA.random(D, d, m, w0, kappa, gamma)


t = 1000
labels = np.arange(t)
Z1 = model0.sample_speakers(t)
Z2 = model0.sample_speakers(t)
E = model0.sample_data(Z1,labels,model0.sample_channels(t))
E2 = model0.sample_data(Z1,labels,model0.sample_channels(t))
T1 = model0.sample_data(Z1,labels,model0.sample_channels(t))
T2 = model0.sample_data(Z2,labels,model0.sample_channels(t))

tar = (E*T1).sum(axis=-1)
non = (E*T2).sum(axis=-1)
eer = tarnon_2_eer(tar,non)
print("cosine eer:\n",eer)

sc = Scoring(model0)
scf = Scoring(model0,fast=True)
scs = Scoring(model0,shortcut=True)
left = sc.side(E)
right1 = sc.side(T1)
right2 = sc.side(T2)

tar = left.llr(right1)
non = left.llr(right2)
eer, cllr, mincllr = tarnon_2_eer_cllr_mincllr(tar,non)
print("toroidal eer, Cllr, minCllr:\n",eer,cllr,mincllr)


left = sc.side(E+E2)
right1 = sc.side(T1)
right2 = sc.side(T2)

tar = left.llr(right1)
non = left.llr(right2)
eer, cllr, mincllr = tarnon_2_eer_cllr_mincllr(tar,non)
print("toroidal eer, Cllr, minCllr:\n",eer,cllr,mincllr," (double enroll)")




print('\nTesting matrix scoring:')
ns = 100
Z = model0.sample_speakers(ns)
E = model0.sample_data(Z,np.arange(ns),model0.sample_channels(ns))
E2 = model0.sample_data(Z,np.arange(ns),model0.sample_channels(ns))
labels = np.arange(ns).repeat(100)
t = len(labels)
T = model0.sample_data(Z,labels,model0.sample_channels(t))

left = sc.side(E)
right = sc.side(T)
llr = left.llrMatrix(right)
tar = np.arange(ns).reshape(-1,1) == labels
non = np.logical_not(tar)
eer, cllr, mincllr = tarnon_2_eer_cllr_mincllr(llr[tar],llr[non])
print("toroidal eer, Cllr, minCllr:\n",eer,cllr,mincllr)

left = sc.side(E+E2)
right = sc.side(T)
llr = left.llrMatrix(right)
tar = np.arange(ns).reshape(-1,1) == labels
non = np.logical_not(tar)
eer, cllr, mincllr = tarnon_2_eer_cllr_mincllr(llr[tar],llr[non])
print("toroidal eer, Cllr, minCllr:\n",eer,cllr,mincllr," (double enroll)")




print('\nfast:')
leftf = scf.side(E)
rightf = scf.side(T)
llr = leftf.llrMatrix(rightf)
tar = np.arange(ns).reshape(-1,1) == labels
non = np.logical_not(tar)
eer, cllr, mincllr = tarnon_2_eer_cllr_mincllr(llr[tar],llr[non])
print("toroidal eer, Cllr, minCllr:\n",eer,cllr,mincllr)

print('\nshortcut:')
lefts = scs.side(E)
rights = scs.side(T)
llr = lefts.llrMatrix(rights)
tar = np.arange(ns).reshape(-1,1) == labels
non = np.logical_not(tar)
eer, cllr, mincllr = tarnon_2_eer_cllr_mincllr(llr[tar],llr[non])
print("toroidal eer, Cllr, minCllr:\n",eer,cllr,mincllr)

lefts = scs.side(E+E2)
rights = scs.side(T)
llr = lefts.llrMatrix(rights)
tar = np.arange(ns).reshape(-1,1) == labels
non = np.logical_not(tar)
eer, cllr, mincllr = tarnon_2_eer_cllr_mincllr(llr[tar],llr[non])
print("toroidal eer, Cllr, minCllr:\n",eer,cllr,mincllr, "(double enroll)")



