import numpy as np

from toroidal.psdamodel import ToroidalPSDA, train_ml, train_map, \
                               KappaPrior_KL, Scoring


import matplotlib.pyplot as plt
from subsphere.pca import Globe

from nlib.phonexia.embeddings import one_hot

from pyllr.quick_eval import tarnon_2_eer, tarnon_2_eer_cllr_mincllr

D = 3
m, n = 1,2
d = np.array([2,1])        
#gamma_z = np.zeros(m)
gamma_z = np.full(m,5.0)
#gamma_y = np.zeros(n-m)
gamma_y = np.full(n-m,1)
kappa = 4000
snr = 100
w0 = np.array([np.sqrt(snr),1])
gamma = np.hstack([gamma_z,gamma_y])
model0 = ToroidalPSDA.random(D, d, m, w0, kappa, gamma)


t = 1000
labels = np.arange(t)
Z1 = model0.sample_speakers(t)
Z2 = model0.sample_speakers(t)
E = model0.sample_data(Z1,labels,model0.sample_channels(t))
T1 = model0.sample_data(Z1,labels,model0.sample_channels(t))
T2 = model0.sample_data(Z2,labels,model0.sample_channels(t))

tar = (E*T1).sum(axis=-1)
non = (E*T2).sum(axis=-1)
eer = tarnon_2_eer(tar,non)
print("cosine eer:\n",eer)

sc = Scoring(model0)
left = sc.side(E)
right1 = sc.side(T1)
right2 = sc.side(T2)

tar = left.llr(right1).ravel()
non = left.llr(right2).ravel()
eer, cllr, mincllr = tarnon_2_eer_cllr_mincllr(tar,non)
print("toroidal eer, Cllr, minCllr:\n",eer,cllr,mincllr)


