import numpy as np

from toroidal.psdamodel import ToroidalPSDA, train_ml, train_map, \
                               KappaPrior_KL, Scoring


import matplotlib.pyplot as plt
from subsphere.pca import Globe

from nlib.phonexia.embeddings import one_hot

from pyllr.quick_eval import tarnon_2_eer, tarnon_2_eer_cllr_mincllr

D = 256
#m = 2
#d = np.array([50,50,30])        
d = np.full((D//5),5)
m = len(d) // 2
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

tarc = (E*T1).sum(axis=-1)
nonc = (E*T2).sum(axis=-1)
eer = tarnon_2_eer(tarc,nonc)
print("cosine eer:\n",eer)

sc = Scoring(model0)
scs = Scoring(model0,fast=True)
scf = Scoring(model0,shortcut=True)
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

tard = left.llr(right1)
nond = left.llr(right2)
eer, cllr, mincllr = tarnon_2_eer_cllr_mincllr(tard,nond)
print("toroidal eer, Cllr, minCllr:\n",eer,cllr,mincllr," (double enroll)")

left = scs.side(E)
right1 = scs.side(T1)
right2 = scs.side(T2)

tars = left.llr(right1)
nons = left.llr(right2)
eer, cllr, mincllr = tarnon_2_eer_cllr_mincllr(tars,nons)
print("toroidal eer, Cllr, minCllr:\n",eer,cllr,mincllr)


left = scs.side(E+E2)
right1 = scs.side(T1)
right2 = scs.side(T2)

tards = left.llr(right1)
nonds = left.llr(right2)
eer, cllr, mincllr = tarnon_2_eer_cllr_mincllr(tards,nonds)
print("toroidal eer, Cllr, minCllr:\n",eer,cllr,mincllr," (double enroll)")

plt.figure()
plt.plot(tar,tard,'.')
plt.plot(non,nond,'.')
plt.xlabel('single enroll')
plt.ylabel('double enroll')
plt.title('exact scoring')
plt.grid()
plt.gca().set_aspect('equal', 'box')
plt.show()

plt.figure()
plt.plot(tars,tards,'.')
plt.plot(nons,nonds,'.')
plt.xlabel('single enroll')
plt.ylabel('double enroll')
plt.title('shortcut scoring')
plt.grid()
plt.gca().set_aspect('equal', 'box')
plt.show()

plt.figure()
plt.plot(tar,tars,'.')
plt.plot(non,nons,'.')
plt.xlabel('exact')
plt.ylabel('shortcut')
plt.title('single enroll')
plt.grid()
plt.gca().set_aspect('equal', 'box')
plt.show()

plt.figure()
plt.plot(tard,tards,'.')
plt.plot(nond,nonds,'.')
plt.xlabel('exact')
plt.ylabel('shortcut')
plt.title('double enroll')
plt.grid()
plt.gca().set_aspect('equal', 'box')
plt.show()
