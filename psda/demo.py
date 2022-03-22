import numpy as np
from numpy.random import randn, randint
import matplotlib.pyplot as plt

from psda import VMF, PSDA, decompose, atleast2
from pyllr import quick_eval

"""
This demo uses a quick-and-dirty data simulator, using Gaussians, not VMF.
It does not work for high dimensions. But you can play with dim = 2 or 3
if you like.
"""
dim = 20
b, w = 10, 50  # within, between concentrations

ns = 100  # number of training speakers
n = 1000  # numer of training examples


# set up model to sample from
norm, mu = decompose(randn(dim))
model0 = PSDA(w, VMF(mu, b))

Z = model0.sample_speakers(ns)
labels = randint(ns,size=(n,))
uu, labels, counts = np.unique(labels, return_inverse=True, return_counts=True)

# sample training data
Xtrain = model0.sample(Z, labels)

if dim == 2:
    plt.figure()
    plt.scatter(Xtrain[:,0],Xtrain[:,1])
    plt.axis('square')
    plt.xlim(-1.2,1.2)
    plt.ylim(-1.2,1.2)
    plt.grid()
    plt.title('Embeddings')
    plt.show()


# one hot label matrix
L = np.full((n,len(counts)),False)   # (n, ns)
L[np.arange(n),labels] = True

# these are the 1st-order stats required by the em traning
means = (L.T @ Xtrain) / counts.reshape(-1,1)

# filter out singleton speakers
means, counts = atleast2(means, counts)

# train the model!
model, obj = PSDA.em(means, counts, niters=10)

plt.figure()
plt.plot(obj,'-*')
plt.grid()
plt.title('PSDA EM algorithm')
plt.xlabel('iteration')
plt.ylabel('marginal likelihood')
plt.show()

# generate some test data
nt = 10000
Z1 = model0.sample_speakers(nt)
Z2 = model0.sample_speakers(nt)
Enroll = model0.sample(Z1, np.arange(nt))  # enrollment embeddings
Test1 = model0.sample(Z1, np.arange(nt))   # target test embeddings
Test2 = model0.sample(Z2, np.arange(nt))   # nnotar test embeddings

# compute PSDA scores
E = model.prep(Enroll)
T1 = model.prep(Test1)
T2 = model.prep(Test2)

tar = E.llr_vector(T1)
non = E.llr_vector(T2)

# compute cosine scores
tarc = (Enroll*Test1).sum(axis=-1)
nonc = (Enroll*Test2).sum(axis=-1)


plt.figure()
plt.plot(non,nonc,'.',label='non')
plt.plot(tar,tarc,'.',label='tar')
plt.grid()
plt.xlabel('PSDA score')
plt.ylabel('cosine score')
plt.legend()
plt.show()



# compute double-enroll PSDA scores
Enroll2 = model0.sample(Z1, np.arange(nt))  # 2nd enrollment embeddings
E2 = model.prep(Enroll + Enroll2)
tar2 = E2.llr_vector(T1)
non2 = E2.llr_vector(T2)

# compute double-enroll cosine scores
E2c = decompose(Enroll + Enroll2)[1]
tar2c = (E2c*Test1).sum(axis=-1)
non2c = (E2c*Test2).sum(axis=-1)


tar12 = np.hstack([tar,tar2])
non12 = np.hstack([non,non2])

tar12c = np.hstack([tarc,tar2c])
non12c = np.hstack([nonc,non2c])


eer_p, cllr_p, mincllr_p = quick_eval.tarnon_2_eer_cllr_mincllr(tar, non)
eer_p2, cllr_p2, mincllr_p2 = quick_eval.tarnon_2_eer_cllr_mincllr(tar2, non2)

eer_c, cllr_c, mincllr_c = quick_eval.tarnon_2_eer_cllr_mincllr(tarc, nonc)
eer_c2, cllr_c2, mincllr_c2 = quick_eval.tarnon_2_eer_cllr_mincllr(tar2c, non2c)

eer_p12, cllr_p12, mincllr_p12 = quick_eval.tarnon_2_eer_cllr_mincllr(tar12, non12)
eer_c12, cllr_c12, mincllr_c12 = quick_eval.tarnon_2_eer_cllr_mincllr(tar12c, non12c)


print("\n\nCosine scoring, single enroll:")
print(f"  EER:     {eer_c*100:.1f}%")
print(f"  Cllr:    {cllr_c:.3f}")
print(f"  minCllr: {mincllr_c:.3f}")

print("\nPSDA scoring, single enroll:")
print(f"  EER:     {eer_p*100:.1f}%")
print(f"  Cllr:    {cllr_p:.3f}")
print(f"  minCllr: {mincllr_p:.3f}")

print("\nCosine scoring, double enroll:")
print(f"  EER:     {eer_c2*100:.1f}%")
print(f"  Cllr:    {cllr_c2:.3f}")
print(f"  minCllr: {mincllr_c2:.3f}")

print("\nPSDA scoring, double enroll:")
print(f"  EER:     {eer_p2*100:.1f}%")
print(f"  Cllr:    {cllr_p2:.3f}")
print(f"  minCllr: {mincllr_p2:.3f}")

print("\nCosine scoring, mixed enroll:")
print(f"  EER:     {eer_c12*100:.1f}%")
print(f"  Cllr:    {cllr_c12:.3f}")
print(f"  minCllr: {mincllr_c12:.3f}")

print("\nPSDA scoring, mixed enroll:")
print(f"  EER:     {eer_p12*100:.1f}%")
print(f"  Cllr:    {cllr_p12:.3f}")
print(f"  minCllr: {mincllr_p12:.3f}")
