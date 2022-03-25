import numpy as np
from numpy.random import randn, randint
import matplotlib.pyplot as plt

from psda.psda import VMF, PSDA, decompose, atleast2
from pyllr import quick_eval
from psda.vmf_sampler import sample_uniform

dim = 100
w0, uniform = 100, VMF.uniform(dim)  # within, between 
# mu = sample_uniform(dim)
# w0, uniform = 100, VMF(mu,1.0)  # within, between 

model0 = PSDA(w0, uniform)


# generate some test data 
ns, nt = 500, 10000
print(f'sampling {ns*2} test speakers')
Z1 = model0.sample_speakers(ns)
Z2 = model0.sample_speakers(ns)
labels1 = randint(ns,size=(nt,))
labels2 = randint(ns,size=(nt,))

print(f'sampling {nt*3} test data')
Enroll = model0.sample(Z1, labels1)  # enrollment embeddings
Test1 = model0.sample(Z1, labels1)   # target test embeddings 
Test2 = model0.sample(Z2, labels2)   # nnotar test embeddings


nw = 200
cllr = np.empty(nw)
mincllr = np.empty(nw)
print(f'scroring {nw} models')
ww = np.exp(np.linspace(np.log(w0/2),np.log(w0*2),nw))
for i, w in enumerate(ww):
    
    model = PSDA(w,uniform)
    
    # compute PSDA scores
    E = model.prep(Enroll)
    T1 = model.prep(Test1)
    T2 = model.prep(Test2)
    
    tar = E.llr_vector(T1)
    non = E.llr_vector(T2)


    eer, cllr[i], mincllr[i] = quick_eval.tarnon_2_eer_cllr_mincllr(tar, non)
    print(f"{i}: w = {w:.2f}, Cllr = {cllr[i]:.2f}")

plt.figure()
plt.semilogx(ww,cllr,label='Cllr')
plt.semilogx(ww,mincllr,label='minCllr')
plt.title(f'w={w0}')
plt.xlabel('w')
plt.ylabel('Cllr')
plt.grid()
plt.legend()
plt.show()

