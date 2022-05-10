import numpy as np

from psda.vmf_map import logkappa_asymptote_intersection, gvmf,\
                         logNormConst
                         
from psda.vmf_sampler import sample_uniform                         

def stats(dim,factor,n=1000):
    mu = sample_uniform(dim).ravel()
    kappa = factor#*np.exp(logkappa_asymptote_intersection(dim))
    vmf = gvmf(logNormConst(dim),mu,kappa)
    X = vmf.sample(n)
    
    mean = X.mean(axis=0)
    smean = (n*(mean@mean)-1)/(n-1)
    
    meansq = (((X.T@X)**2).sum()-n)/(n*(n-1))
    var = meansq-smean**2
    return smean, var*dim


import matplotlib.pyplot as plt

#dim = [2,4,8,16,32,64,128,256,512]
dim = [32,64,128,256,512]
f = 100
logfac = np.linspace(-np.log(f),np.log(100*f),50)
fac = np.exp(logfac)

plt.figure()
for d in dim:
    print(f"computing dim={d}")
    S = np.array([np.array(stats(d,f)) for f in fac])
    plt.plot(logfac,S[:,0],label=f'mu({d})')
    plt.plot(logfac,S[:,1],'--',label=f'd*var({d})')
    logk0 = logkappa_asymptote_intersection(d)
    plt.plot(logk0,0.1,'*')
plt.legend()
plt.grid()
plt.xlabel("log kappa")
plt.title("cosine score stats for VMF(dim,kappa)")
plt.show()