import numpy as np
from scipy.special import ive, iv, gammaln, gamma
from besseli import LogBesselI, k_and_logk


# def logfactorial(x):
#     """
#     Natural log of factorial. Invokes scipy.special.gammaln.
#     log x! = log gamma(x+1)
#     """
#     return gammaln(x+1)

sqrtlog2pi = np.log(2*np.pi)/2
log2 = np.log(2)



def logBesselI(nu, k=None, logk=None):
    k, logk = k_and_logk(k, logk)
    if nu < 0: return np.log(ive(nu,k)) + k
    return LogBesselI(nu)(k,logk)
    

def dot(x,mu):
    if np.isscalar(mu) or np.isscalar(x):
        return mu*x
    return x@mu
    
def logvmf_wikipedia(x,mu,k=None, logk=None):
    k, logk = k_and_logk(k, logk)
    dim = 0 if np.isscalar(mu) else len(mu)
    nu = dim/2-1
    return nu*logk + k*dot(x,mu) - dim*sqrtlog2pi - logBesselI(nu,k,logk)

def logvmf_mardiajupp(x,mu,k=None, logk=None):
    k, logk = k_and_logk(k, logk)
    dim = 0 if np.isscalar(mu) else len(mu)
    nu = dim/2-1
    return nu*(logk-log2) + k*dot(x,mu) - gammaln(dim/2) - logBesselI(nu,k,logk)

def vmf(x,mu,k):
    dim = 0 if np.isscalar(mu) else len(mu)
    nu = dim/2-1
    return np.exp(k*dot(x,mu)) * np.power(k,nu) / iv(nu,k)





