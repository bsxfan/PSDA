import numpy as np

from psda.vmf_onedim import logNormConst, gvmf
from psda.vmf import decompose
from psda.besseli import k_and_logk

from scipy.optimize import minimize_scalar
from scipy.special import psi, gammaln



def logkappa_asymptote_intersection(dim):
    assert dim >= 1
    nu = dim/2-1
    return ( nu*np.log(2) + gammaln(nu+1) - np.log(2*np.pi) ) / (nu + 0.5)     



class GammaPrior:
    def __init__(self, mean, beta):
        self.beta = beta                #  = mean/var
        self.alpha = mean*beta
        self.rep = self.mean()          # a representative
        
    def mean(self):
        return self.alpha / self.beta
    
    def meanlog(self):
        return psi(self.alpha) - np.log(self.beta)

    def var(self):
        return self.mean()/self.beta
    
    
    def mode(self):
        alpha, beta = self.alpha, self.beta
        return 0.0 if alpha<1 else (alpha-1)/beta
    
    def loglh(self,k=None, logk=None):
        """
        unnormalized logpdf
        """
        k, logk = k_and_logk(k, logk)
        return (self.alpha-1)*logk - self.beta*k
    
    
    @classmethod
    def assign(cls,dim,meanfactor,beta):
        """
        When dim >=2. the gamma prior mean for kappa is set at meanfactor*k0, 
        where k0 depends on dim and is neutral in the sense of being neither 
        concentrated nor uniform. 
        
        When dim=1, k0 is arbitrarily set to 1.
        
        beta is inversely proportional to the variance
        
        
        inputs:
            
            dim: the enclosing Euclidean dimension, 1,2,...

            meanfactor>0, with 1 as neutral value
                meanfactor > 1, encodes belief in concentration
                meanfactor < 1, encodes belief in dispersion
                
            beta>0, the gamma parameter     
        
        
        
        """
        if dim >= 2:
            kappa = np.exp(logkappa_asymptote_intersection(dim))
        else:
            assert dim == 1
            kappa = 1
        return cls(kappa*meanfactor,beta)
        


class KLPrior:
    def __init__(self,mu,kappa0,pseudo_count):
        self.mu = mu = np.atleast_1d(mu)
        assert mu.ndim==1
        dim = len(mu)
        self.logC = logC = logNormConst(dim)
        self.ref = gvmf(logC,mu,kappa0)
        self.pseudo_count = pseudo_count
        self.kappa0 = kappa0
        self.rep = self.mode()                  # a representative
        

    def mode(self):
        return self.kappa0

    def loglh(self, k=None, logk=None):
        k, logk = k_and_logk(k, logk, True, False)
        other = gvmf(self.logC,self.mu,k)
        return self.pseudo_count*self.ref.kl(other)

    
    

def kappallh(n, dot, logC, k=None, logk=None):
    k, logk = k_and_logk(k,logk) 
    return n*logC(k, logk) + k*dot           
    


def mapestimate(n, sumx, kappa_prior, logC = None, logkappa = None):
    sumx = np.atleast_1d(sumx)
    assert sumx.ndim==1
    if logC is None:
        logC = logNormConst(len(sumx))
    sz, mu = decompose(sumx)
    if logkappa is None:
        logkappa = np.log(kappa_prior.rep)
    dot = mu @ sumx
    def f(logk):
        k = np.exp(logk)
        return -n*logC(k, logk) - k*dot - kappa_prior.loglh(k, logk)        
    res = minimize_scalar(f,[logkappa-1,logkappa]) 
    kappa = np.exp(res.x)
    return mu, kappa


if __name__ == "__main__":
    
    from psda.vmf_onedim import gvmf, logNormConst
    from psda.vmf_sampler import sample_uniform
    
    import matplotlib.pyplot as plt
    
    
    dim = 1
    mu = sample_uniform(dim).ravel()
    logC = logNormConst(dim)
    kappa = 1 
    vmf = gvmf(logC,mu,kappa)      
    n = 100
    x = vmf.sample(n)
    
    prior = GammaPrior.assign(dim, 1.0, 0.001)
    sumx = x.sum()
    #muhat, kappahat = mapestimate(n,sumx,prior)
    dot = mu*sumx
    
    fac = np.linspace(np.log(1/10),np.log(2),200)
    llh = kappallh(n, dot, logC,None,np.log(kappa)+fac)
    plt.plot(fac,llh,label='llh')
    plt.plot(fac,prior.loglh(logk=np.log(kappa)+fac),label='llh')
    plt.legend()
    plt.grid()
    
    
    
    
    
    
    
    
    






