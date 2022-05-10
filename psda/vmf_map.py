import numpy as np

from psda.vmf_onedim import logNormConst, gvmf
from psda.vmf import decompose
from psda.besseli import k_and_logk

from scipy.optimize import minimize_scalar
from scipy.special import psi, gammaln



def logkappa_asymptote_intersection(dim):
    assert dim > 1
    nu = dim/2-1
    return ( nu*np.log(2) + gammaln(nu+1) - np.log(2*np.pi)/2 ) / (nu + 0.5)     


def horz_asymptote(dim):
    assert dim > 1
    nu = dim/2-1
    return nu*np.log(2) + gammaln(nu+1)

def lin_asymptote(dim,logk):
    assert dim > 1
    nu = dim/2-1
    return np.log(2*np.pi)/2 + (nu+0.5)*logk

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
        y = (-self.pseudo_count)*self.ref.kl(other)
        if np.isscalar(y): return y
        if y.size==1: return y.item()
        return y.ravel()
            

    @classmethod
    def assign(cls, dim, mu, modefactor, pseudo_count):
        """
        When dim >=2. the prior mode for kappa is set at modefactor*k0, 
        where k0 depends on dim and is neutral in the sense of being neither 
        concentrated nor uniform. 
        
        When dim=1, k0 is arbitrarily set to 1.
        
        beta is inversely proportional to the variance
        
        
        inputs:
            
            dim: the enclosing Euclidean dimension, 1,2,...

            modefactor>0, with 1 as neutral value
                modefactor > 1, encodes belief in concentration
                modefactor < 1, encodes belief in dispersion
                
            pseudocount>0, the number (a fraction is allowed) of phantom
                           data points drawn from the reference VMF  
        
        
        
        """
        if dim >= 2:
            kappa = np.exp(logkappa_asymptote_intersection(dim))
        else:
            assert dim == 1
            kappa = 1
        return cls(mu, kappa*modefactor, pseudo_count)
    
    

def kappallh(n, dot, logC, k=None, logk=None):
    k, logk = k_and_logk(k,logk) 
    return n*logC(k, logk) + k*dot           
    

def mu_ml(sumx):
    sumx = np.atleast_1d(sumx)
    assert sumx.ndim==1
    sz, mu = decompose(sumx)
    return mu


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
    
    from subsphere.pca import Globe
    
    
    import matplotlib.pyplot as plt
    
    
    dim = 1
    mu = sample_uniform(dim).ravel()
    logC = logNormConst(dim)
    kappa = 2
    vmf = gvmf(logC,mu,kappa)      
    n = 100
    x = vmf.sample(n)
    sumx = x.sum()

    muhat = mu_ml(sumx)    
    #prior = GammaPrior.assign(dim, 1.0, 0.001)
    prior = KLPrior.assign(dim, muhat, 1, 1)
    muhat, kappahat = mapestimate(n,sumx,prior)
    dot = mu*sumx
    
    kmin = min(kappa,kappahat) / 5
    kmax = max(kappa,kappahat) * 5
    
    
    kk = np.linspace(kmin,kmax,200)
    llh = kappallh(n, dot, logC,kk)
    prior_llh = prior.loglh(kk)
    y = llh + prior_llh
    ymax = y.max()
    priormax = prior_llh.max()
    llhmax = llh.max()
    plt.figure()
    plt.plot(kk,y-ymax,label='joint')
    plt.plot(kk,prior_llh-priormax,label='prior')
    plt.plot(kk,llh-llhmax,label='llh')
    plt.plot(kappa,0.0,'*',label='true') 
    plt.plot(kappahat,0.0,'*',label='map') 
    plt.legend()
    plt.grid()
    plt.show()
    
    
    
    dim = 20
    mu = sample_uniform(dim).ravel()
    logC = logNormConst(dim)
    
    
    logk = np.linspace(-10,10,200)
    logk0 = logkappa_asymptote_intersection(dim)
    y = logC(logk=logk,exp_scale=True)
    y0 = horz_asymptote(dim) #logC(logk=logk0,exp_scale=True)
    y1 = lin_asymptote(dim, logk)
    plt.figure()
    plt.plot(logk,y)
    plt.plot(logk,y1,'--')
    plt.plot([logk[0],logk0],[y0,y0],'*')
    plt.grid()
    plt.show()
    

    kappa = 100*np.exp(logkappa_asymptote_intersection(dim))
    vmf = gvmf(logC,mu,kappa)    
    # X = vmf.sample(1000)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # Globe.plotgrid(ax)
    # ax.scatter(*X.T, color='g', marker='.')
    # ax.scatter(*mu, color='r', marker='*')
    # fig.show()
    
    
    
    n = 100
    x = vmf.sample(n)
    sumx = x.sum(axis=0)

    muhat = mu_ml(sumx)    
    prior = KLPrior.assign(dim, muhat, 1, 0.1)
    muhat, kappahat = mapestimate(n,sumx,prior)
    dot = mu@sumx
    
    kmin = min(kappa,kappahat) / 5
    kmax = max(kappa,kappahat) * 5
    
    
    kk = np.linspace(kmin,kmax,200)
    llh = kappallh(n, dot, logC,kk)
    prior_llh = prior.loglh(kk)
    y = llh + prior_llh
    ymax = y.max()
    priormax = prior_llh.max()
    llhmax = llh.max()
    plt.figure()
    plt.plot(kk,y-ymax,label='joint')
    plt.plot(kk,prior_llh-priormax,label='prior')
    plt.plot(kk,llh-llhmax,label='llh')
    plt.plot(kappa,0.0,'*',label='true') 
    plt.plot(kappahat,0.0,'*',label='map') 
    plt.legend()
    plt.grid()
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    






