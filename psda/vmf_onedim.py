import numpy as np
from numpy.random import rand

from scipy.special import expit as sigmoid


from psda.vmf import LogNormConst, VMF

def logNormConst(dim):
    if dim==1: 
        return LogNormConstOneDim()
    else: 
        return LogNormConst(dim) 

def vmf(logC, mu=None, k=None):
    if logC.dim==1: return VMFOneDim(mu, k, logC)
    return VMF(mu, k, logC)
    
    
class LogNormConstOneDim:

    def __init__(self): 
        self.dim = 1

    def __call__(cls,k):
        return -np.logaddexp(k,-k)

class VMFOneDim:
    """
    VMF when dim = 1
    """
    def __init__(self, mu = None, k = None, logC = None):
        if logC is None: logC = logNormConst(1)
        assert logC.dim == 1
        self.logC = logC
        self.dim = 1

        assert k is not None or mu is not None 
        if mu is None:
            mu = 1.0
            k = 0.0
            kmu = 0
        elif k is None:
            kmu = mu
            if not np.isscalar(kmu) and len(kmu)==1: kmu = kmu.item()
            k = np.abs(kmu)
            if np.isscalar(k): 
                assert k > 0
            else:
                assert all(k>0)
            mu = kmu / k
        else:
            if not np.isscalar(mu) and len(mu)==1: mu = mu.item()
            kmu = k*mu
        self.mu = mu
        self.k = k
        self.kmu = kmu
        self.logCk = logC(k)

        self.pplus1 = sigmoid(2*kmu)
        self.pmin1 = sigmoid(-2*kmu)
        
    def __repr__(self):
        return f"VMFOneDim:\nmu={self.mu},\nk={self.k}"
        

    def mean(self):
        return self.pplus1 - self.pmin1
        

    def sample(self, n_or_labels = None):
        """
        Generate samples from the von Mises-Fisher distribution.
        - If self contains a single distribution, supply n, the number of
          required samples.
        - If self contains multiple distributions, supply labels (n, ) to select
          for each sample the distribution to be sampled from.
        - If neither n nor labels is given, one sample from each distribution
          is returned.
        """
        kmu = self.kmu
        if n_or_labels is None:
            if np.isscalar(kmu):
                return self.sample(1)
            else:
                n = len(kmu)
                return self.sample(np.arange(n))
            
        if np.isscalar(n_or_labels):   # n iid samples from a single distribution
            n = n_or_labels
            assert  np.isscalar(kmu)
            ber = self.pplus1 > rand(n) 
            return (2.0*ber -1).reshape(-1,1)
            

        else:                          # index distribution by labels 
            labels = n_or_labels
            pplus1 = self.pplus1[labels]
            n = len(pplus1)
            ber = pplus1 > rand(n) 
            return (2.0*ber -1).reshape(-1,1)
        
    @classmethod
    def uniform(cls):
        return cls()
        
            

if __name__ == "__main__":

    kmu = VMFOneDim().sample(10)
    VMFOneDim(kmu).sample([0,1,2])    
    
    



