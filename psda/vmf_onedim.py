import numpy as np
from numpy.random import rand

from scipy.special import expit as sigmoid
from scipy.special import entr, rel_entr


from psda.vmf import LogNormConst, VMF
from psda.besseli import k_and_logk



def logNormConst(dim):
    if dim==1: 
        return LogNormConstOneDim()
    else: 
        return LogNormConst(dim) 

def gvmf(logC, mu=None, k=None):
    if logC.dim==1: return VMFOneDim(mu, k, logC)
    return VMF(mu, k, logC)
    
    
class LogNormConstOneDim:

    def __init__(self): 
        self.dim = 1

    def __call__(cls, k=None, logk=None, fast = True, exp_scale = False):
        """
        fast is ignored, it is fast already
        """
        
        k, logk = k_and_logk(k, logk, True, False)
        y = -np.logaddexp(k,-k)
        if exp_scale: return y + k
        return y

class VMFOneDim:
    """
    VMF when dim = 1
    """
    def __init__(self, mu = None, k = None, logC = None):
        if logC is None: logC = logNormConst(1)
        assert logC.dim == 1
        self.logC = logC
        self.dim = 1

        if mu is not None:
            mu = np.atleast_1d(mu)
            if mu.ndim==1:
                mu = mu.reshape(-1,1)
            assert mu.ndim==2    

        if mu is None:   # uniform
            assert k is None
            mu = np.atleast_2d(1.0)
            k = 0.0
            kmu = k*mu
        elif k is None:
            kmu = mu
            k = np.abs(kmu)
            if np.isscalar(k): 
                assert k > 0
            else:
                assert all(k>0)
            mu = kmu / k
        else:   # both mu and k given
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
        pplus1 = self.pplus1
        if n_or_labels is None:
            if pplus1.size==1:
                return self.sample(1)
            else:
                n = len(pplus1)
                return self.sample(np.arange(n))
            
        if np.isscalar(n_or_labels):   # n iid samples from a single distribution
            n = n_or_labels
            assert  pplus1.size==1
            ber = pplus1 > rand(n) 
            return (2.0*ber -1).reshape(-1,1)
            

        else:                          # index distribution by labels 
            labels = n_or_labels
            pplus1 = pplus1[labels,:]
            n = pplus1.size
            ber = pplus1 > rand(n).reshape(-1,1) 
            return (2.0*ber -1)
        
    @classmethod
    def uniform(cls):
        return cls()
        
            
        
    def entropy(self):
        return entr(self.pplus1) + entr(self.pmin1)


    def kl(self, other):
        return rel_entr(self.pplus1, other.pplus1) \
             + rel_entr(self.pmin1, other.pmin1)
        
        

if __name__ == "__main__":

    kmu = VMFOneDim.uniform().sample(10)
    VMFOneDim(kmu).sample([0,1,2])    
    
    



