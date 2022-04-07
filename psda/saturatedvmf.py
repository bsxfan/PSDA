import numpy as np
from scipy.special import gammaln

from psda.besseli import k_and_logk

log2 = np.log(2)
logsqrt2pi = np.log(2*np.pi)/2

class SatVMF:
    def __init__(self, dim):
        self.dim = dim
        self.nu = nu = dim/2-1
        self.slope = slope = nu + 0.5 
        self.tau = (nu*log2 - logsqrt2pi + gammaln(nu+1)) / slope
    
    def logCvmf_e(self, k=None, logk=None):
        k, logk = k_and_logk(k, logk, False, True)
        return logsqrt2pi + self.slope*logk
    
    def logCvmf(self, k=None, logk=None):
        k, logk = k_and_logk(k, logk)
        return logsqrt2pi + self.slope*logk - k
    
    
    def rho(self, k=None, logk=None):
        nu = self.nu
        k, logk = k_and_logk(k, logk, True, False)
        return (k-nu-0.5) / k
    
    
    def rhoinv(self, rho=None, logrho=None):
        nu = self.nu
        rho, logrho = k_and_logk(rho, logrho, True, False)
        return (nu+0.5)/(1-rho)
        