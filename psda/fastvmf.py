import numpy as np

from psda.besseli import fastLogCvmf_e, fast_logrho



class FastLogI:
    def __init__(self, nu):
        self.nu = nu
        self.fast = fast = fastLogCvmf_e(nu)  # fast approximation function
        self.slow = fast.slow                 # slower reference function
        self.abcd = abcd = fast.params
        
        
        
    def logCvmf_e(self, log_kappa, slow = False):
        if not slow: return self.fast(log_kappa)
        return self.slow(log_kappa)
    
    
    def logI(self, log_kappa, slow = False):
        nu = self.nu
        return nu*log_kappa + np.exp(log_kappa) - self.logCvmf_e(log_kappa, slow)

    
    def logI_e(self, log_kappa, slow = False):
        nu = self.nu
        return nu*log_kappa - self.logCvmf_e(log_kappa, slow)
    
    
    # def logrho(self, log_kappa, slow = False):
    #     if slow:
    #         return log_kappa + self.slow(log_kappa) - self.slow1(log_kappa) 
    #     else:
    #         return log_kappa + self.fast(log_kappa) - self.fast1(log_kappa) 
            
    # def rho(self, log_kappa, slow = False):
    #     return np.exp(self.logrho(log_kappa, slow))
    
    
    
class Sra:
    def __init__(self,nu):
        self.nu = nu
        self.dim = 2*(nu+1)
    
    def invrho(self, r):
        dim = self.dim
        rr = r**2
        return np.log(r*(dim-rr)/(1-rr))
    
    
    # doesn't work
    # def rho(self, logk):
    #     if not np.isscalar(logk):
    #         return np.array([self.rho(logki) for logki in logk])
    #     dim = self.dim
    #     k = np.exp(logk)
    #     r = 0.5
    #     for i in range(10):
    #         rr = r**2
    #         r = k*(1-rr)/(dim-rr)
    #     return r    
            
    



if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    
    dim = 5
    
    
    nu = dim/2 -1
    
    # logI = FastLogI(nu)
    
    
    # log_kappa = np.linspace(-5,15,200)
    
    # fast = logI.logrho(log_kappa)
    # slow = logI.logrho(log_kappa, slow=True)
    
    # plt.figure()
    # plt.plot(log_kappa,slow,label='slow')
    # plt.plot(log_kappa,fast,'--',label='fast')
    # plt.grid()
    # plt.legend()
    # plt.xlabel('log kappa')
    # plt.ylabel('logrho')
    
    
    #offset = logI.slow(-np.inf) - logI.slow1(-np.inf)
    #plt.plot(log_kappa,log_kappa+offset)
    
    logrho = fast_logrho(nu,quiet=False)    
    
    log_kappa = np.linspace(-5,15,200)
    
    fast = np.exp(logrho(log_kappa))
    slow = np.exp(logrho.slow(log_kappa))
    #fast = logrho(log_kappa)
    #slow = logrho.slow(log_kappa)
    
    sra = lambda r: r*(dim-r**2)/(1-r**2)
    rr = np.linspace(0.001,0.999,200)
    
    
    plt.figure()
    plt.plot(log_kappa,slow,label='slow')
    #plt.plot(log_kappa,fast,'--',label='fast')
    # plt.plot(np.log(sra(rr)),np.log(rr),'--',label='sra')
    plt.plot(np.log(sra(rr)),rr,'--',label='sra')
    plt.grid()
    plt.legend()
    plt.xlabel('log kappa')
    plt.ylabel('logrho')
    
    
    
    
    
    
    
    
    
    

    
    