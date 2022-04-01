import numpy as np

from psda.besseli import fastLogCvmf_e, fast_logrho



class FastLogI:
    def __init__(self, nu):
        self.nu = nu
        self.fast = fast = fastLogCvmf_e(nu)  # fast approximation function
        self.slow = fast.slow                 # slower reference function
        self.abcd = abcd = fast.params
        
        d = abcd[-1]
        self.fast1 = fast1 = fastLogCvmf_e(nu+1,d=d,tune=False)
        #self.fast1 = fast1 = fastLogCvmf_e(nu+1,d=d)
        self.slow1 = fast1.slow
        
        
    def logCvmf_e(self, log_kappa, slow = False):
        if not slow: return self.fast(log_kappa)
        return self.slow(log_kappa)
    
    
    def logI(self, log_kappa, slow = False):
        nu = self.nu
        return nu*log_kappa + np.exp(log_kappa) - self.logCvmf_e(log_kappa, slow)

    
    def logI_e(self, log_kappa, slow = False):
        nu = self.nu
        return nu*log_kappa - self.logCvmf_e(log_kappa, slow)
    
    
    def logrho(self, log_kappa, slow = False):
        if slow:
            return log_kappa + self.slow(log_kappa) - self.slow1(log_kappa) 
        else:
            return log_kappa + self.fast(log_kappa) - self.fast1(log_kappa) 
            
    def rho(self, log_kappa, slow = False):
        return np.exp(self.logrho(log_kappa, slow))



if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    
    dim = 128
    
    
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
    
    # fast = np.exp(logrho(log_kappa))
    # slow = np.exp(logrho.slow(log_kappa))
    fast = logrho(log_kappa)
    slow = logrho.slow(log_kappa)
    
    plt.figure()
    plt.plot(log_kappa,slow,label='slow')
    plt.plot(log_kappa,fast,'--',label='fast')
    plt.grid()
    plt.legend()
    plt.xlabel('log kappa')
    plt.ylabel('logrho')
    
    
    
    
    
    
    
    
    
    

    
    