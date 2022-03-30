import numpy as np

from scipy.special import gammaln, logsumexp, ive

logfactorial = lambda x: gammaln(x+1)
log2 = np.log(2)
log2pi = np.log(2*np.pi)

def logBesselI_ive(nu, x):
    """
    Underflows for x too small relative to nu. 
    If that happens, ive returns 0 and the log throws
    a warning and returns -inf.
    
    This behaviour is fixed below.
    
    """
    return np.log(ive(nu,x)) + x



class LogBesselI:
    def __init__(self, nu, n=5):
        self.nu = nu
        self.n = n
        m = np.arange(n)
        self.exp = (2*m+nu).reshape(-1,1)
        self.den = (logfactorial(m) + gammaln(m+1+nu)).reshape(-1,1)
        self.thr = np.sqrt(self.nu+1)

    def small(self,logx):
        """
        short series expansion for log Inu(x) for 0 < x, smallish 
        """
        num = self.exp*(logx-log2)
        return logsumexp(num-self.den,axis=0)


    def large(self, x):
        
        return 



    def __call__(self, x):
        if np.isscalar(x):
            return self.__call__(np.array([x]))[0]
        y = ive(self.nu,x)
        nz = y > 0
        zz = np.logical_not(nz)
        y[nz] = np.log(y[nz]) + x[nz]
        y[zz] = self.small(np.log(x[zz]))
        return y



# Doesn't work and is slower than ive in any case
# class LogBesselI_big:
#     def __init__(self, nu, n=5):
#         """
#         Won't work for (2n-3)**2 >= 4nu**2
#         For now it's OK, because we will have large nu and small n'
        
#         To make it work more generally, change the code to process the sign of 
#         delta below separately.
        
#         """
#         self.nu = nu
#         self.n = n
#         logw = np.zeros(n)
#         nu2 = nu**2
#         for i in range(1,n):
#             odd = 2*i-1               # 1, 3, 5, ...
#             delta = 4*nu2-odd**2
#             assert delta > 0, "n too large wrt nu (can be fixed by some coding here)"
#             logw[i] = logw[i-1] + np.log(delta) \
#                                 - np.log(8*i)
#         self.logwpos = logw[0:n:2].reshape(-1,1)                        
#         self.logwneg = logw[1:n:2].reshape(-1,1)
#         exp = np.arange(n)
#         self.exppos = exp[0:n:2].reshape(-1,1)
#         self.expneg = exp[1:n:2].reshape(-1,1)
                


#     def __call__(self,logx):
#         logwpos, logwneg = self.logwpos, self.logwneg
#         exppos, expneg = self.exppos, self.expneg
        
#         pos = logsumexp(logx*exppos + logwpos, axis=0) 
#         neg = logsumexp(logx*expneg + logwneg, axis=0)
#         delta = neg - pos
#         #assert all(delta < 0)
#         delta = np.abs(delta)           
#         f = np.log1p(np.exp(delta))
#         x = np.exp(logx)
#         return x + f - (logx-log2pi)/2
    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    dim = 256
    nu = dim/2-1

    logk = np.linspace(-5,5,200)
    k = np.exp(logk)
    
    logBesselI = LogBesselI(nu,5)
    
    small = logBesselI.small(logk)
    splice = logBesselI(k)
    
    ref = logBesselI_ive(nu, k)
    
    
    
    plt.semilogx(k,small,'g',label='small')
    plt.semilogx(k,ref,'r--',label='ref')
    plt.semilogx(k,ref-small,label='err')
    plt.legend()
    plt.grid()
    plt.show()
    
    
    
    
    
    
    



    
        
        
        
        
        
        


