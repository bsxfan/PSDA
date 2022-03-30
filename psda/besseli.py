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
    """
    Callable to implement log I_nu(x), for nu >=0 and x >= 0.
    
    Unlike scipy.special.ive, this callable will not underflow if x is small
    relative to nu.
    
    The degree, nu is stored in the callable, while x is supplied to the call,
    e.g.:
    
        logI =  LogBesselI(nu)
        y1 = logI(x1)
        y2 = logI(x2)
    
    I_nu(x) >= 0, so log is -inf or real 
    I_nu(x) (and its log) is monotonic rising
    log I_0(0) = 0, 
    log I_nu(0) = -inf (no warning), for nu > 0
    
    """
    
    
    def __init__(self, nu, n=5):
        assert nu >= 0, 'Bessel-I is defined for nu < 0, but this code excludes that.'
        self.nu = nu
        self.n = n
        m = np.arange(n)
        self.exp = (2*m+nu).reshape(-1,1)
        self.den = (logfactorial(m) + gammaln(m+1+nu)).reshape(-1,1)
        self.at0 = 0.0 if nu==0 else -np.inf

    def small(self,logx):
        """
        short series expansion for log Inu(x) for 0 < x, smallish 
        """
        num = self.exp*(logx-log2)
        return logsumexp(num-self.den,axis=0)



    def __call__(self, x, logx = None):
        """
        Also supply logx if available
        """
        if np.isscalar(x):
            assert x >= 0
            if x == 0: return self.at0
            return self.__call__(np.array([x]))[0]
        assert all(x >= 0)
        y = ive(self.nu,x)  
        ok = np.logical_or(x==0, y > 0) # ive gives correct answer (0 or 1) for x==0
        uf = np.logical_not(ok)                # underflow if y==0 and x > 0
        with np.errstate(divide='ignore'):
            y[ok] = np.log(y[ok]) + x[ok]      # y may be 0 if x ==0 
        logx =  np.log(x[uf]) if logx is None else logx[uf]
        y[uf] = self.small(logx)      # fix underflows 
        return y



class LogBesselIPair:
    """
    This is a callable that computes log I_nu and log I_{nu+1} and their
    derivatives in a single call.
    
    The degree nu is fixed within the object.
    
    To compute rho = I_{n+1} / I_nu, you need both I's and then the derivatives 
    are almost free:
        
    d/dx I(nu,x) = I(nu-1, z) - (nu/x)I(nu,x)
                 = (nu/x)I(nu,x) + I(nu+1,x)
                 = (I(nu-1,x) + I(nu+1,x)) / 2
        
             so
    
    d/dx I(nu+1,x) = I(nu, z) - ((nu+1)/x) I(nu+1,x)
    
    
    
    """
    def __init__(self, nu, n = 5):
        self.nu = nu
        self.logI = LogBesselI(nu,n)
        self.logI1 = LogBesselI(nu+1,n)
        
    def __call__(self, logx):
        """
        input: logx

        For y, y1 = logI_nu(x), logI_{nu+1}, 
        returns: y, y1, dy_dlogx, dy1_dlogx
        
        
        """
        nu = self.nu
        x = np.exp(logx)
        y = self.logI(x,logx)
        y1 = self.logI1(x,logx)
        dy_dlogx = nu + np.exp(logx + y1 - y)    
        dy1_dlogx = np.exp(logx + y - y1) - (nu+1)
        return y, y1, dy_dlogx, dy1_dlogx
        

    def logI(self, logx):
        """
        input: logx

        For y = logI_nu(x)
        returns: y, dy_dlogx
        
        
        """
        nu = self.nu
        x = np.exp(logx)
        y = self.logI(x,logx)
        y1 = self.logI1(x,logx)
        dy_dlogx = nu + np.exp(logx + y1 - y)    
        return y, dy_dlogx



        
    def logRho(self,logx):
        """
        input: logx

        For rho(x) = I_nu+1(x) / I_nu(x)
        returns: log rho(x), dlogrho_dlogx
    
        """

        if np.isscalar(logx):
            y, dy_dlogx = self.logRho(np.array([logx]))
            return y[0], dy_dlogx[0]


        y, y1, dy_dlogx, dy1_dlogx = self(logx)
        return y1-y, dy1_dlogx - dy_dlogx
    
    
    def rho(self,x):
        """
        input: x
        
        For y = rho(x)
        returns: y, dy_dx
        
        The limit at x = 0 is implemented on for scalar x
        
        """
        if np.isscalar(x):
            if x == 0: return 0, 1/(2*(self.nu+1))  # limit at 0
            y, dydx = self.rho(np.array[x])
            return y[0], dydx[0]
        assert all(x > 0), 'zeros in array argument not coded yet'
        logx = np.log(x)
        logr, dlogr_dlogx = self.logRho(logx)
        r = np.exp(logr)
        dr_dx = r * dlogr_dlogx / x
        return r, dr_dx




    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    dim = 256
    nu = dim/2-1

    logk = np.linspace(-5,5,200)
    k = np.exp(logk)
    
    logBesselI = LogBesselI(nu,5)
    
    small = logBesselI.small(logk)
    splice = logBesselI(k)
    
    
    with np.errstate(divide='ignore'):
        ref = logBesselI_ive(nu, k)
    
    
    
    plt.semilogx(k,small,'g',label='small')
    plt.semilogx(k,ref,'r--',label='ref')
    plt.semilogx(k,ref-small,label='err')
    plt.legend()
    plt.grid()
    plt.show()
    
    
    
    
    pair = LogBesselIPair(100)
    logx = np.linspace(-6,14,200)
    logr, ddlogx = pair.logRho(logx)
    plt.plot(logx,np.exp(logr),label='rho')
    plt.plot(logx,ddlogx*np.exp(logr),label='dy/dx')
    plt.grid()
    plt.xlabel('log k')
    plt.ylabel('rho')
    plt.title('nu = 100')
    plt.legend()
    
    
    
    
    
    



    
        
        
        
        
        
        


