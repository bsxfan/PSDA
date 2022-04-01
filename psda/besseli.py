import numpy as np

from scipy.special import gammaln, logsumexp, ive
from scipy.optimize import minimize_scalar

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
        self.exponent = (2*m+nu).reshape(-1,1)
        self.den = (logfactorial(m) + gammaln(m+1+nu)).reshape(-1,1)
        self.at0 = 0.0 if nu==0 else -np.inf

    def small(self,logx):
        """
        short series expansion for log Inu(x) for 0 < x, smallish 
        """
        num = self.exponent * (logx-log2)
        return logsumexp(num-self.den,axis=0)



    def __call__(self, x, logx = None):
        """
        Evaluates log I(nu, x), so that it also works for small values of x.
          - x = 0 is valid
          - scipy.special.ive is used, unless it underflows
          - underflow happens when x is small relative to nu, this is
            fixed by a logsumexp low-order series epansion
        
        
        inputs:
            
            x: scalar or vector, the values should be non-negative
               if x==0 and nu > 0 -inf is returned quietly
               if x==0 and nu = 0, 0 is returned
               
            logx: make this available if you already have it lying around,
                  it is needed when x is small
                  
                  
        returns: scalar or vector log I(nu,x)          
        
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

    def log_ive(self, x, logx = None):
        """
        Returns log [I(nu, x) exp(-x)] = log I(nu,x) - x. 
            
        See __call__ for more details.
        
        inputs:
            
            x: scalar or vector, the values should be non-negative
               if x==0 and nu > 0 -inf is returned quietly
               if x==0 and nu = 0, 0 is returned
               
            logx: make this available if you already have it lying around,
                  it is needed when x is small
                  
                  
        returns: scalar or vector log I(nu,x)          
        
        """
        if np.isscalar(x):
            assert x >= 0
            if x == 0: return self.at0
            return self.log_ive(np.array([x]))[0]
        assert all(x >= 0)
        y = ive(self.nu,x)  
        ok = np.logical_or(x==0, y > 0) # ive gives correct answer (0 or 1) for x==0
        uf = np.logical_not(ok)                # underflow if y==0 and x > 0
        with np.errstate(divide='ignore'):
            y[ok] = np.log(y[ok])              # y may be 0 if x ==0 
        logx =  np.log(x[uf]) if logx is None else logx[uf]
        y[uf] = self.small(logx) - x[uf]      # fix underflows 
        return y


    def logCvmf(self,log_kappa):
        """
        log normalization constant (numerator) for Von-Mises-Fisher 
        distribution, with nu = dim/2-1
        
        
            log Cvmf(kappa) = log [ kappa^nu / I_nu(kappa) ]
            
            
            VMF(x | mu, kappa) \propto Cvmf(kappa) exp[kappa*mu'x]
            
        
        
        input: log_kappa, where kappa >= 0 is the concentration
        
        returns: function value(s) 
                 
                 The output has the same shape as the input.


        Notes:
            
            Cvmf omits a factor that is dependent only on the dimension.
        
            The limit at kappa=0 is handled in this call, if you set 
            log_kappa = -np.inf. This only works for a scalar input.
            
            If you need the derivative, see LogBesselIPair.logCvmf().               
        
        
        """
        nu = self.nu
        if np.isscalar(log_kappa) and log_kappa == -np.inf:
            return nu*log2 + gammaln(nu+1)
        logI = self(np.exp(log_kappa),log_kappa)
        y = nu*log_kappa - logI
        return y
    
    

    def logCvmf_e(self,log_kappa):
        """
        log normalization constant (numerator) for Von-Mises-Fisher 
        distribution, with nu = dim/2-1
        
        
            log Cvmf_e(kappa) = log [ kappa^nu / (I_nu(kappa) exp(-kappa)) ]
                              = nu*log(kappa) + kappa - log I_nu(kappa)            
            
            VMF(x | mu, kappa) \propto Cvmf_e(kappa) exp[kappa*(mu'x-1)]
            
        
        
        input: log_kappa, where kappa >= 0 is the concentration
        
        returns: function value(s) 
                 
                 The output has the same shape as the input.
        """
        
        nu = self.nu
        if np.isscalar(log_kappa) and log_kappa == -np.inf:
            return nu*log2 + gammaln(nu+1)  # yes, this is the same as logCvmf(-inf)
        log_ive = self.log_ive(np.exp(log_kappa),log_kappa)
        y = nu*log_kappa - log_ive
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
    
    
    What is rho? For a Von Mises-Fisher distribution, with concentration kappa,
    0 <= rho(kappa) < 1 gives the radius of the expected value. 
    
    
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
    
    
    def logCvmf(self,log_kappa):
        """
        log normalization constant (numerator) for Von-Mises-Fisher 
        distribution, with nu = dim/2-1
        
        
            Cvmf(kappa) = nu^kappa / I_nu(kappa)
        
        
        input: log_kappa, where kappa > 0 is the concentration
        
        returns: function value(s), derivative(s) 
                 
                 Both outputs have the same shape as the input.


        Notes:
            
            Cvmf omits a factor that is dependent only on the dimension.
        
            The limit at kappa=0 is not handled in this call, because in
            an optimization context, log_kappa should remain finite. But in 
            case you find it useful elsewhere (arxiv.org/abs/2203.14893): 
            
                lim_{kappa --> 0} Cvmf(kappa) = 2^nu Gamma(nu+1)
                
        
        
        """
        nu = self.nu
        logI, dlogI_dlogkappa = self.logI(log_kappa)
        y = nu*log_kappa - logI
        dy_dlogkappa = nu - dlogI_dlogkappa
        return y, dy_dlogkappa
        



        
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


def softplus(x): 
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    

def fastLogCvmf_e(nu, d=np.pi, tune = True, quiet = True):
    left = nu*log2 + gammaln(nu+1)     # left flat asymptote
    right_offs = log2pi/2              # offset for right linear asymptote
    right_slope = nu + 0.5             # slope for right linear asymptote
    
    a = left
    def bc(d): 
        b = right_slope / d
        c = (right_offs - a) / b
        return b, c
    b, c = bc(d)    
    #print(f'nu={nu}: b={b}, c = {c}, d={d}')
    
    
    f = lambda x: a + b*softplus(c + d*x)
    f.nu = nu
    f.slow = target = LogBesselI(nu).logCvmf_e
    if not tune: 
        f.params = (a,b,c,d)
        return f
    
    neg_err = lambda x: -(f(x)-target(x))**2
    x = minimize_scalar(neg_err,(2.0,4.0)).x
    tarx = target(x)
    
    if not quiet:
        print(f'\ntuning softplus for nu={nu}')
        print(f'  max abs error of {np.sqrt(-neg_err(x))} at {x}')
    
    def obj(d):
        b, c = bc(d)    
        fx = a + b*softplus(c + d*x)
        return (fx-tarx)**2
    d = minimize_scalar(obj,(d*0.9,d*1.1)).x
    if not quiet: print(f'  new d={d}, local error = {obj(d)}')
    b, c = bc(d)
    
    # this happens anyway, f already sees the new b,c,d
    # f = lambda x: a + b*softplus(c + d*x)

    neg_err = lambda x: -(f(x)-target(x))**2
    x = minimize_scalar(neg_err,(2.0,4.0)).x
    tarx = target(x)
    if not quiet: print(f'  new max abs error of {np.sqrt(-neg_err(x))} at {x}')

    f.params = (a,b,c,d)

    return f
    
    
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
    
    
    plt.figure()
    plt.semilogx(k,small,'g',label='small')
    plt.semilogx(k,ref,'r--',label='ref')
    plt.semilogx(k,ref-small,label='err')
    plt.legend()
    plt.grid()
    plt.show()
    
    
    
    
    pair = LogBesselIPair(100)
    logx = np.linspace(-6,14,200)
    logr, ddlogx = pair.logRho(logx)
    plt.figure()
    plt.plot(logx,np.exp(logr),label='rho')
    plt.plot(logx,ddlogx*np.exp(logr),label='dy/dx')
    plt.grid()
    plt.xlabel('log k')
    plt.ylabel('rho')
    plt.title('nu = 100')
    plt.legend()
    plt.show()
    
    
    
    logx = np.linspace(-6,20,200)
    #x = np.exp(logx)
    plt.figure()
    for dim in [128, 256, 512]:
        nu = dim/2-1
        bi = LogBesselI(nu)
        #y = bi.logCvmf(logx)
        y = bi.logCvmf_e(logx)
        plt.plot(logx,y,label=f'dim={dim}')
        y = (nu+0.5)*logx + log2pi/2
        plt.plot(logx,y,'--')
    plt.grid()
    plt.xlabel('log k')
    plt.ylabel('log C_nu(k) + k')
    plt.legend()
    plt.show()
    
    

    

    logx = np.linspace(0,9,200)
    #x = np.exp(logx)
    plt.figure()
    #for dim in [100, 110, 120]:
    for dim in [128, 256, 512]:
        nu = dim/2-1
        fast = fastLogCvmf_e(nu,tune=True,quiet=False)
        target = fast.slow
        y = target(logx)
        plt.plot(logx,y,label=f'dim={dim}')
        plt.plot(logx,fast(logx),'--')
    plt.grid()
    plt.xlabel('log k')
    plt.ylabel('log C_nu(k) + k')
    plt.legend()
    plt.show()
    
        
        
        
        
        
        


