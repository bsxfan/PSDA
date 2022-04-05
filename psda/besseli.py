"""
Bessel-I is numerically tricky  it can underflow and overflow and is much 
slower than typical float functions like log and exp.

Bessel-I is available in a few forms in scipy.special:
    iv(nu,x): I_nu(x)           # underflows and overflows for large nu
    ive(nu,x): I_nu(x) exp(-x)  # underflows, better against overflow, but it 
                                              still happens if x is too large
    ivp: derivatives for iv
    
Bessel-I and even its lagrithm is available in Tensorflow.    

Bessel-I is not available in Pytorch (except for nu = 0 and 1).     
    
        
In this module are some tools to compute log I_nu in regions where iv or ive
would underflow, or overflow.

We also have some tools to create a fast, bespoke approximation to the 
log of the exponentially scaled normalization constant of the Von Mises-Fisher
distribution, with nu = dim/2-1 and concentration kappa:
    
    log Cvmf_e(nu,kappa) = nu*log(kappa) - log(ive). 

The approximation form is:

    affine --> softplus --> affine

The affine parameters are tuned for every nu. Tuning invokes scipy.special.ive,
but once tuned, the approximation can run anywhere without scipy. So it can be 
used for example in Pytorch (with backprop) on the GPU. No Pytorch tools are 
included here, but sending the tuned approximation for use on any platform that 
has a softplus available is trivial. 
 
We found the approximation to be an order of magnitude faster than the (patched)
scipy.special.ive function.


"""

import numpy as np

from scipy.special import gammaln, logsumexp, ive
from scipy.optimize import minimize_scalar

logfactorial = lambda x: gammaln(x+1)
log2 = np.log(2)
log2pi = np.log(2*np.pi)



def x_and_logx(x = None, logx = None, compute_x = True, compute_logx = True):
    """
    Convenience method for used by all functions that take x and/or logx.
    """
    assert x is not None or logx is not None, "at least one of x or logx is required"
    if compute_x and x is None:
        x = np.exp(logx)
    if compute_logx and logx is None:    
        with np.errstate(divide='ignore'):
            logx = np.log(x)
    return x, logx        




def log_ive_raw(nu, x = None, logx = None):
    """
    This wrapper returns:
        
        np.log(ive(nu,x))    
    
    scipy.special.ive underflows for x too small relative to nu. This cannot 
    be fixed in ive, without changing to a logarithmic function return value.   
    If ive underflows (returns 0), then the log throws a warning and this
    raw wrapper function returns -inf.
    
    If both nu and x are too large, NaN is returned quietly, 
    e.g. ive(255,np.exp(21)). I believe this a bug. The ive function values 
    for even larger inputs do still have floating point representations.
        
    This underflow and NaN behaviour is 'patched up' in the class LogBesselI 
    and its methods, which provide logrithmic input and output interfaces where 
    needed. 
    
    
    inputs: x and/or logx. 
            Only x is used, but it will be computed from logx if not given.
    
    
    """
    x, logx = x_and_logx(x, logx, True, False)
    
    return np.log(ive(nu,x))



class LogBesselI:
    """
    Callable to implement log I_nu(x), for nu >=0 and x >= 0.
    
    Unlike scipy.special.ive, this callable will not underflow if x is small
    relative to nu and it catches NaNs and recomputes the value using 2 terms
    of a series expansion for large arguments.
    
    The degree, nu is stored in the callable, while x is supplied to the call,
    e.g.:
    
        logI =  LogBesselI(nu)
        y1 = logI(x1)
        y2 = logI(x2)
    
    I_nu(x) >= 0, so log is -inf or real 
    I_nu(x) (and its log) is monotonic rising
    
    log I_0(0) = 0, 
    log I_nu(0) = -inf (no warning), for nu > 0
    
    For large x, I_nu(x) --> exp(x) / sqrt(2 pi x)
    
    
    """
    
    
    def __init__(self, nu, n=5):
        assert nu >= 0, 'Bessel-I is defined for nu < 0, but this code excludes that.'
        self.nu = nu
        self.n = n
        m = np.arange(n)
        self.exponent = (2*m+nu).reshape(-1,1)
        self.den = (logfactorial(m) + gammaln(m+1+nu)).reshape(-1,1)
        self.at0 = 0.0 if nu==0 else -np.inf
        
        

    def small_log_iv(self, x=None, logx=None):
        """
        Short series expansion for: 
            
            log iv(nu,x)) = log Inu(x) 
            
        for smallish x > 0. At a fixed number of terms, accuracy depends on nu. 
        We use this series expansion only if ive underflows, effecting an  
        automatic decision when to invoke this expansion. We found log(ive) 
        to be accurate up to the point (going to smaller x) where underflow 
        still does not happen. 

        inputs: x and/or logx. 
                Only logx is used, but it will be computed from x if not given.


        
        """
        x, logx = x_and_logx(x, logx, False, True)
        num = self.exponent * (logx-log2)
        return logsumexp(num-self.den,axis=0)


    def large_log_ive(self, x = None, logx = None, asymptote = True):
        """
        Evaluates linear asymptote for log ive(nu,x) for large x.
        
            log ive(nu,x)) = log Inu(x) - x   --> (log2pi - logx) / 2

        If input flag asymptote = False, the results is refined using also the 
        next term of a series expansion for large arguments.
        
        Example:

            nu = 255
            logI = LogBesselI(nu)
            for logx in (20,21):
                raw = log_ive_raw(nu, np.exp(logx))
                s1 = logI.large_log_ive(logx,asymptote=True)
                s2 = logI.large_log_ive(logx,asymptote=False)
                print(f"logx={logx}: {raw:.5f}, {s2:.5f}, {s1:.5f}")
        
            > logx=20: -10.91901, -10.91901, -10.91894
            > logx=21:       nan, -11.41896, -11.41894


        We use this call to patch up log(ive) in cases where ive returns NaN.
        (We assume this happens only for large x. If this is not the case, 
         the log1p below can also NaN if x is too small relative to nu.) 
        

        inputs: x and/or logx. 
                For asymptote=True, only x is used.
                For asymptote=False, both are used.



        """
        x, logx = x_and_logx(x, logx, not asymptote, True)
        lin_asymptote = - (log2pi + logx)/2
        if asymptote: 
            return lin_asymptote
        return np.log1p(-(4*nu**2-1)/(8*x)) + lin_asymptote




    def __call__(self, x=None, logx=None, exp_scale = False):
        """
        Evaluates log I(nu, x), so that it also works for small and large 
        values of x.
        
          - x = 0 is valid
          - scipy.special.ive is used, unless it underflows or returns NaN
          - ive underflow happens when x is small relative to nu, this is
            fixed by a logsumexp low-order series epansion
          - ive NaN is probably a bug. It happens for large nu and x. It is
            fixed with a different (small) series expansion.
        
        
        inputs:
            
            x and/ or logx: scalars or vectors
               
               Both x and lox are used.
               
               The x-values should be non-negative:
                 if x==0 and nu > 0 -inf is returned quietly
                 if x==0 and nu = 0, 0 is returned
               
                  
            exp_scale: flag default=False: log Bessel-I is returned. 
                                  If true: log ive is returned instead.      
                  
                  
        returns: scalar or vector:
            
            log I(nu,x),       if not exp_scale          
            log I(nu,x) - x,   if exp_scale          
        
        """
        x, logx = x_and_logx(x, logx, True, False)
        if np.isscalar(x):
            x = np.array([x])
            if logx is not None:
                 logx = np.array([logx])
            return self.__call__(x,logx,exp_scale)[0]

        assert all(x >= 0)

        # try ive for all x
        y = ive(self.nu,x) 
        
        # apply logs when x=0, or y > 0 and not NaN
        ok = np.logical_or(x==0, y > 0) # ive gives correct answer (0 or 1) for x==0
        with np.errstate(divide='ignore'): # y may be 0 if x ==0
            y[ok] = np.log(y[ok])       
        if not exp_scale: y[ok] += x[ok]  # undo scaling done by ive
        
        # patch overflow
        nan = np.isnan(y)  # we assume this signals overflow
        xnan = x[nan]
        log_xnan =  np.log(xnan) if logx is None else logx[nan]
        y[nan] = self.large_log_ive(xnan, log_xnan, asymptote=False)
        if not exp_scale: y[nan] += x[nan]  # undo scaling done by ive

        # patch underflow
        not_ok = np.logical_not(ok)
        not_nan = np.logical_not(nan)
        uf = np.logical_and(not_ok, not_nan)   
        logx_uf =  np.log(x[uf]) if logx is None else logx[uf]
        y[uf] = self.small_log_iv(logx=logx_uf)  
        if exp_scale: y[uf] -= x[uf]

        return y



    def log_iv(self, x=None, logx = None):
        """
        Returns log I(nu, x). This is the same as __call__ with the default
        flag.
            
        See __call__ for more details.
        
        inputs:
            
            x and/ or logx: scalars or vectors
               
               Both x and lox are used.
                  
                  
        returns: scalar or vector log I(nu,x)          
        
        """
        return self(x, logx, exp_scale = False)

    

    def log_ive(self, x=None, logx=None):
        """
        Returns exponentially scaled log Bessel-I: 
            
            log [I(nu, x) exp(-x)] = log I(nu,x) - x. 
            
        This inokes __call__ with exp_scaling=True. See __call__ for more 
        details.
        
        inputs:
            
            x and/ or logx: scalars or vectors
               
               Both x and lox are used. Supply both if you have them.
                  
                  
        returns: scalar or vector log I(nu,x) - x          
        
        """
        
        return self(x, logx, exp_scale = True)


    def logCvmf(self, x = None, logx = None, exp_scaling = False):
        """
        log normalization constant (numerator) for Von-Mises-Fisher 
        distribution, with nu = dim/2-1
        
        
            log Cvmf(x) = log [ x^nu / I_nu(x) ]
            
            
            VMF(z | mu, x) \propto Cvmf(x) exp[kappa*mu'z]
            
        
        
        input: x, and/or logx, where x >= 0 is the concentration
        
               Both x and logx are used. Supply both if you have both.
        
        returns: function value(s) 
                 
                 The output has the same shape as the input.


        Notes:
            
            Cvmf omits a factor that is dependent only on the dimension.
        
            The limit at x=0 (or logx=-np.inf) is handled in this call, 
            but only works for a scalar input.
            
            If you need the derivative, see LogBesselIPair.logCvmf().               
        
        
        """
        nu = self.nu
        x, log_kappa = x_and_logx(x, logx)
        if np.isscalar(x) and x == 0:
            return nu*log2 + gammaln(nu+1)  # irrespective of exp_scaling
        logI = self(x, logx, exp_scaling)
        y = nu*logx - logI
        return y
    
    

    def logCvmf_e(self, x=None, logx=None):
        """
        log normalization constant (numerator) for Von Mises-Fisher 
        distribution, with nu = dim/2-1
        
        
            log Cvmf_e(kappa) = log [ kappa^nu / (I_nu(kappa) exp(-kappa)) ]
                              = nu*log(kappa) + kappa - log I_nu(kappa)            
            
            VMF(x | mu, kappa) \propto Cvmf_e(kappa) exp[kappa*(mu'x-1)]
            
        
        
        input: x, and/or logx, where x >= 0 is the concentration
        
               Both x and logx are used. Supply both if you have both.
        
        returns: function value(s) 
                 
                 The output has the same shape as the input.
        """
        
        return self.logCvmf(x, logx, exp_scaling=True)
        




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
    0 <= rho(kappa) < 1 gives the norm of the expected value. 
    
    
    """
    def __init__(self, nu):
        self.nu = nu
        self.logI = LogBesselI(nu)
        self.logI1 = LogBesselI(nu+1)
        
    def __call__(self, x=None, logx=None):
        """
        input: x and/or logx. Both are used, so supply both if you have them.

        returns: an IPair, from which function values and derivatives
                 can be obtained as properties.         
        
        """
        nu = self.nu
        x, logx = x_and_logx(x, logx, True, True)
        y = self.logI(x,logx)
        y1 = self.logI1(x,logx)
        return IPair(nu, x, logx, y, y1)
        


class IPair:
    def __init__(self, nu, x, logx, y, y1):
        self.nu = nu
        self.x = x
        self.logx = logx
        self.logI = y
        self.logI1 = y1
        
        
    @property    
    def dlogI_dlogx(self):
        nu = self.nu
        logx, y, y1 = self.logx, self.logI, self.logI1
        return nu + np.exp(logx + y1 - y)

    @property    
    def dlogI1_dlogx(self):
        nu = self.nu
        logx, y, y1 = self.logx, self.logI, self.logI1
        return np.exp(logx + y - y1) - (nu+1)

    @property
    def dlogI_dx(self):
        return self.dI_dlogx / self.x 

    @property
    def dlogI1_dx(self):
        return self.dI1_dlogx / self.x

    
    
    @property
    def logCvmf(self):
        """
        see documentation for BesselI.logCvmf
        """
        nu = self.nu
        logx, logI = self.logx, self.logI
        return nu*logx - logI
        
    @property
    def dlogCvmf_dlogx(self):
        return self.nu - self.dlogI_dlogx

    @property
    def dlogCvmf_dx(self):
        return self.dlogCvmf_dlogx / self.x


        
    @property 
    def log_rho(self):
        """
        rho(x) = I_nu+1(x) / I_nu(x)
        """
        x = self.x
        if np.isscalar(x) and x==0:
            return -np.inf
        return self.logI1 - self.logI

    @property 
    def dlog_rho_dlogx(self):
        """
        rho(x) = I_nu+1(x) / I_nu(x)
        """
        return self.dlogI1_dlogx - self.dlogI_dlogx
    
    @property 
    def dlog_rho_dx(self):
        return self.dlog_rho_dlogx / self.x

        
    @property
    def rho(self):
        """
        rho(x) = I_nu+1(x) / I_nu(x)
        """
        return np.exp(self.log_rho)



    @property
    def drho_dlogx(self):
        """
        rho(x) = I_nu+1(x) / I_nu(x)
        """
        return self.rho * self.dlog_rho_dlogx

    @property
    def drho_dx(self):
        """
        rho(x) = I_nu+1(x) / I_nu(x)
        """
        return self.rho * self.dlog_rho_dx




def softplus(x): 
    """
    This is just one way to define a numericallly stable softplus. 
    It implements log(1+exp(x)), but will not overflow for large x.
    """
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    

def fastLogCvmf_e(nu, d=np.pi, tune = True, quiet = True, err = None):
    
    """
    This works: 
        very well for large nu, 
        ok for smaller nu 
        and worst for nu=0 (VMF dim = 2). 
    
    It tunes a pre-and-post-scaled-and-shifted softplus approximation to the
    function:
    
         log Cvmf_e(log kappa). 
         
    The approximation is constrained to always obey the left limit and the 
    right linear asymptote of logCvmf_e.
    
    
    The true functions for small nu are less smooth than our approximations,
    espically for nu=0, which has an extra bulge below the softplus elbow.
    For large nu, the trur function is very smooth and well-approximated by
    the softplus approximation.
    
    
    inputs: 
        
        nu >=0, the Bessel-I order  (nu=dim/2-1 for VMF distribution)
        
        d>0: (default = pi) the tunable softplus input scaling factor
        
        tune: boolean (default True)
        
        quiet: boolean (default True), print tuning progress if False
    
        err: an optional tuning objective
    
    
    returns: 
        
            f: a function handle for the fast approximation, with:
                
                f.nu 
                f.params = (a,b,c,d), the scaling and shifting constants
                f.slow, function handle for the slower reference implementation
    
    """
    
    
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
    
    
    approx = lambda logx: a + b*softplus(c + d*logx)
    def f(x=None, logx=None):
        x, logx = x_and_logx(x, logx, False, True)
        return approx(logx)
    
    f.nu = nu
    f.slow = slow = LogBesselI(nu).logCvmf_e
    if not tune: 
        f.params = (a,b,c,d)
        return f
    
    target = lambda logx: slow(logx=logx)
    
    if err is None:
        err = lambda logx,y: (y-target(logx))**2
    neg_err = lambda logx: -err(logx,approx(logx))
    logx = minimize_scalar(neg_err,(2.0,4.0)).x
    
    if not quiet:
        print(f'\ntuning softplus for nu={nu}')
        print(f'  max abs error of {np.sqrt(-neg_err(logx))} at {logx}')
    
    def obj(d):
        b, c = bc(d)    
        flogx = a + b*softplus(c + d*logx)
        return err(logx,flogx)
    d = minimize_scalar(obj,(d*0.9,d*1.1)).x
    if not quiet: print(f'  new d={d}, local error = {obj(d)}')
    b, c = bc(d)
    
    # this happens anyway, f already sees the new b,c,d
    # f = lambda x: a + b*softplus(c + d*x)

    logx = minimize_scalar(neg_err,(2.0,4.0)).x
    if not quiet: print(f'  new max abs error of {np.sqrt(-neg_err(logx))} at {logx}')

    f.params = (a,b,c,d)

    return f
    


def fast_logrho(nu, quiet = True):
    """
    This works ok for nu=0 and well for nu>0
    
    It tunes two softplus log Cvmf_e approximations and uses their difference
    to approximate log rho. 
    
    
    inputs: 
        
        nu >=0, the Bessel-I order  (nu=dim/2-1 for VMF distribution)
        
        quiet: boolean (default True), print tuning progress if False
    
    
    
    returns: 
        
            f: a function handle for the fast approximation, which maps:
                
                x and/or logx to log rho(x). 
                
                   The approximation uses only logx.
                
            Extra info is returned in attached fields:    
                f.nu 
                f.C = function handle for fast logCvmf_e(nu)
                f.C1, function handle for fast logCvmf_e(nu+1)
                f.slow function handle to reference log rho
    


    Note: An altenative is to tune 
    
             exp(affine->softplus->affine) 
          
          to fit rho. This approximation gives a sigmoid. The true 
          rho(log_kappa) is close to a sigmoid, but is somewhat less smooth, 
          especially for small nu.



    """
    
    
    C = fastLogCvmf_e(nu,tune=nu>0,quiet=quiet)
    C1slow = LogBesselI(nu+1).logCvmf_e
    
    def teacher(logx):
        return logx + C.slow(logx=logx) - C1slow(logx=logx)

    
    def student1(logx,c1):
        return logx + C(logx=logx) - c1
    
    
    def err1(logx,c1):
        return (np.exp(teacher(logx))-np.exp(student1(logx,c1)))**2
    
    
    C1 = fastLogCvmf_e(nu+1, quiet=quiet, err=err1)
    
    
    def f(x=None, logx=None): 
        x, logx = x_and_logx(x, logx, False, True)
        return logx + C(logx=logx) - C1(logx=logx)
    
    
    def slow(x=None, logx=None):
        x, log_x = x_and_logx(x, logx, False, True)
        return teacher(logx)
    
    f.slow = slow
    f.nu = nu
    f.C = C
    f.C1 = C1

    return f    
    


    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    dim = 256
    nu = dim/2-1

    logk = np.linspace(-5,5,200)
    k = np.exp(logk)
    
    logBesselI = LogBesselI(nu,5)
    
    small = logBesselI.small_log_iv(logx=logk)
    
    
    with np.errstate(divide='ignore'):
        ref = log_ive_raw(nu, k)
    
    
    plt.figure()
    plt.semilogx(k,small,'g',label='small')
    plt.semilogx(k,ref,'r--',label='ref')
    plt.semilogx(k,ref-small,label='err')
    plt.legend()
    plt.grid()
    plt.show()
    
    
    
    
    logx = np.linspace(-6,14,200)
    nu = 127
    pair = LogBesselIPair(nu)(logx=logx)
    rho, drho_dlogx = pair.rho, pair.drho_dlogx
    plt.figure()
    plt.plot(logx,rho,'r',label='rho')
    plt.plot(logx,drho_dlogx,label='dy/dx')
    plt.grid()
    plt.xlabel('log k')
    plt.ylabel('rho')
    plt.title(f'nu = {nu}')
    
    fastlogrho = fast_logrho(nu)
    y = np.exp(fastlogrho(logx=logx))
    plt.plot(logx,y,'g--',label='rho approx')
    
    
    plt.legend()
    plt.show()
    
    
    
    logx = np.linspace(-6,20,200)
    plt.figure()
    for dim in [128, 256, 512]:
        nu = dim/2-1
        bi = LogBesselI(nu)
        y = bi.logCvmf_e(logx=logx)
        plt.plot(logx,y,label=f'dim={dim}')
        y = (nu+0.5)*logx + log2pi/2
        plt.plot(logx,y,'--')
    plt.grid()
    plt.xlabel('log kappa')
    plt.ylabel('log C_nu(kappa) + k')
    plt.legend()
    plt.show()
    
    

    

    logx = np.linspace(-5,21,200)
    x = np.exp(logx)
    plt.figure()
    #for dim in [100, 110, 120]:
    for dim in [128, 256, 512]:
    #for dim in [2, 3, 4]:
        nu = dim/2-1
        fast = fastLogCvmf_e(nu,tune=nu>0,quiet=False)
        target = fast.slow
        y = target(x,logx)
        plt.plot(logx,y,label=f'dim={dim}')
        plt.plot(logx,fast(x,logx),'--')
        
        
    plt.grid()
    plt.xlabel('log k')
    plt.ylabel('log C_nu(k) + k')
    plt.legend()
    plt.show()
    
        
    print("\n\n")
    nu = 255
    logI = LogBesselI(nu)
    for logx in (20,21):
        x = np.exp(logx)
        raw = log_ive_raw(nu, x)
        s1 = logI.large_log_ive(x, logx,asymptote=True)
        s2 = logI.large_log_ive(x, logx,asymptote=False)
        print(f"logx={logx}: {raw:.5f}, {s2:.5f}, {s1:.5f}")
        
        
        
        


