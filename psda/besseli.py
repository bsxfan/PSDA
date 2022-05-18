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



def k_and_logk(k = None, logk = None, compute_k = True, compute_logk = True):
    """
    Convenience method used by all functions that have inputs in the style:
        
        k and/or logk
        
    """
    assert k is not None or logk is not None, "at least one of k or logk is required"
    if compute_k and k is None:
        k = np.exp(logk)
    if compute_logk and logk is None:    
        with np.errstate(divide='ignore'):
            logk = np.log(k)
    return k, logk        




def log_ive_raw(nu, k = None, logk = None):
    """
    This wrapper returns:
        
        np.log(ive(nu,k))    
    
    scipy.special.ive underflows for k too small relative to nu. This cannot 
    be fixed in ive, without changing to a logarithmic function return value.   
    If ive underflows (returns 0), then the log throws a warning and this
    raw wrapper function returns -inf.
    
    If both nu and k are too large, NaN is returned quietly, 
    e.g. ive(255,np.exp(21)). I believe this a bug. The ive function values 
    for even larger inputs do still have floating point representations.
        
    This underflow and NaN behaviour is 'patched up' in the class LogBesselI 
    and its methods, which provide logrithmic input and output interfaces where 
    needed. 
    
    
    inputs: k and/or logk. 
            Only k is used, but it will be computed from logk if not given.
    
    
    """
    k, logk = k_and_logk(k, logk, True, False)
    
    return np.log(ive(nu,k))



class LogBesselI:
    """
    Callable to implement log I_nu(k), for nu >=0 and k >= 0.
    
    Unlike scipy.special.ive, this callable will not underflow if k is small
    relative to nu and it catches NaNs and recomputes the value using 2 terms
    of a series expansion for large arguments.
    
    The degree, nu is stored in the callable, while x is supplied to the call,
    e.g.:
    
        logI =  LogBesselI(nu)
        y1 = logI(k1)
        y2 = logI(k2)
    
    I_nu(k) >= 0, so log is -inf or real 
    I_nu(k) (and its log) is monotonic rising
    
    log I_0(0) = 0, 
    log I_nu(0) = -inf (no warning), for nu > 0
    
    For large k, I_nu(k) --> exp(k) / sqrt(2 pi k)
    
    
    """
    
    
    def __init__(self, nu, n=5):
        assert nu >= 0, 'Bessel-I is defined for nu < 0, but this code excludes that.'
        self.nu = nu
        self.n = n
        m = np.arange(n)
        self.exponent = (2*m+nu).reshape(-1,1)
        self.den = (logfactorial(m) + gammaln(m+1+nu)).reshape(-1,1)
        self.at0 = 0.0 if nu==0 else -np.inf
        
        

    def small_log_iv(self, k=None, logk=None):
        """
        Short series expansion for: 
            
            log iv(nu,k)) = log Inu(k) 
            
        for smallish k > 0. At a fixed number of terms, accuracy depends on nu. 
        We use this series expansion only if ive underflows, effecting an  
        automatic decision when to invoke this expansion. We found log(ive) 
        to be accurate up to the point (going to smaller x) where underflow 
        still does not happen. 

        inputs: k and/or logk. 
                Only logk is used, but it will be computed from k if not given.


        
        """
        k, logk = k_and_logk(k, logk, False, True)
        num = self.exponent * (logk-log2)
        return logsumexp(num-self.den,axis=0)


    def large_log_ive(self, k = None, logk = None, asymptote = True):
        """
        Evaluates linear asymptote for log ive(nu,k) for large k.
        
            log ive(nu,k)) = log Inu(k) - k   --> (log2pi - logk) / 2

        If input flag asymptote = False, the results is refined using also the 
        next term of a series expansion for large arguments.
        
        Example:

            nu = 255
            logI = LogBesselI(nu)
            for logk in (20,21):
                raw = log_ive_raw(nu, np.exp(logk))
                s1 = logI.large_log_ive(logk,asymptote=True)
                s2 = logI.large_log_ive(logk,asymptote=False)
                print(f"logk={logk}: {raw:.5f}, {s2:.5f}, {s1:.5f}")
        
            > logk=20: -10.91901, -10.91901, -10.91894
            > logk=21:       nan, -11.41896, -11.41894


        We use this call to patch up log(ive) in cases where ive returns NaN.
        (We assume this happens only for large k. If this is not the case, 
        the log1p below can also NaN if k is too small relative to nu.) 
        

        inputs: k and/or logk. 
                For asymptote=True, only logk is used.
                For asymptote=False, both are used.



        """
        nu = self.nu
        k, logk = k_and_logk(k, logk, not asymptote, True)
        lin_asymptote = - (log2pi + logk)/2
        if asymptote: 
            return lin_asymptote
        return np.log1p(-(4*nu**2-1)/(8*k)) + lin_asymptote




    def __call__(self, k=None, logk=None, exp_scale = False):
        """
        Evaluates log I(nu, k), so that it also works for small and large 
        values of k.
        
          - k = 0 is valid
          - scipy.special.ive is used, unless it underflows or returns NaN
          - ive underflow happens when k is small relative to nu, this is
            fixed by a logsumexp low-order series epansion
          - ive NaN is probably a bug. It happens for large nu and k. It is
            fixed with a different (small) series expansion.
        
        
        inputs:
            
            k and/ or logk: scalars or vectors
               
               Both k and logk are used.
               
               The k-values should be non-negative:
                 if k==0 and nu > 0 -inf is returned quietly
                 if k==0 and nu = 0, 0 is returned
               
                  
            exp_scale: flag default=False: log Bessel-I is returned. 
                                  If true: log ive is returned instead.      
                  
                  
        returns: scalar or vector:
            
            log I(nu,k),       if not exp_scale          
            log I(nu,k) - k,   if exp_scale          
        
        """
        k, logk = k_and_logk(k, logk, True, False)
        if np.isscalar(k):
            k = np.array([k])
            if logk is not None:
                 logk = np.array([logk])
            return self.__call__(k,logk,exp_scale)[0]
        elif k.ndim>1:
            sh = k.shape
            if logk is not None: logk = logk.ravel()
            y = self.__call__(k.ravel(),logk,exp_scale)
            return y.reshape(sh)
        
        assert all(k >= 0)

        # try ive for all k
        y = ive(self.nu,k) 
        
        # apply logs when k=0, or y > 0 and not NaN
        ok = np.logical_or(k==0, y > 0) # ive gives correct answer (0 or 1) for x==0
        with np.errstate(divide='ignore'): # y may be 0 if x ==0
            y[ok] = np.log(y[ok])       
        if not exp_scale: y[ok] += k[ok]  # undo scaling done by ive
        
        # patch overflow
        nan = np.isnan(y)  # we assume this signals overflow
        knan = k[nan]
        log_knan =  np.log(knan) if logk is None else logk[nan]
        y[nan] = self.large_log_ive(knan, log_knan, asymptote=False)
        if not exp_scale: y[nan] += knan  # undo scaling done by ive

        # patch underflow
        not_ok = np.logical_not(ok)
        not_nan = np.logical_not(nan)
        uf = np.logical_and(not_ok, not_nan)   
        logk_uf =  np.log(k[uf]) if logk is None else logk[uf]
        y[uf] = self.small_log_iv(logk=logk_uf)  
        if exp_scale: y[uf] -= k[uf]

        return y



    def log_iv(self, k=None, logk = None):
        """
        Returns log I(nu, k). This is the same as __call__ with the default
        flag.
            
        See __call__ for more details.
        
        inputs:
            
            k and/ or logk: scalars or vectors
               
               Both k and lox are used.
                  
                  
        returns: scalar or vector log I(nu,k)          
        
        """
        return self(k, logk, exp_scale = False)

    

    def log_ive(self, k=None, logk=None):
        """
        Returns exponentially scaled log Bessel-I: 
            
            log [I(nu, k) exp(-k)] = log I(nu,k) - k. 
            
        This inokes __call__ with exp_scaling=True. See __call__ for more 
        details.
        
        inputs:
            
            k and/ or logk: scalars or vectors
               
               Both k and logk are used. Supply both if you have them.
                  
                  
        returns: scalar or vector log I(nu,k) - k          
        
        """
        
        return self(k, logk, exp_scale = True)


    def logCvmf(self, k = None, logk = None, exp_scale = False):
        """
        log normalization constant (numerator) for Von-Mises-Fisher 
        distribution, with nu = dim/2-1
        
        
            log Cvmf(k) = log [ k^nu / I_nu(k) ]
            
            
            VMF(x | mu, k) \propto Cvmf(k) exp[kappa*mu'x]
            
        
        
        input: k and/or logk, where k >= 0 is the concentration
        
               Both k and logk are used. Supply both if you have both.
        
        returns: function value(s) 
                 
                 The output has the same shape as the input.


        Notes:
            
            Cvmf omits a factor that is dependent only on the dimension.
        
            The limit at k=0 (or logk=-np.inf) is handled in this call, 
            but only works for a scalar input.
            
            If you need the derivative, see LogBesselIPair.logCvmf().               
        
        
        """
        nu = self.nu
        k, logk = k_and_logk(k, logk)
        if np.isscalar(k) and k == 0:
            return nu*log2 + gammaln(nu+1)  # irrespective of exp_scaling
        logI = self(k, logk, exp_scale)
        y = nu*logk - logI
        return y
    
    

    def logCvmf_e(self, k=None, logk=None):
        """
        log normalization constant (numerator) for Von Mises-Fisher 
        distribution, with nu = dim/2-1
        
        
            log Cvmf_e(k) = log [ k^nu / (I_nu(k) exp(-k)) ]
                              = nu*log(k) + k - log I_nu(k)            
            
            VMF(x | mu, k) \propto Cvmf_e(k) exp[k*(mu'x-1)]
            
        
        
        input: k, and/or logk, where k >= 0 is the concentration
        
               Both k and logk are used. Supply both if you have both.
        
        returns: function value(s) 
                 
                 The output has the same shape as the input.
        """
        
        return self.logCvmf(k, logk, exp_scale=True)
        




class LogBesselIPair:
    """
    This is a callable that computes log I_nu and log I_{nu+1} and their
    derivatives in a single call.
    
    The degree nu is fixed within the object.
    
    To compute rho = I_{n+1} / I_nu, you need both I's and then the derivatives 
    are almost free:
        
    d/dk I(nu,k) = I(nu-1, k) - (nu/x)I(nu,k)
                 = (nu/k)I(nu,k) + I(nu+1,k)
                 = (I(nu-1,k) + I(nu+1,k)) / 2
        
             so
    
    d/dk I(nu+1,k) = I(nu, z) - ((nu+1)/k) I(nu+1,k)
    
    
    What is rho? For a Von Mises-Fisher distribution, with concentration k,
    0 <= rho(k) < 1 gives the norm of the expected value. 
    
    
    """
    def __init__(self, nu):
        self.nu = nu
        self.logI = LogBesselI(nu)
        self.logI1 = LogBesselI(nu+1)
        
    def __call__(self, k=None, logk=None):
        """
        input: k and/or logk. Both are used, so supply both if you have them.

        returns: an IPair, from which function values and derivatives
                 can be obtained as properties.         
        
        """
        nu = self.nu
        k, logk = k_and_logk(k, logk, True, True)
        y = self.logI(k,logk)
        y1 = self.logI1(k,logk)
        return IPair(nu, k, logk, y, y1)
        


class IPair:
    def __init__(self, nu, k, logk, y, y1):
        self.nu = nu
        self.k = k
        self.logk = logk
        self.logI = y
        self.logI1 = y1
        
        
    @property    
    def dlogI_dlogk(self):
        nu = self.nu
        logk, y, y1 = self.logk, self.logI, self.logI1
        return nu + np.exp(logk + y1 - y)

    @property    
    def dlogI1_dlogk(self):
        nu = self.nu
        logk, y, y1 = self.logk, self.logI, self.logI1
        return np.exp(logk + y - y1) - (nu+1)

    @property
    def dlogI_dk(self):
        return self.dI_dlogk / self.k 

    @property
    def dlogI1_dk(self):
        return self.dI1_dlogk / self.k

    
    
    @property
    def logCvmf(self):
        """
        see documentation for BesselI.logCvmf
        """
        nu = self.nu
        logk, logI = self.logk, self.logI
        return nu*logk - logI
        
    @property
    def dlogCvmf_dlogk(self):
        return self.nu - self.dlogI_dlogk

    @property
    def dlogCvmf_dk(self):
        return self.dlogCvmf_dlogk / self.k


        
    @property 
    def log_rho(self):
        """
        rho(k) = I_nu+1(k) / I_nu(k)
        """
        k = self.k
        if np.isscalar(k) and k==0:
            return -np.inf
        return self.logI1 - self.logI

    @property 
    def dlog_rho_dlogk(self):
        """
        rho(x) = I_nu+1(x) / I_nu(x)
        """
        return self.dlogI1_dlogk - self.dlogI_dlogk
    
    @property 
    def dlog_rho_dk(self):
        return self.dlog_rho_dlogk / self.k

        
    @property
    def rho(self):
        """
        rho(x) = I_nu+1(x) / I_nu(x)
        """
        return np.exp(self.log_rho)



    @property
    def drho_dlogk(self):
        """
        rho(x) = I_nu+1(x) / I_nu(x)
        """
        return self.rho * self.dlog_rho_dlogk

    @property
    def drho_dk(self):
        """
        rho(k) = I_nu+1(k) / I_nu(k)
        """
        return self.rho * self.dlog_rho_dk




def softplus(x): 
    """
    This is just one way to define a numericallly stable softplus. 
    It implements log(1+exp(x)), but will not overflow for large x.
    """
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    

def fastLogCvmf_e(logI: LogBesselI, 
                  d=np.pi, tune = True, quiet = True, err = None):
    
    """
    This works: 
        very well for large nu, 
        ok for smaller nu 
        and worst for nu=0 (VMF dim = 2). 
    
    It tunes a pre-and-post-scaled-and-shifted softplus approximation to the
    function:
    
         log Cvmf_e(log k). 
         
    The approximation is constrained to always obey the left limit and the 
    right linear asymptote of logCvmf_e.
    
    
    The true functions for small nu are less smooth than our approximations,
    espically for nu=0, which has an extra bulge below the softplus elbow.
    For large nu, the trur function is very smooth and well-approximated by
    the softplus approximation.
    
    
    inputs: 
        
        logI: LogBesselI: this contains nu = dim/2-1  
        
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
    
    nu = logI.nu
    slow = logI.logCvmf_e
    
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
    
    
    approx = lambda logk: a + b*softplus(c + d*logk)
    def f(k=None, logk=None):
        k, logk = k_and_logk(k, logk, False, True)
        return approx(logk)
    
    f.nu = nu
    f.slow = slow
    if not tune: 
        f.params = (a,b,c,d)
        return f
    
    target = lambda logk: slow(logk=logk)
    
    if err is None:
        err = lambda logk,y: (y-target(logk))**2
    neg_err = lambda logk: -err(logk,approx(logk))
    logk = minimize_scalar(neg_err,(2.0,4.0)).x
    
    if not quiet:
        print(f'\ntuning softplus for nu={nu}')
        print(f'  max abs error of {np.sqrt(-neg_err(logk))} at {logk}')
    
    def obj(d):
        b, c = bc(d)    
        flogk = a + b*softplus(c + d*logk)
        return err(logk,flogk)
    d = minimize_scalar(obj,(d*0.9,d*1.1)).x
    if not quiet: print(f'  new d={d}, local error = {obj(d)}')
    b, c = bc(d)
    
    # this happens anyway, f already sees the new b,c,d
    # f = lambda x: a + b*softplus(c + d*x)

    logk = minimize_scalar(neg_err,(2.0,4.0)).x
    if not quiet: print(f'  new max abs error of {np.sqrt(-neg_err(logk))} at {logk}')

    f.params = (a,b,c,d)

    return f
    


def fast_logrho(logI: LogBesselI, fastLogCe = None, quiet = True):
    """
    This works ok for nu=0 and well for nu>0
    
    It tunes two softplus log Cvmf_e approximations and uses their difference
    to approximate log rho. 
    
    
    inputs: 
        
        logI: LogBesselI: this contains nu = dim/2-1  
        
        quiet: boolean (default True), print tuning progress if False
    
    
    
    returns: 
        
            f: a function handle for the fast approximation, which maps:
                
                k and/or logk to log rho(k). 
                
                   The approximation uses only logk.
                
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
    nu = logI.nu
    logI1 = LogBesselI(nu+1)
    Cslow = logI.logCvmf_e
    Cslow1 = logI1.logCvmf_e
    
    C = fastLogCe or fastLogCvmf_e(logI, tune=nu>0, quiet=quiet)
    
    def teacher(logk):
        return logk + Cslow(logk=logk) - Cslow1(logk=logk)

    
    def student1(logk,c1):
        return logk + C(logk=logk) - c1
    
    
    def err1(logk,c1):
        return (np.exp(teacher(logk))-np.exp(student1(logk,c1)))**2
    
    
    C1 = fastLogCvmf_e(logI1, quiet=quiet, err=err1)
    
    
    # fast log rho
    def fast(k=None, logk=None): 
        k, logk = k_and_logk(k, logk, False, True)
        return logk + C(logk=logk) - C1(logk=logk)
    
    # slow log rho
    def slow(k=None, logk=None):
        k, logk = k_and_logk(k, logk, False, True)
        return teacher(logk)
    
    fast.slow = slow
    fast.nu = nu
    fast.C = C
    fast.C1 = C1

    return fast    
    


    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    dim = 256
    nu = dim/2-1

    logk = np.linspace(-5,5,200)
    k = np.exp(logk)
    
    logBesselI = LogBesselI(nu,5)
    
    small = logBesselI.small_log_iv(logk=logk)
    
    
    with np.errstate(divide='ignore'):
        ref = log_ive_raw(nu, k)
    
    
    plt.figure()
    plt.semilogx(k,small,'g',label='small')
    plt.semilogx(k,ref,'r--',label='ref')
    plt.semilogx(k,ref-small,label='err')
    plt.legend()
    plt.grid()
    plt.show()
    
    
    
    
    logk = np.linspace(-6,14,200)
    nu = 127
    pair = LogBesselIPair(nu)(logk=logk)
    rho, drho_dlogk = pair.rho, pair.drho_dlogk
    plt.figure()
    plt.plot(logk,rho,'r',label='rho')
    plt.plot(logk,drho_dlogk,label='dy/dlogk')
    plt.grid()
    plt.xlabel('log k')
    plt.ylabel('rho')
    plt.title(f'nu = {nu}')
    
    fastlogrho = fast_logrho(LogBesselI(nu))
    y = np.exp(fastlogrho(logk=logk))
    plt.plot(logk,y,'g--',label='rho approx')
    
    
    plt.legend()
    plt.show()
    
    
    
    logk = np.linspace(-6,20,200)
    plt.figure()
    for dim in [128, 256, 512]:
        nu = dim/2-1
        logI = LogBesselI(nu)
        y = logI.logCvmf_e(logk=logk)
        plt.plot(logk,y,label=f'dim={dim}')
        y = (nu+0.5)*logk + log2pi/2
        plt.plot(logk,y,'--')
    plt.grid()
    plt.xlabel('log k')
    plt.ylabel('log C_nu(k) + k')
    plt.title('asymptotes')
    plt.legend()
    plt.show()
    
    

    

    logk = np.linspace(-5,21,200)
    x = np.exp(logk)
    plt.figure()
    #for dim in [100, 110, 120]:
    for dim in [128, 256, 512]:
    #for dim in [2, 3, 4]:
        nu = dim/2-1
        fast = fastLogCvmf_e(LogBesselI(nu), tune=nu>0, quiet=False)
        target = fast.slow
        y = target(x,logk)
        plt.plot(logk,y,label=f'dim={dim}')
        plt.plot(logk,fast(x,logk),'--')
        
        
    plt.grid()
    plt.xlabel('log k')
    plt.ylabel('log C_nu(k) + k')
    plt.legend()
    plt.show()
    
        
    print("\n\n")
    nu = 255
    logI = LogBesselI(nu)
    for logk in (20,21):
        k = np.exp(logk)
        raw = log_ive_raw(nu, k)
        s1 = logI.large_log_ive(k, logk,asymptote=True)
        s2 = logI.large_log_ive(k, logk,asymptote=False)
        print(f"logk={logk}: {raw:.5f}, {s2:.5f}, {s1:.5f}")
        
        
        
        


