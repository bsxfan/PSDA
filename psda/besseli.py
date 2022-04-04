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

def log_ive_raw(nu, x):
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
    
    """
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
        
        

    def small_log_iv(self,logx):
        """
        Short series expansion for: 
            
            log iv(nu,x)) = log Inu(x) 
            
        for smallish x > 0. At a fixed number of terms, accuracy depends on nu. 
        We use this series expansion only if ive underflows, effecting an  
        automatic decision when to invoke this expansion. We found log(ive) 
        to be accurate up to the point (going to smaller x) where underflow 
        still does not happen. 
        
        """
        num = self.exponent * (logx-log2)
        return logsumexp(num-self.den,axis=0)


    def large_log_ive(self, logx, asymptote = True):
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
        
        """
        lin_asymptote = - (log2pi + logx)/2
        if asymptote: 
            return lin_asymptote
        return np.log1p(-(4*nu**2-1)/(8*np.exp(logx))) + lin_asymptote




    def __call__(self, x, logx = None, exp_scale = False):
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
            
            x: scalar or vector, the values should be non-negative
               if x==0 and nu > 0 -inf is returned quietly
               if x==0 and nu = 0, 0 is returned
               
            logx: make this available if you already have it lying around,
                  it is needed when x is small, or large
                  
            exp_scale: flag default=False: log Bessel-I is returned. 
                            If true: log ive is returned instead.      
                  
                  
        returns: scalar or vector:
            
            log I(nu,x),       if not exp_scale          
            log I(nu,x) - x,   if exp_scale          
        
        """
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
        log_nan =  np.log(x[nan]) if logx is None else logx[nan]
        y[nan] = self.large_log_ive(log_nan,asymptote=False)
        if not exp_scale: y[nan] += x[nan]  # undo scaling done by ive

        # patch underflow
        not_ok = np.logical_not(ok)
        not_nan = np.logical_not(nan)
        uf = np.logical_and(not_ok, not_nan)   
        log_uf =  np.log(x[uf]) if logx is None else logx[uf]
        y[uf] = self.small_log_iv(log_uf)  
        if exp_scale: y[uf] -= x[uf]

        return y



    def log_iv(self, x, logx = None):
        """
        Returns log I(nu, x). This is the same as __call__ with the default
        flag.
            
        See __call__ for more details.
        
        inputs:
            
            x: scalar or vector, the values should be non-negative
               if x==0 and nu > 0 -inf is returned quietly
               if x==0 and nu = 0, 0 is returned
               
            logx: make this available if you already have it lying around,
                  it is needed when x is small, or large
                  
                  
        returns: scalar or vector log I(nu,x)          
        
        """
        return self(x, logx, exp_scale = False)

    

    def log_ive(self, x, logx = None):
        """
        Returns exponentially scaled log Bessel-I: 
            
            log [I(nu, x) exp(-x)] = log I(nu,x) - x. 
            
        This inokes __call__ with exp_scaling=True. See __call__ for more 
        details.
        
        inputs:
            
            x: scalar or vector, the values should be non-negative
               if x==0 and nu > 0 -inf is returned quietly
               if x==0 and nu = 0, 0 is returned
               
            logx: make this available if you already have it lying around,
                  it is needed when x is small, or large
                  
                  
        returns: scalar or vector log I(nu,x) - x          
        
        """
        
        return self(x, logx, exp_scale = True)


    def logCvmf(self, log_kappa, exp_scaling = False):
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
            return nu*log2 + gammaln(nu+1)  # irrespective of exp_scaling
        logI = self(np.exp(log_kappa), log_kappa, exp_scaling)
        y = nu*log_kappa - logI
        return y
    
    

    def logCvmf_e(self,log_kappa):
        """
        log normalization constant (numerator) for Von Mises-Fisher 
        distribution, with nu = dim/2-1
        
        
            log Cvmf_e(kappa) = log [ kappa^nu / (I_nu(kappa) exp(-kappa)) ]
                              = nu*log(kappa) + kappa - log I_nu(kappa)            
            
            VMF(x | mu, kappa) \propto Cvmf_e(kappa) exp[kappa*(mu'x-1)]
            
        
        
        input: log_kappa, where kappa >= 0 is the concentration
        
        returns: function value(s) 
                 
                 The output has the same shape as the input.
        """
        
        return self.logCvmf(log_kappa, exp_scaling=True)
        




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

    

def fastLogCvmf_e(nu, d=np.pi, tune = True, quiet = True, err = None):
    
    """
    This works: 
        very well for large nu, 
        ok for smaller nu 
        and worst for nu=0 (VMF dim = 2). 
    
    It tunes a pre-and-post-scaled-and-shifted softplus approximation to the
    function:
    
         log Cvmf_e(log kappa). 
         
    The approximation is constrained to always obays the left limit and the 
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
    
    
    f = lambda x: a + b*softplus(c + d*x)
    f.nu = nu
    f.slow = target = LogBesselI(nu).logCvmf_e
    if not tune: 
        f.params = (a,b,c,d)
        return f
    
    if err is None:
        err = lambda x,y: (y-target(x))**2
    neg_err = lambda x: -err(x,f(x))
    x = minimize_scalar(neg_err,(2.0,4.0)).x
    #tarx = target(x)
    
    if not quiet:
        print(f'\ntuning softplus for nu={nu}')
        print(f'  max abs error of {np.sqrt(-neg_err(x))} at {x}')
    
    def obj(d):
        b, c = bc(d)    
        fx = a + b*softplus(c + d*x)
        return err(x,fx)
    d = minimize_scalar(obj,(d*0.9,d*1.1)).x
    if not quiet: print(f'  new d={d}, local error = {obj(d)}')
    b, c = bc(d)
    
    # this happens anyway, f already sees the new b,c,d
    # f = lambda x: a + b*softplus(c + d*x)

    x = minimize_scalar(neg_err,(2.0,4.0)).x
    if not quiet: print(f'  new max abs error of {np.sqrt(-neg_err(x))} at {x}')

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
        
            f: a function handle for the fast approximation, with:
                
                f.nu 
                f.params = (a,b,c,d), the scaling and shifting constants
                f.slow, function handle the slower reference implementation
    


    Note: An altenative is to tune 
    
             exp(affine->softplus->affine) 
          
          to fit rho. This approximation gives a sigmoid. The true 
          rho(log_kappa) is close to a sigmoid, but is somewhat less smooth, 
          especially for small nu.



    """
    
    
    C = fastLogCvmf_e(nu,tune=nu>0,quiet=quiet)
    C1slow = LogBesselI(nu+1).logCvmf_e
    
    def teacher(x):
        return x + C.slow(x) - C1slow(x)

    
    def student1(x,c1):
        return x + C(x) - c1
    
    
    def err1(x,c1):
        return (np.exp(teacher(x))-np.exp(student1(x,c1)))**2
    
    
    C1 = fastLogCvmf_e(nu+1, quiet=quiet, err=err1)
    
    
    f = lambda x: x + C(x) - C1(x)
    f.slow = teacher
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
    
    small = logBesselI.small_log_iv(logk)
    splice = logBesselI(k)
    
    
    with np.errstate(divide='ignore'):
        ref = log_ive_raw(nu, k)
    
    
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
    
    

    

    logx = np.linspace(-5,5,200)
    x = np.exp(logx)
    plt.figure()
    #for dim in [100, 110, 120]:
    for dim in [128, 256, 512]:
    #for dim in [2, 3, 4]:
        nu = dim/2-1
        #fast = fastLogCvmf_e(nu,tune=nu>0,quiet=False)
        fast = fastLogCvmf_e(nu,tune=nu>0,quiet=False)
        target = fast.slow
        y = target(logx)
        plt.plot(logx,y,label=f'dim={dim}')
        plt.plot(logx,fast(logx),'--')
        
        # y = nu*logx - LogBesselI(nu)(x,exp_scale=True)
        # plt.plot(logx,y,'r',label='new')
        
    plt.grid()
    plt.xlabel('log k')
    plt.ylabel('log C_nu(k) + k')
    plt.legend()
    plt.show()
    
        
    print("\n\n")
    nu = 255
    logI = LogBesselI(nu)
    for logx in (20,21):
        raw = log_ive_raw(nu, np.exp(logx))
        s1 = logI.large_log_ive(logx,asymptote=True)
        s2 = logI.large_log_ive(logx,asymptote=False)
        print(f"logx={logx}: {raw:.5f}, {s2:.5f}, {s1:.5f}")
        
        
        
        


