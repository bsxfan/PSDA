import numpy as np

from scipy.special import ive, gammaln, logsumexp
from scipy.optimize import toms748


from psda.vmf_sampler import rotate_to_mu, sample_vmf_canonical_mu

def logfactorial(x):
    """
    Natural log of factorial. Invokes scipy.special.gammaln.
    log x! = log gamma(x+1)
    """
    return gammaln(x+1)



class LogBesselI:
    """
    We use scipy for larger arguments and a logsumexp over a small
    series expansion for small arguments. The scipy implementation is OK
    for large arguments, because we use the exponentially scaled (ive)
    variant.

    For later, if we need derivatives:
    See: https://functions.wolfram.com/Bessel-TypeFunctions/BesselI/20/ShowAll.html
    
    d/dx I(nu,x) = I(n-1, z) - (nu/x)I(nu,x)
                 = (nu/x)I(nu,x) + I(nu+1,x)
                 = (I(nu-1,x) + I(nu+1,x)) / 2

    """
    def __init__(self, nu, n=5):
        self.nu = nu
        self.n = n
        m = np.arange(n)
        self.exp = (2*m+nu).reshape(-1,1)
        self.den = (logfactorial(m) + gammaln(m+1+nu)).reshape(-1,1)
        self.thr = np.sqrt(self.nu+1)
        
    def switchover(self):
        x = self.thr
        return x, self.small(x), self.large(x)
     
        
        
    def __call__(self,x):
        y = self.splice(x)
        return y
        
    
    def splice(self,x):
        if np.isscalar(x):
            return self.__call__(np.array([x]))[0]
        zeros = x==0
        small = np.logical_and(x < self.thr, x > 0)         
        large = np.logical_not(small)
        y = np.zeros_like(x)
        y[zeros] = self.nu == 0
        y[small] = self.small(x[small])
        y[large] = self.large(x[large])
        return y
    
    
    
    def small(self,x):
        """
        short series expansion for log Inu(x) for 0 < x, smallish 
        """
        num = self.exp*np.log(x/2)
        return logsumexp(num-self.den,axis=0)


    def large(self,x):
        """
        log Inu(x), for x not too small (log 0 warning if x too small)
        """
        return np.log(ive(self.nu,x)) + x



class LogNormConst:
    """
    This is a callable for the log normalization constant for Von Mises-Fisher 
    distribution, computed as a function of the dimension and the concentration 
    parameter. A term that dependens only on the dimension is omitted.
    
    The dimensionality, via nu = dim/2-1, is supplied at construction, 
    while the concentration is passed later to the constructed callable.


    As a callable it returns:
        
        log C_nu(k) = log k**nu - log Inu(k)   
    
    
    An additional method, rho(k) computes the norm of the expected 
    value of a VMF with this normalization constant, Cnu(k). 
    
    Also supplied are rhoinv(rho) that uses root finding to invert rho(k), 
    and rhoinv_fast(rho), that does a fast approximation. 
    
    """
    def __init__(self,dim,n=5):
        self.nu = nu = dim/2-1
        self.dim = dim
        self.logInu = LogBesselI(nu,n)
        self.logInu1 = LogBesselI(nu+1,n)


    def __call__(self, k):
        """
        Returns the log normalization constant, omitting a term dependent
        only on the dimensionality, nu.
        
        k > 0: the VMF concentration parameter
        
        The limit at k--> 0 exists, but is not implemented yet 
        
        """
        assert k >= 0
        nu, logInu = self.nu, self.logInu
        # min (nu=dim/2-1) = 0, so 1e-20 << sqrt(1+nu), 
        # but just testing for k==0 works too  
        if k < 1e-20:     
            return nu*np.log(2) + gammaln(nu+1)
        return nu*np.log(k) - logInu(k) 

    
    def rho(self,k):
        """
        The norm of the expected value for VMF(nu,k). The expected value is 
        monotonic rising, from rho(0) = 0 to rho(inf) = 1. The limit at 0
        is handled explicitly, but the one at infinity is not implemented.
        """
        if np.isscalar(k):  
            return self.rho(np.array([k]))[0]
        nz = k>0
        r = np.zeros_like(k)
        if any(nz):
            knz = k[nz]
            r[nz] = np.exp(self.logInu1(knz) - self.logInu(knz))
        return r    

    def rhoinv_fast(self,rho):
        """
        Fast, approximate inversion of rho given by Banerjee'05
        """
        if np.isscalar(rho):  
            return self.rhoinv_fast(np.array([rho]))[0]
        dim = self.dim
        nz = rho>0
        k = np.zeros_like(rho)
        if any(nz):
            rhonz = rho[nz]
            rho2 = rhonz**2
            k[nz] = rhonz*(dim-rho2) / (1-rho2)
        return k    
    
    
    def rhoinv(self,rho):
        """
        Slower, more accurate inversion of rho using a root finder.
        """
        if not np.isscalar(rho):
            return np.array([self.rhoinv(ri) for ri in rho])
        if rho == 0: return 0.0
        k0 = self.rhoinv_fast(rho)
        f = lambda x: self.rho(np.exp(x)) - rho
        left = np.log(k0)
        fleft = f(left)
        if fleft == 0: return k0
        if fleft < 0:
            right = left
            fright = fleft
            while fright <=0:
               right = 2.3 + right 
               fright = f(right)
        else:  # fleft > 0
            right = left
            fright = fleft
            while fleft >= 0:
               left = left - 2.3 
               fleft = f(left)
        return np.exp(toms748(f,left,right))
                
def decompose(x):
    """
    If x is a vector, return: norm, x/norm
    
    If x is a matrix, do the same for every row. The norms are returned
    as a 1d array (not as a column).
    
    """
    if x.ndim == 1:
        norm = np.sqrt((x**2).sum(axis=-1))
        if norm == 0: return 0.0, x
        return norm, x/norm
    norm = np.sqrt((x**2).sum(axis=-1,keepdims=True))
    zeros = norm == 0
    if any(zeros):
        norm[zeros] = 1
    return norm.ravel(), x/norm
        
def compose(norm,mu):
    """
    Does the inverse of decompose. Returns norm.reshape(-1,1)*mu

        norm: scalar or vector
        mu: vector or matrix
    
    """
    if not np.isscalar(norm): norm = norm.reshape(-1,1)
    return norm*mu

class VMF:
    """
    Von Mises-Fisher distribution. The parameters are supplied at construction. 
    """
    def __init__(self, mu, k = None, logC = None):
        """
        mu: (dim, ): mean direction, must be lengh-normalized.
            
            (n,dim): if mu is a matrix, each row gives a different distribution
            
            If k is not given, then mu is the natural parameter, which is 
            not length-normed. Then mu will be decomposed so that its norm
            becomes the concentration.
        
        k>=0: scalar, concentration parameter
              
             (n, ): if k is a vector mu must be a matrix and they must agree in 
                    shape[0]
              
             If k is omitted, mu is assumed to be the unnormalized natural
             parameter and k is recovered from the norm of mu.
             
             logC: LogNormConst, optional. If already available, it can be 
                   supplied to save memory and compute. 
        
        """
        if k is None:
            kmu = mu
            k, mu = decompose(mu)
        else:
            kmu = compose(k, mu)
        self.mu = mu
        self.k = k
        self.kmu = kmu
        self.dim = dim = mu.shape[-1]
        if logC is None: 
            logC = LogNormConst(dim)
        else:
            assert logC.dim == dim
        self.logC = logC    
        self.logCk = logC(k)    
        self.rho = logC.rho   # function to compute k -> norm of mean
        
    def mean(self):
        """
        Returns the expected value in R^d, which is inside the sphere, 
        not on it.
        """
        r = self.rho(self.k)
        return compose(r,self.mu)
    
    def kmu(self):
        """
        returns the natural parameter, which is in R^d
        """
        return self.kmu
        
    @classmethod
    def max_likelihood(cls, mean, logC = None):    
        """
        The returns the maximum-likelihood estimate(s) for one or more VMFs, given
        the sufficient stats.
        
        mean: (dim,) the empirical mean (average) of the observations. 
              The observations are on the unit hypersphere and the mean must be
              inside it (with norm stricly < 1)
              
              (n,dim): do n independent ML estimates
              
        returns: a VMF object, containing one or more distributions, all of the 
                 same dimensionality      
        
        
        """
        norm, mu = decompose(mean)
        assert norm < 1,  "The mean norm must be strictly < 1"
        dim = len(mean)
        if logC is None: 
            logC = LogNormConst(dim)
        else:
            assert logC.dim == mean.shape[-1]
        k = logC.rhoinv(norm)
        return cls(mu,k,logC)
    
    
    
    def sample_quick_and_dirty(self, n_or_labels):
        """
        Quick and dirty (statistically incorrect) samples, meant only for
        preliminary tyre-kicking.
        
        If self contains a single distribution, supply n, the number of 
        required samples.
        
        If self contains multiple distribution, supply labels (n, ) to select 
        for each sample the distribution to be sampled from

        """


        if np.isscalar(n_or_labels):
            n = n_or_labels
            labels = None
            assert self.mu.ndim == 1
        else:
            labels = n_or_labels
            n = len(labels)
            assert self.mu.ndim == 2
        
        
        dim, k = self.dim, self.k
        mean = self.mean()
        if labels is not None:
            mean = mean[labels,:]
        X = np.random.randn(n,dim)/np.sqrt(k) + mean
        return decompose(X)[1]
    
    
    def sample(self, n_or_labels):
        """
        Generate samples from the von Mises-Fisher distribution.
        If self contains a single distribution, supply n, the number of
        required samples.
        If self contains multiple distributions, supply labels (n, ) to select
        for each sample the distribution to be sampled from.
        Reference:
        o Stochastic Sampling of the Hyperspherical von Mises–Fisher Distribution
          Without Rejection Methods - Kurz & Hanebeck, 2015
        o Simulation of the von Mises-Fisher distribution - Wood, 1994
        """

        dim, mu = self.dim, self.mu

        if np.isscalar(n_or_labels):   # n iid samples from a single distribution
            n = n_or_labels
            assert mu.ndim == 1
            assert np.isscalar(self.k)
            X = np.vstack([sample_vmf_canonical_mu(dim,self.k) for i in range(n)])
            X = rotate_to_mu(X,mu)

        else:                          # index distribution by labels 
            labels = n_or_labels
            assert mu.ndim == 2
            if np.isscalar(self.k):    # broadcast k
                kk = np.full((len(labels),),self.k)
            else:
                kk = self.k[labels]

            X = np.vstack([sample_vmf_canonical_mu(dim,k) for k in kk])

            for lab in np.unique(labels):
                ii = labels==lab
                X[ii] = rotate_to_mu(X[ii],mu[lab])

        return X
        
        
    def logpdf(self, X):
        """
        If X is a vector, return scalar or vector, depending if self contains
        one or more distributions.
        
        If X is a matrix, returns an (m,) vector or an (m,n) matrix, where m
        is the number of rows of X and n is the number of distributions in self.
        """
        llh = X @ self.kmu.T
        return llh + self.logCk
    
    
    def entropy(self):
        return -self.logpdf(self.mean())
    
    def kl(self, other):
        mean = self.mean()
        return self.logpdf(mean) - other.logpdf(mean)
    
    
    def __repr__(self):
        if np.isscalar(self.k):
            return f"VMF(mu:{self.mu.shape}, k={self.k})"
        return f"VMF(mean:{self.mean.shape}, k:{self.k.shape})"

    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

# k = np.exp(np.linspace(-5,4,1000))
# dim, n = 256, 1
# C1 = LogNormConst(dim,n)
# y = C1(k)

# thr = C1.logI.thr
# y_thr = C1(thr)


# plt.figure()
# plt.semilogx(k,y,label='spliced compromise')
# plt.semilogx(thr,y_thr,'*',label='splice location')



# plt.grid()
# plt.xlabel('concentration parameter')
# plt.ylabel('Von Mises-Fisher log norm. const.')
# plt.legend()
# plt.title(f'approximating terms: {n}')
# plt.show()



# dim, n = 256, 5
# C5 = LogNormConst(dim,n)
# y = C5(k)

# thr = C5.logI.thr
# y_thr = C5(thr)


# plt.figure()
# plt.semilogx(k,y,label='spliced compromise')
# plt.semilogx(thr,y_thr,'*',label='splice location')



# plt.grid()
# plt.xlabel('concentration parameter')
# plt.ylabel('Von Mises-Fisher log norm. const.')
# plt.legend()
# plt.title(f'approximating terms: {n}')
# plt.show()


    # k = np.exp(np.linspace(-5,20,20))
    # dim = 256
    # logC = LogNormConst(dim)
    # rho = logC.rho(k)
    # plt.semilogx(k,rho)
    # kk = logC.rhoinv_fast(rho)
    # plt.semilogx(kk,rho,'--')

    x0 = np.array([0.9,0.9])/1.5
    # vmf = VMF(x0)
    vmf = VMF.max_likelihood(x0)
    
    X = vmf.sample(10)
    plt.scatter(X[:,0],X[:,1])
    plt.axis('square')
    plt.xlim(-1.2,1.2)
    plt.ylim(-1.2,1.2)
    plt.grid()

