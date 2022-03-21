"""
PSDA: Probabilistic Spherical Discriminant Analysis

"""
import numpy as np
from numpy import ndarray

from psda.vmf import VMF, compose, decompose



class PSDA:
    """
    Probabilistic Spherical Discriminant Analysis Model
    
    """
    
    def __init__(self, within_concentration:float, between_distr:VMF):
        """        
        model = PSDA(w, VMF(mu, b))  
                
                w,b > 0 
                mu (dim, ) is a lengh-normalized speaker mean
                
                
                or
                
        model = PSDA(w, VMF(mean))  
                
                w,b > 0, 
                mean (dim,) is speaker mean inside (not on) unit hypersphere
                            
                or
                
        model = em(means,counts) # see the documention for em()   
        
                model = PSDA.em(means,counts) works too


        """
        
        if not isinstance(between_distr, VMF):
            between_distr = VMF(*between_distr)
        self.w = w = within_concentration
        self.between = between = between_distr
        self.b = b = between.k
        self.mu = between.mu
        self.bmu = between.kmu
        self.logC = logC = between.logC
        self.logCb = logC(b)
        self.logCw = logC(w)
        
        
    def zposterior(self, data_sum:ndarray):
        """
        Computes the hidden variable posterior, given the sufficient statistics
        (sum of the observations). 
        
        If mutiple sums (each sum is a vector) are supplied, multiple posteriors
        are computed.
        
        The posterior(s) one or more are returned in a single VMF object.
        """
        
        w, bmu = self.w, self.bmu
        return VMF(w*data_sum + bmu)

    def marg_llh(self, data_sum, count):
        """
        Computes the marginal log-likelihood log P(X | same speaker), where
        z is integrated out. We use: log P(X | z) P(z) / P(z | X)
        
        Returns a vector of independent calculations if data_sum is a matrix 
        and count is a vector.
        """
        post = self.zposterior(data_sum)
        return self.logCw*count + self.logCb - post.logCk 
    
    
    
    
    def sample_speakers(self, n):
        return self.between.sample(n)
    
    
    def sample(self, speakers, labels):
        within = VMF(speakers, self.w, self.logC)
        return within.sample(labels)
    
    
    
    
    def prep(self, X: ndarray):
        """
        Does some precomputation for fast computation of a matrix of LLR
        scores.
        
            X: vector or matrix of observations (in rows)
               Each row can represent a single observation, or multiple 
               observations. For single observation, the row must be on the
               unit hypersphere. For multiple observations, the row must be the 
               sum of observations, which can be anywhere in R^d.
               
               To be clear, when doing multi-enroll trials, the enrollment 
               embeddings must be summed, not averaged. Do not length-norm 
               again after summing.
               
               
            returns: a Side that contains precomputed stuff for fast scoring. 
                     The Side provides a method for scoring against another 
                     Side.
        """
        return Side(self,X.astype(np.float64))
    
    
    
    def llr_matrix(self, enroll:ndarray, test:ndarray) -> ndarray:
        """
        Convenience method. See PSDA.prep() for details.
        """
        return self.prep(enroll).llr_matrix(self.prep(test))
    
    
    @classmethod
    def em(cls, means: ndarray, counts:ndarray, niters = 10, w0 = 1.0, quiet = False):
        """
        Trains a PSDA model from data.
        
            means: (n, dim) the means of each of the n classes available for training
        
            counts: (n,) the number of examples of each class
        
            niters: the number of EM iterations to run
        
            w0>0: (optional) initial guess for within-class concentration
        
        returns: the trained model as a PSDA object
        
        
        """
        assert counts.min() > 1, "all speakers need at least 2 observations" 
        means = means.astype(np.float64)
        ns, dim = means.shape
        assert len(counts) == ns
        total = counts.sum()
        psda = cls.em_init(means,w0)
        for i in range(niters):
            psda, llh = psda.em_iter(means,counts,total)
            if not quiet: print(f"em iter {i}: {llh}")
        return psda
    


    def em_iter(self, means, counts, total):
        """
        Invoked by em 
        
        returns: 
            a new updated PSDA
            marginal log-likelihood (em objective)
        """
        zpost = self.zposterior(compose(counts,means))
        llh = self.logCw*total + self.logCb - zpost.logCk.sum() 
    
        z_exp = zpost.mean()
        zbar = z_exp.mean(axis=0)
        between = VMF.max_likelihood(zbar, self.logC)
        r = ((z_exp*means).sum(axis=-1)@counts) / total
        assert 0 < r < 1
        w = self.logC.rhoinv(r)
        return PSDA(w,between), llh
    
    @classmethod
    def em_init(cls,means, w0):
        """
        Invoked by em
        """
        norms, means = decompose(means)
        assert all(norms < 1), "Invalid means"
        between = VMF.max_likelihood(means.mean(axis=0))
        return PSDA(w0, between)
    
def atleast2(means, counts):
    ok = counts > 1
    return means[ok,:], counts[ok]

    
    
class Side:
    """
    Represents a trial side, for one or more observations. When two trial sides
    are scored against each other, one containing m and the other n observations
    an (m,n) llr score matrix is produced.
    
    """
    
    def __init__(self, psda:PSDA, X: ndarray):
        """
        This constructor is invoked by psda.prep(X), see the docs of PSDA. 
        """
        self.logC = psda.logC
        self.logCb = psda.logCb
        self.wX = wX = psda.w*X
        self.wX_norm2 = (wX**2).sum(axis=-1)
        self.pstats = pstats = wX + psda.bmu
        self.pstats_norm2 = pnorm2 = (pstats**2).sum(axis=-1)
        pnorm = np.sqrt(pnorm2)
        self.num = self.logC(pnorm)
        
        
    def llr_matrix(self,rhs):
        """
        Scores the one or more (m) trial sides contained in self against
        all (m) of the trial side(s) in rhs. Returns an (m,n) matrix of
        LLR scores.
        """
        norm2 = self.pstats_norm2.reshape(-1,1) + rhs.wX_norm2 + \
                2*self.pstats @ rhs.wX.T
        denom = self.logC(np.sqrt(norm2))
        return self.num.reshape(-1,1) + rhs.num - denom - self.logCp 
        
    
    
    
        





if __name__ == "__main__":
    
    from numpy.random import randn, randint
    
    
    dim = 10
    b, w = 1, 10

    ns = 10**3
    n = 10**4
         
    
    norm, mu = decompose(randn(dim)) 
    model0 = PSDA(w, VMF(mu, b))
    
    Z = model0.sample_speakers(ns)
    labels = randint(ns,size=(n,))
    uu, labels, counts = np.unique(labels, return_inverse=True, return_counts=True)
    
    Xtrain = model0.sample(Z, labels)
    L = np.full((n,len(counts)),False)   # (n, ns)
    L[np.arange(n),labels] = True
    means = (L.T @ Xtrain) / counts.reshape(-1,1)
    
    
    means, counts = atleast2(means, counts)
    
    model = PSDA.em(means, counts, niters=20)
    
    



    
