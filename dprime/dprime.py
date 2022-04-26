import numpy as np
import torch


from psda.vmf import VMF, decompose



def lengthnorm(x):
    return decompose(x)[1]
        

def sumsquaredscores(X):
    n, dim = X.shape
    XX = X @ X.T if n < dim else X.T @ X
    return (XX**2).sum() - n # duplicates are included, but not self-scores



def score(X):
        """
        X: (n,dim) length-normed embeddings
        """
        n,dim = X.shape
        xsum = X.sum(axis=0)
        nscores = n*(n-1)      # duplicates are counted
        scoresum = xsum @ xsum - n  # duplicates are included, but not self-scores
        sqscoresum = sumsquaredscores(X)
        return ScoreStats(nscores, scoresum, sqscoresum)
            

class ScoreStats:
    def __init__(self,s0,s1,s2):
        """
        """
        self.nscores = s0
        self.scoresum = s1
        self.sqscoresum = s2
        
    @classmethod    
    def zero(cls):
        return cls(0,0.0,0.0)
    
    def scatter(self): 
        s0, s1, s2 = self.nscores, self.scoresum, self.sqscoresum
        return (s2-s1**2/s0) / s0
    
    
    def mean(self):
        return self.scoresum / self.nscores
        
        
    def __add__(self,that):
        return ScoreStats(self.nscores + that.nscores,
                          self.scoresum + that.scoresum,
                          self.sqscoresum + that.sqscoresum)
    
    def __sub__(self,that):
        return ScoreStats(self.nscores - that.nscores,
                          self.scoresum - that.scoresum,
                          self.sqscoresum - that.sqscoresum)

    def __repr__(self):
        n = self.nscores
        mu = self.mean()
        std = np.sqrt(self.scatter())
        return f"ScoreStats: {n} scores at {mu} +- {std}"        
    


class XStats:
    def __init__(self,x0,x1,x2, tar = None):
        self.x0 = x0
        self.x1 = x1
        self.x2 = x2
        self.tar = self.score() if tar is None else tar
        
    @classmethod    
    def zero(cls, dim):
        x0 = 0
        x1 = np.zeros(dim)
        x2 = np.zeros((dim,dim))
        tar = ScoreStats.zero()
        return cls(x0,x1,x2,tar)
        
        
    def __add__(self, X):
        x0 = X.shape[0]
        x1 = X.sum(axis=0)
        x2 = X.T @ X        
        that = XStats(x0,x1,x2)
        return XStats(self.x0+x0, self.x1+x1, self.x2+x2, self.tar+that.tar )
    
    @classmethod
    def accumulate(cls, X, labels):
        dim = X.shape[-1]
        stats = cls.zero(dim)
        for i in np.unique(labels):
            stats = stats + X[labels==i,:]
        return stats


    def sumsquaredscores(self):
        n, XX = self.x0, self.x2
        return (XX**2).sum() - n # duplicates are included, but not self-scores
        
    def sumscores(self):
        n, xsum = self.x0, self.x1
        return xsum @ xsum - n # duplicates are included, but not self-scores

    def nscores(self):
        n = self.x0
        return n*(n-1) # duplicates are included, but not self-scores

    def score(self):
        s0 = self.nscores()
        s1 = self.sumscores()
        s2 = self.sumsquaredscores()
        return ScoreStats(s0,s1,s2)
    
    def nontar(self):
        mixed = self.score()
        return mixed - self.tar


    def dprime(self):
        tar, non = self.tar, self.nontar()
        std = np.sqrt((tar.scatter()+non.scatter())/2)
        return (tar.mean() - non.mean()) / std 
     
        
     
    def obj(self,temp=1/100):
        """
        Minimization-friendly objective. Approaches -log(dprime) when temp>0 is 
        small. 
        
        input:
            
            temp >= 0
        
            - Any temp > 0 prevents the objective from blowing up
              when dprime is near zero.
            
            - Any temp > 2 prevents the objective from returning NaN when
              dprime is negative.
              
            - 0 < temp << 1 approaches -log(dprime)
            
        returns to-be-minimized objective    

        """
        tar, non = self.tar, self.nontar()
        num = np.log((tar.scatter()+non.scatter())/2) 
        den = np.log1p((tar.mean() - non.mean())/temp) + np.log(temp)
        return num-den

    def __repr__(self):
        return f"XStats:\n  tar: {self.tar}\n  non: {self.nontar()} "        



def score_analysis(X, labels):
    tar = ScoreStats(0,0,0)
    for i in np.unique(labels):
        tar = tar + score(X[labels==i,:])
    mixed = score(X)
    non = mixed - tar
    return tar, non


def dprime(tar: ScoreStats, non:ScoreStats):
    std = np.sqrt((tar.scatter()+non.scatter())/2)
    return (tar.mean() - non.mean()) / std 

    
    
if __name__ == "__main__":


    from numpy.random import randn, randint
    
    print('--- Testing basics ---')
    n, dim = 3, 5
    X = lengthnorm(randn(n, dim))
    s = score(X)
    
    G = np.triu(X@X.T,1)
    Ones = np.triu(np.ones_like(G),1)
    
    
       
    print(s.nscores,2*Ones.sum())
    print(s.scoresum,2*G.sum())
    print(s.sqscoresum,2*(G**2).sum())
    
    mu = G.sum() / Ones.sum()
    print(s.mean(),mu)
    
    v = (np.triu(G-mu,1)**2).sum() / Ones.sum()
    print(s.scatter(),v)
    
    
    print('\n\n--- Testing dprime ---')
    n, dim = 10000, 100
    ns = 1000
    labels = randint(ns,size=(n,))
    Z = randn(ns,dim)
    X = lengthnorm(Z[labels,:]+randn(n, dim))
    
    dp1 = dprime(*score_analysis(X, labels))
    
    XS = XStats.accumulate(X, labels)
    dp2 = XS.dprime()
    
    print(dp1,dp2)
    
    
    
    

    