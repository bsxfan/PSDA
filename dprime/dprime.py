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

    def scatter(self): 
        s0, s1, s2 = self.nscores, self.scoresum, self.sqscoresum
        return (s2-s1**2/s0) / s0
    
    
    def mean(self):
        return self.scoresum / self.nscores
        
        
    def __add__(self,that):
        return ScoreStats(self.nscores + that.nscores,
                          self.scoresum + that.scoresum,
                          self.sqscoresum + that.sqscoresum)
    

    def __repr__(self):
        return f"ScoreStats({self.nscores}, {self.scoresum}, {self.sqscoresum})"        
    
    
    
    
if __name__ == "__main__":


    from numpy.random import randn
    
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
    
    
    

    