import numpy as np
from numpy.random import randn

from scipy.linalg import eigh

from psda.vmf import VMF, decompose



def lengthnorm(x):
    return decompose(x)[1]
        


def invsqrtm(C):
    e, V = eigh(C) # C = (V*e) @ V.T
    return (V / np.sqrt(e)) @ V.T


class UnitSphere:
    def __init__(self, D):
        """
        D is the enclosing Euclidean dimension. The sphere has dimension D-1
        """
        self.D = D
        
        
        
    def align(self, X, Z):
        """
        Given: 
            
            X: points in S^{D-1} and 
            Z: corresponding points in S^{d-1}
            
        Find the orientation matrix F (D,d) that aligns them and return
        the induced subsphere.
            
        
        X: (n,D)
        Z: (n, d)
        
        """
        n, D = X.shape
        assert D == self.D
        nz, d = Z.shape
        assert 2 <= d < D and n==nz
        
        R = X.T @ Z    # (D,d)
        F =  R @ invsqrtm(R.T @ R)  # (D,d) @ (d,d)
        return ConcentricSubsphere(F)
    
    
    def randomCCSub(self, d):
        D = self.D
        assert 2 <= d < D
        R = randn(D,d)
        F =  R @ invsqrtm(R.T @ R)  # (D,d) @ (d,d)
        return ConcentricSubsphere(F)
        

    def sampleVMF(self, n_or_mu, kappa = 0):
        """
        sampleVMF(n) returns n uniform samples in S^{D-1}
        sampleVMF(mu, kappa) returns one sample for every row of mu
        """
        
        if np.isscalar(n_or_mu):
            n = n_or_mu
            assert kappa == 0
            return VMF(self.D).sample(n)
        else:
            mu = n_or_mu
            D = mu.shape[-1]
            assert D == self.D
            return VMF(mu, kappa).sample()

        
        
        
class ConcentricSubsphere:
    def __init__(self, F):
        """
        F: (D,d), with 2 <= d < D and F'F = I_d
           F determines the orientation of the subsphere 
        """
        self.F = F
        self.D, self.d = F.shape



    def project(self, X):
        """
        project D-dimensional data from the enclosing unitpshere onto this 
        subsphere
        
        X: (n,D), or (D,)
        """
        F = self.F  # (D,d)        
        return lengthnorm(X @ F)
        
    
    def represent(self,Z):
        """
        Given length-normalized input(s) of dimension d, represent it as (a)
        point(s) of dimension D in the enclosing unitsphere
        
        
        Z: (n,d), or (d,), length-normalized
        """
        F = self.F  # (D,d) 
        return Z @ F.T
    
    
    
def PCA(X, d, niters=10, quiet = False):
    n,D = X.shape
    assert 2 <= d < D
    U = UnitSphere(D)

    Z = lengthnorm(randn(n, d))
    S = U.align(X, Z)

    for i in range(niters):
        Z = S.project(X)
        
        if not quiet:
            Y = S.represent(Z)
            obj = (X*Y).sum() / n
            print(f"PCA {i}: {obj}")
        
        S = U.align(X, Z)
        
    return S


if __name__ == "__main__":
    
    
    D, d = 3, 2
    n = 200
    
    Ud = UnitSphere(d)
    UD = UnitSphere(D)
    
    S0 = UD.randomCCSub(d)
    Z = Ud.sampleVMF(n)
    Y = S0.represent(Z)
    
    X = UD.sampleVMF(Y,20)
    S = PCA(X,d)
    


    
    
    
    
    
    
    