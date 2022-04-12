import numpy as np
from numpy.random import randn

from scipy.linalg import eigh, svd, sqrtm, solve

from psda.vmf import VMF, decompose



def lengthnorm(x):
    return decompose(x)[1]
        


def invsqrtm(C):
    """
    Inverse, symmetric matrix square root of a symmetric, positive definite 
    matrix. The result is symmetric positive definite.
    """
    e, V = eigh(C) # C = (V*e) @ V.T
    assert all(e>0), "input must be possitive definite"
    return (V / np.sqrt(e)) @ V.T


def retract_eigh(R):
    """
    Project tall matrix to the Stiefel manifold, so that it has orthonormal 
    columns. The projection maximizes trace(input.T @ output), subject to the 
    constraint. This is equivalent to minimzing Euclidean distance: the 
    projection is the closest point on the manifold.
    
    input:
        
        R: (D,d) with D >= d and the rank must be full (= d).
        
    output: F (D,d), such that F'F = I_d 
    
    
    This method is the fastest of the retract methods in this module.
    
    """
    if R.ndim==1: return lengthnorm(R)
    return R @ invsqrtm(R.T @ R)  # (D,d) @ (d,d)
    


def retract_svd(R):
    """
    The result is mathematically equivalent to retract_eigh(R). See the 
    documentation for that function.
    
    This method is about 2x slower that retract_eigh, when using the default 
    scipy eigh and svd. Nevertheless, it is attractive becasue of its relative
    simplicity and more direct computation. If we ever need to autodiff backprop 
    through this function, it is possible this one may be preferable ...
    
    """
    if R.ndim==1: return lengthnorm(R)
    U, s, Vt = svd(R, full_matrices=False)
    return U@Vt

def retract_sqrtm(R):
    """
    The result is mathematically equivalent to retract_eigh. See the 
    documentation for that function.
    
    This variant has no obvious advantages and is also slowest.
    """
    if R.ndim==1: return lengthnorm(R)
    return solve(sqrtm(R.T@R), R.T).T


retract = retract_eigh


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
        the induced concentric subsphere.
            
        
        X: (n,D)
        Z: (n,d)
        
        """
        n, D = X.shape
        assert D == self.D
        nz, d = Z.shape
        assert 2 <= d < D and n==nz
        
        F = retract(X.T @ Z)    # (D,d)
        return ConcentricSubsphere(F)
    
    
    
    
    
    
    def randomConcentricSubsphere(self, d):
        D = self.D
        assert 1 <= d < D
        F = retract(randn(D,d))
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
        self.D, self.d = D, d = F.shape
        self.Ud = UnitSphere(d)
        self.UD = UnitSphere(D)



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
        
        Input: 

            Z: (n,d), or (d,), length-normalized
            
            
        Output: (n,D), or (D,) length-normalized    
        """
        F = self.F  # (D,d) 
        return Z @ F.T
    
    
    def sample(self, n, kappa=None):
        """
        If kappa is None: generate n samples on the subsphere
        
        If kappa > 0: genarate n samples on the subsphere and then replace 
                      each sample, s, with a sample from VMF(s, kappa).
                      This amounts to sampling from a factor-analysis model.
        """
        Ud, UD = self.Ud, self.UD
        Z = Ud.sampleVMF(n)
        Y = self.represent(Z)
        if kappa is None: return Y
        return UD.sampleVMF(Y,kappa)
        
    
class Subsphere(ConcentricSubsphere):
    """
    Generalization of ConcentricSubsphere to allow a subsphere center away from 
    the enclosing unitsphere center and a subsphere radius of less than one. 
    """
    def __init__(self,F,c,theta=None):
        """
        F: (D,d) with orhtonormal columns, orientatio matrix
        c: (D,) c'c=1, unit direction vector of subsphere center
        theta: angle, such that subsphere radius = cos(theta)
                      and center is at sin(theta) * c
                      
               (optional) if not given, c is taken as subsphere center with 
                                        c'c < 1       
                                        
               Note: theta is the angle between the unitshpere and subsphere 
                     radii                         
                                        
        """
        super().init(F)
        if theta is None:
            k = c
            s, c = decompose(k)
            theta = np.arcsin(s)
        self.c = c
        self.theta = theta    # angle between unitshpere and subsphere radii
                                        
        self.r = np.cos(theta)         # subsphere radius
        self.s = s = np.sin(theta)
        self.k = s*c                   # subsphere center
        
    def represent(self,Z):
        """
        Given length-normalized input(s) of dimension d, represent it as (a)
        point(s) of dimension D in the enclosing unitsphere
        
        Input: 

            Z: (n,d), or (d,), length-normalized
            
            
        Output: (n,D), or (D,) length-normalized    
        """
        return self.r*super().represent(Z) + self.k
    

    def relocate(self, X, xbar, Z):
        """
        This is a generalization of align. Given data and
        temporarily fixed Z, it:
            - updates theta, given current values for F and c
            - updates F and c, given the new theta
            
        returns a new Subsphere, with the updated values    
        """
        Rbar = (X.T@Z)/X.shape[0]  #(D,d)
        theta = np.atan2(self.c@xbar, (self.F*Rbar).sum())
        r, s = np.cos(theta), np.sin(theta)
        D,d = Rbar.shape
        Fbrev = retract(np.hstack([r*Rbar,s*xbar.reshape(-1,1)]))
        F, c = Fbrev[:,:-1], Fbrev[:,-1]
        return Subsphere(F, c, theta)




    
def PCA(X, d, niters=10, quiet = False):
    n,D = X.shape
    assert 2 <= d < D
    U = UnitSphere(D)

    #S = U.randomConcentricSubsphere(d)   # this works too
    Z = lengthnorm(randn(n, d))
    S = U.align(X, Z)               # this seems to give a better start

    for i in range(niters):
        Z = S.project(X)
        
        if not quiet:
            Y = S.represent(Z)
            obj = (X*Y).sum() / n
            print(f"PCA {i}: {obj}")
        
        S = U.align(X, Z)
        
    return S


if __name__ == "__main__":
    
    
    D, d = 256, 128
    n = 2000
    
    Ud = UnitSphere(d)
    UD = UnitSphere(D)
    
    
    # create a factor analysis model
    S0 = UD.randomConcentricSubsphere(d)
    kappa = 20
    #kappa = None
    
    # sample from it
    print('sampling')
    X = S0.sample(n, kappa)
    
    print('\nPCA')
    S = PCA(X,d)
    


    
    
    
    
    
    
    