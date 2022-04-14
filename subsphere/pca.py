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
    
            If d = D, then F is orthogonal and we have: 
                F' = inv(F), so that also FF' = I_d 
    
    
    This method is the fastest of the retract methods in this module.
    
    """
    if R.ndim==1: return lengthnorm(R)
    return R @ invsqrtm(R.T @ R)  # (D,d) @ (d,d)
    


def retract_svd(R):
    """
    The result is mathematically equivalent to retract_eigh(R). See the 
    documentation for that function.
    
    This method is usually slower that retract_eigh, when using the default 
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


def randStiefel(D,d):
    return retract(randn(D,d))

def randorth(D):
    return randStiefel(D,D)



class UnitSphere:
    def __init__(self, d):
        """
        d is the enclosing Euclidean dimension. The sphere has dimension d-1
        """
        self.d = d
        
    def __repr__(self):
        name = type(self).__name__
        return f"{name} in R^{self.d}"
        
    def subsphere(self,F,c=None,theta=None):
        if c is None:
            assert theta is None
            return ConcentricSubsphere(self,F)
        return Subsphere(self, F, c, theta)    
    
        
        
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
        assert D == self.d
        nz, d = Z.shape
        assert 1 <= d < D and n==nz
        
        F = retract(X.T @ Z)    # (D,d)
        return self.subsphere(F)
    
    def locate(self, X, xbar, Z, theta):
        """
        This is a generalization of align. 
        
        Given data and temporarily fixed Z and theta, it updates F and c and
        returns a new Subsphere, with the given theta and the updated F, c.    
        """
        n, D = X.shape
        assert D == self.d
        nz, d = Z.shape
        assert 1 <= d < D and n==nz
        
        Rbar = (X.T@Z)/n  #(D,d)
        r, s = np.cos(theta), np.sin(theta)
        Fbrev = retract(np.hstack([r*Rbar,s*xbar.reshape(-1,1)]))
        F, c = Fbrev[:,:-1], Fbrev[:,-1]
        return self.subsphere(F, c, theta)
    
    
    
    
    
    def randomConcentricSubsphere(self, d):
        """
        d: the new subsphere dimension is d-1
        """
        D = self.d
        assert 1 <= d <= D
        return self.subsphere(randStiefel(D,d))
        

    def randomSubsphere(self, d, theta=0):
        """
        d: the new subsphere dimension is d-1
        
        theta: angle between radii of the embedding unitsphere and the 
               new subsphere. Also cos(theta) is the subsphere radius and 
               sin(theta) is the distance between the unitsphere and subsphere 
               centers.
               
               If the subsphere is a circle of constant latitude (e.g. equator, 
               polar circle), then theta is the latitude (e.g. 0 for the 
               equator). 
               

        """
        if theta==0: return self.randomConcentricSubsphere(d)
        D = self.d
        assert 1 <= d < D
        Fbrev = randStiefel(D,d+1)
        F, c = Fbrev[:,:-1], Fbrev[:,-1]
        return self.subsphere(F, c, theta)




    def sampleVMF(self, n_or_mu, kappa = 0):
        """
        sampleVMF(n) returns n uniform samples in S^{d-1}
        sampleVMF(mu, kappa) returns one sample for every row of mu
        """
        
        if np.isscalar(n_or_mu):
            n = n_or_mu
            assert kappa == 0
            return VMF(self.d).sample(n)
        else:
            mu = n_or_mu
            d = mu.shape[-1]
            assert d == self.d
            return VMF(mu, kappa).sample()
        

class Globe(UnitSphere):
    def __init__(self):
        super().__init__(3)
        
        
    @classmethod
    def latitude(cls,lat,long=None):
        c = np.array([0,0,1.0])
        F = np.vstack([np.eye(2),np.zeros(2)])
        S = cls().subsphere(F, c, lat)               
        if long is None: return S
        if np.isscalar(long):
            long = np.linspace(0,2*np.pi,long)
        Z = np.vstack([np.cos(long),np.sin(long)]).T
        return S.represent(Z)

        
    @classmethod
    def meridian(cls,long,lat=None):
        F = np.vstack(([np.cos(long),np.sin(long),0],[0,0,1])).T
        S = cls().subsphere(F)               
        if lat is None: return S
        if np.isscalar(lat):
            lat = np.linspace(0,2*np.pi,lat)
        Z = np.vstack([np.cos(lat),np.sin(lat)]).T
        return S.represent(Z)


    @classmethod
    def plotgrid(cls, ax, fine=True):
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])        
        if fine:
            lat = np.array([-75,-60,-45,-30,-15,0,15,30,45,60,75])*np.pi/180
            long = np.array([0,15,30,45,60,75,90,105,120,135,150,165])*np.pi/180
        else:
            lat = np.array([-60,-30,0,30,60])*np.pi/180
            long = np.array([0,30,60,90,120,150])*np.pi/180
        for lati in lat:
            ax.plot(*cls.latitude(lati,100).T,color='gray')
        for longi in long:
            ax.plot(*cls.meridian(longi,100).T,color='gray')
        

        
        
class ConcentricSubsphere(UnitSphere):
    """
    Objects of this class have a dual nature. They are: 
        : (d-1)-dimensional Unitspheres
        : and subspheres of the emclosing (parent) (D-1)-dimensional Unitsphere
    These two spaces are isomorphic: length and angles are preserved by the
    bijection x = F z, with inverse z = F' x. (This linear map is only a 
    bijection if the domain and codomain are the respective spheres noted above.)                                   

    When this class is seen as a UnitSphere, it is the (d-1)-dimensional one
    where z lives. When seen as a ConcentricSubsphere, it is the (d-1)-dim 
    embedded subsphere where x lives. 
    
    The attribute self.UD refers to the enclosing (D-1)-dim Unitsphere (x lives 
    in a subspace of it.)
    
    The attribute self.Ud = self refers to the (d-1)-dim Unitsphere, where z
    lives.
                                                          
                                                          
                                                          
    
    """
    def __init__(self, parent, F):
        """
        F: (D,d), with 2 <= d < D and F'F = I_d
           F determines the orientation of the subsphere 
        """
        D, d = F.shape
        assert D == parent.d and 1 <= d <= D
        super().__init__(d)   # sets self.d = d
        self.F = F
        self.D = D
        self.Ud = self
        self.UD = parent

    def __repr__(self):
        name = type(self).__name__
        D, d = self.D, self.d
        return f"{name}: UnitSphere in R^{d}, also embedded in UnitSphere in R^{D}"



    def project(self, X):
        """
        Project D-dimensional data from the enclosing unitpshere onto this 
        subsphere. A representation in the (d-1)-dim Unitshpere is returned. 
        
        Input: X: (n,D), or (D,)
        Output: Z(n,D), or (d,), length-normalized
        
        
        
        """
        assert X.shape[-1] == self.D
        F = self.F  # (D,d)        
        Z = lengthnorm(X @ F)
        return Z
        
    
    def represent(self,Z):
        """
        Given length-normalized input(s) of dimension d, represent it as (a)
        point(s) of dimension D in the enclosing unitsphere
        
        Input: 

            Z: (n,d), or (d,), length-normalized
            
            
        Output: X (n,D), or (D,) length-normalized    
        """
        F = self.F  # (D,d) 
        return Z @ F.T
    
    def roundtrip(self, X):
        return self.represent(self.project(X))


    
    def sample(self, n, kappa=np.inf):
        """
        If kappa is infinite: generate n samples, exactly on the subsphere
        
        If kappa > 0: genarate n samples on the subsphere and then replace 
                      each sample, s, with a sample from VMF(s, kappa).
                      This amounts to sampling from a factor-analysis model.
        """
        Ud, UD = self.Ud, self.UD
        Z = Ud.sampleVMF(n)
        X = self.represent(Z)
        if np.isfinite(kappa):
            X = UD.sampleVMF(X,kappa)
        return X    
        
    
class Subsphere(ConcentricSubsphere):
    """
    Generalization of ConcentricSubsphere to allow a subsphere center away from 
    the enclosing unitsphere center and a subsphere radius of less than one. 
    """
    def __init__(self,parent, F,c,theta=None):
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
        super().__init__(parent, F)
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
        F, c = self.F, self.c
        assert F.shape == Rbar.shape
        theta = np.arctan2(c@xbar, (F*Rbar).sum())
        r, s = np.cos(theta), np.sin(theta)
        Fbrev = retract(np.hstack([r*Rbar,s*xbar.reshape(-1,1)]))
        F, c = Fbrev[:,:-1], Fbrev[:,-1]
        return self.UD.subsphere(F, c, theta)




    
def concentricPCA(X, d, niters=10, quiet = False):
    n,D = X.shape
    assert 1 <= d < D
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


def eccentricPCA(X, d, theta_init, niters=10, quiet = False, clamp_theta = False):
    n,D = X.shape
    assert 1 <= d < D
    U = UnitSphere(D)
    xbar = X.mean(axis=0)
    
    #S = U.randomSubsphere(d)   # this may work too
    Z = lengthnorm(randn(n, d))
    S = U.locate(X, xbar, Z, theta_init)

    for i in range(niters):
        Z = S.project(X)
        
        if not quiet:
            Y = S.represent(Z)
            obj = (X*Y).sum() / n
            print(f"PCA {i}: {obj}")
        
        if clamp_theta:
            S = U.locate(X, xbar, Z, theta_init)
        else:            
            S = S.relocate(X, xbar, Z)
        
    return S


class LinearSubspace:
    def __init__(self,F):
        """
        Input: 
            
           F: (D,d), D >= d is a representative of the subspace, which is 
              spanned by the columns of F. F must be of full rank (=d), i.e. 
              its columns must be linearly independent.
           
        After construction, self.F will have orthonormal columns: F.T@F = I 

           
        """
        if F.ndim==1: F = F.reshape(-1,1)
        self.D, self.d = D,d = F.shape
        assert 1 <= d <= D
        self.F = retract(F)

    @classmethod
    def random(cls,D,d):
        return cls(randn(D,d))


    def compare(self, that=None):
        """
        Compares this subspace with another. 
        
        The compared subspaces must agree in the dimension of the enclosing 
        space, but need not agree in their own dimension. 
        
        The result is the (soft) number of dimensions in which they agree.
        - self.compare(self) is self.d  i.e. own dimension, the maximum
        - comparison between orthogonal subspaces gives 0, the minimum
        
        If G is a subspace of F, then F.compare(G) = G.d
        
        The comparison is symmetric and is also invariant to orthogonal 
        transforms of the subspace representative, self.F. In fact, if 
        self.d = that.d, the comparison returns:

            max_{U,V} = trace[ (self.F @ U).T @ (that.F @ V) ]
        
        where U,V are orthogonal of appropriate sizes. No explicit search for 
        U, V is required. The solution is given by SVD of the of the (small)
        product:
        
            self.compare(that) = sum_of_singular_values (self.F.T @ that.F)
        
        
        
        """
        assert self.D == that.D
        if that is None: return self.d
        R = self.F.T @ that.F
        s = svd(R,compute_uv=False)
        return s.sum()
    
    


    def __matmul__(self,M):
        return type(self)(self.F @ M)


    def __getitem__(self,args):
        return type(self)(self.F.__getitem__(args))
    
    
    def __repr__(self):
        return "LinearSubspace represented by:\n" + str(self.F)


if __name__ == "__main__":
    
    
    # D, d = 256, 128
    # n = 2000
    
    # Ud = UnitSphere(d)
    # UD = UnitSphere(D)
    
    
    # # create a factor analysis model
    # S0 = UD.randomConcentricSubsphere(d)
    # kappa = 20
    # #kappa = None
    
    # # sample from it
    # print('sampling')
    # X = S0.sample(n, kappa)
    
    # print('\nPCA')
    # S = concentricPCA(X,d)
    
    D, d = 3, 2
    n = 200
    
    UD = UnitSphere(D)
    
    
    # create a factor analysis model
    S0 = UD.randomSubsphere(d,np.pi/6)
    kappa = 100
    #kappa = None
    
    # sample from it
    print('sampling')
    X = S0.sample(n, kappa)
    
    print('\nPCA')
    S = eccentricPCA(X, d, 0.1, niters=10)

    
    
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap
    
    #cmap = get_cmap('Spectral')
    #cc = [cmap(s/ns) for s in labels]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Globe.plotgrid(ax)
    # ax.scatter(*UD.sampleVMF(300).T, color='k', marker='.', label='uniform on unitsphere')
    ax.scatter(*X.T, color='r', marker='.',label='data close to subsphere')
    ax.scatter(*S.roundtrip(X).T, color='g', marker='.',label='learnt subsphere')
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.legend()
    fig.show()    
    
    
    
    