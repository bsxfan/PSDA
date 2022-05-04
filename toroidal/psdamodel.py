import numpy as np

from psda.vmf_sampler import sample_uniform
from subsphere.pca import randStiefel, lengthnorm

from psda.vmf_onedim import vmf, logNormConst

class ToroidalPSDA:
    def __init__(self, kappa, m, w, K, gamma, v):
        self.kappa = kappa
        self.w = w
        self.K = K
        self.gamma = gamma
        self.v = v
        
        self.D = D = K[0].shape[0]
        self.n = n = len(K)
        self.m = m
        assert 0 <= m <= n
        self.d = d = np.array([Ki.shape[1] for Ki in K])
        self.T = T = d.sum()
        self.Tz = Tz = d[:m].sum()
        self.Ty = Ty = d[m:].sum()
        assert T <= D
        self.F = F = np.zeros((D,T))
        at = 0
        for Ki in K:
            Di, di = Ki.shape
            assert Di == D
            F[:,at:at+di] = Ki
            at += di
            
        if m>0:
            self.Lz = Lz = np.zeros((D,Tz))
            at = 0
            for Ki, wi in zip(K[:m],w[:m]):
                Di, di = Ki.shape
                Lz[:,at:at+di] = wi*Ki
                at += di

        if m<n:
            self.Ly = Ly = np.zeros((D,Ty))
            at = 0
            for Ki, wi in zip(K[m:],w[m:]):
                Di, di = Ki.shape
                Ly[:,at:at+di] = wi*Ki
                at += di


            
        self.logCD = logNormConst(D)    
        logCd = {di:logNormConst(di) for di in np.unique(d)}
        self.prior = prior = [vmf(logCd[len(vi)],vi,gammai) \
                              for gammai, vi in zip(gamma,v)]
        self.zprior = prior[:m]    
        self.yprior = prior[m:]    
            
            
    def sample_speakers(self,n):
        assert self.m > 0, "this model has no speaker factors"
        Tz = self.Tz
        Z = np.zeros((n, Tz))
        at = 0
        for vmfi in self.zprior:
            di = vmfi.dim
            Z[:,at:at+di] = vmfi.sample(n)
            at += di
        return Z

    def sample_channels(self,n):
        Ty = self.Ty
        if Ty == 0: return None     # no channel factors
        Y = np.zeros((n, Ty))
        at = 0
        for vmfi in self.yprior:
            di = vmfi.dim
            Y[:,at:at+di] = vmfi.sample(n)
            at += di
        return Y
     
    def sample(self, Z_or_n, labels = None, kappa = None, return_mu = False):
        if self.m > 0:
            Z = Z_or_n
            ns,Tz = Z.shape
            assert Tz == self.Tz
            Z = Z @ self.Lz.T
            if labels is not None: 
                Z = Z[labels,:]
            n = Z.shape[0]    
            Y = self.sample_channels(n) @ self.Ly.T
            Mu = Z if Y is None else Z + Y
        else:  # m = 0
            assert labels is None
            assert np.isscalar(Z_or_n)
            n = Z_or_n
            Mu = self.sample_channels(n) @ self.Ly.T
        if kappa is None: kappa = self.kappa
        X = Mu if np.isinf(kappa) else vmf(self.logCD, Mu,self.kappa).sample()    
        if not return_mu: return X
        return X, Mu
    
    
    
    @classmethod
    def random(cls, D, d, w, kappa, gamma_z = None, gamma_y = None):
        if gamma_z is None:
            assert gamma_y is not None
            gamma = gamma_y
            m = 0
        elif gamma_y is None:
            assert gamma_z is not None
            gamma = gamma_z
            m = len(gamma)
        else:
            gamma = np.hstack([gamma_z,gamma_y])
            m = len(gamma_z)
        n = len(gamma)    
        assert len(d) == n
        T = d.sum()
        assert T <= D
        F = randStiefel(D,T)
        w = lengthnorm(w)
        K = []
        v = []
        at = 0
        for di in d:
            K.append(F[:,at:at+di])
            v.append(sample_uniform(di))
            at += di
        return cls(kappa,m,w,K,gamma,v)    
    
class Embedding:
    def __init__(self,w,K):

        assert len(w) == len(K)
        self.n = len(w)

        self.D = D = K[0].shape[0]
        assert all([Ki.shape[0]==D for Ki in K])

        self.w = w
        self.K = K
        
        self.d = d = np.array([Ki.shape[1] for Ki in K])
        self.T = d.sum()
        
    def stack(self):
        return np.hstack([self.K])    
    
    def wstack(self):
        return np.hstack([wi*Ki for wi,Ki in zip(self.w,self.K)])

    @classmethod
    def unstack(cls,F,d):
        return np.hsplit(F,d.cumsum()[:-1])
    
    @classmethod
    def random(cls,w,D,d):
        assert len(d) == len(w)
        T = d.sum()
        assert T <= D
        w = lengthnorm(w)
        K = cls.unstack(randStiefel(D,T),d)
        return cls(w,K)
    
    
class Prior:
    def __init__(self,gamma,v):
        self.d = d = np.array([len(vi) for vi in v])
        logCd = {di:logNormConst(di) for di in np.unique(d)}
        self.vmf = [vmf(logCd[di],vi,gammai) \
                    for di,gammai, vi in zip(d,gamma,v)]
        self.gamma_v = np.hstack([gammai*vi for gammai,vi in zip(gamma,v)])    

    def sample(self,n):
        return np.hstack([vmfi.sample(n) for vmfi in self.vmf])

    
    
class Posterior:
    def __init__(self, prior, stats):
        stats = stats + prior.gamma_v
        s = prior.d.cumsum()[:-1]
        stats = np.hsplit(stats,s)
        self.vmf = [vmf(prior.vmf[i].logC,statsi) \
                    for i, statsi in enumerate(stats)]
        self.means = [vmfi.mean() for vmfi in self.vmf]
        self.mean = np.hstack(self.means)
        
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from subsphere.pca import Globe
    
    
    D = 3
    m, n = 1, 2
    snr = 5
    w = np.array([np.sqrt(snr),1])
    d = np.array([1,2])
    gamma_z = np.zeros(m)
    gamma_y = np.zeros(n-m)
    kappa = 200
    
    
    model1 = ToroidalPSDA.random(D, d, w, kappa, gamma_z, gamma_y)
    
    Z = model1.sample_speakers(100)
    X, Mu = model1.sample(Z,return_mu=True)

    fig = plt.figure()
    ax = fig.add_subplot(221, projection='3d')
    Globe.plotgrid(ax)
    ax.scatter(*Mu.T, color='r', marker='.',label='Mu')
    ax.scatter(*X.T, color='g', marker='.',label='X')
    ax.legend()
    ax.set_title('Model1: zdim = 1, ydim = 2, snr = 5')
    
    snr = 1/5
    w = np.array([np.sqrt(snr),1])
    model2 = ToroidalPSDA.random(D, d, w, kappa, gamma_z, gamma_y)
    
    Z = model2.sample_speakers(100)
    X, Mu = model2.sample(Z,return_mu=True)

    ax = fig.add_subplot(222, projection='3d')
    Globe.plotgrid(ax)
    ax.scatter(*Mu.T, color='r', marker='.',label='Mu')
    ax.scatter(*X.T, color='g', marker='.',label='X')
    ax.legend()
    ax.set_title('Model2: zdim = 1, ydim = 2, snr = 1')


    snr = 5
    w = np.array([np.sqrt(snr),1])
    d = np.array([2,1])
    gamma_z = np.zeros(m)
    gamma_y = np.zeros(n-m)
    kappa = 200
    
    
    model3 = ToroidalPSDA.random(D, d, w, kappa, gamma_z, gamma_y)
    
    Z = model3.sample_speakers(100)
    X, Mu = model3.sample(Z,return_mu=True)

    ax = fig.add_subplot(223, projection='3d')
    Globe.plotgrid(ax)
    ax.scatter(*Mu.T, color='r', marker='.',label='Mu')
    ax.scatter(*X.T, color='g', marker='.',label='X')
    ax.legend()
    ax.set_title('Model3: zdim = 2, ydim = 1, snr = 5')


    m, n = 2, 3
    snr = 20
    w = np.array([np.sqrt(snr),np.sqrt(snr),1])
    d = np.array([1,1,1])
    gamma_z = np.zeros(m)
    gamma_y = np.zeros(n-m)
    kappa = 200
    
    
    model4 = ToroidalPSDA.random(D, d, w, kappa, gamma_z, gamma_y)
    
    Z = model4.sample_speakers(100)
    X, Mu = model4.sample(Z,return_mu=True)

    ax = fig.add_subplot(224, projection='3d')
    Globe.plotgrid(ax)
    ax.scatter(*Mu.T, color='r', marker='.',label='Mu')
    #ax.scatter(*X.T, color='g', marker='.',label='X')
    ax.legend()
    ax.set_title('Model4: zdims = [1,1] ydim = 1, snr = 20')



    plt.show()
        
        


    
        