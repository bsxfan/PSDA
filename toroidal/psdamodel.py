import numpy as np

from psda.vmf_sampler import sample_uniform
from subsphere.pca import randStiefel, lengthnorm

from psda.vmf_onedim import vmf, logNormConst

class ToroidalPSDA:
    def __init__(self, kappa, m, w, K, gamma, v):
        assert len(w) == len(K) == len(gamma) == len(v)
        
        self.kappa = kappa
        self.logCD = logNormConst(D)
        
        self.D = K[0].shape[0]
        self.n = n = len(K)
        self.m = m
        assert 0 <= m <= n
        
        self.E = Embedding(w,K)


        if m>0:
            self.Ez = Embedding(w[:m],K[:m])
            self.zprior = Prior(gamma[:m],v[:m])
        else:
            self.zprior = None
            self.Ez = None


        if m<n:
            self.Ey = Embedding(w[m:],K[m:])
            self.yprior = Prior(gamma[m:],v[m:])
        else:
            self.yprior = None
            self.Ey = None
        
        
        
            
            
    def sample_speakers(self,n):
        assert self.zprior is not None, "this model has no speaker factors"
        Z = self.zprior.sample(n)
        return Z

    def sample_channels(self,n):
        if self.yprior is None: return None
        Y = self.yprior.sample(n)
        return Y
     
    def sample(self, Z_or_n, labels = None, kappa = None, return_mu = False):
        if self.m > 0:
            Z = Z_or_n
            ns,Tz = Z.shape
            assert Tz == self.Ez.T
            Z = self.Ez.embed(Z)
            if labels is not None: 
                Z = Z[labels,:]
            n = Z.shape[0]    
            Y = self.sample_channels(n)
            Mu = Z if Y is None else Z + self.Ey.embed(Y)
        else:  # m = 0
            assert labels is None
            assert np.isscalar(Z_or_n)
            n = Z_or_n
            Mu = self.Ey.embed(self.sample_channels(n))
        if kappa is None: kappa = self.kappa
        X = Mu if np.isinf(kappa) else vmf(self.logCD, Mu,self.kappa).sample()    
        if not return_mu: return X
        return X, Mu
    
    
    
    @classmethod
    def random(cls, D, d, w, kappa, gamma_z = None, gamma_y = None):
        assert gamma_z is not None or gamma_y is not None
        if gamma_z is None:
            gamma = gamma_y
            m = 0
        elif gamma_y is None:
            gamma = gamma_z
            m = len(gamma)
        else:
            gamma = np.hstack([gamma_z,gamma_y])
            m = len(gamma_z)
        n = len(gamma)    
        assert len(d) == n
        v = [sample_uniform(di) for di in d]
        E = Embedding.random(w,D,d)
        return cls(kappa,m,E.w,E.K,gamma,v)    
        
        
    
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
        
        self.L = self.wstack()
        
        
    def embed(self,Z):
        X =  Z @ self.L.T 
        return X
        
    def project(self,X):
        return X @ self.L
        
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
        self.stats = stats + prior.gamma_v
        s = prior.d.cumsum()[:-1]
        stats = np.hsplit(stats,s)
        self.vmf = [vmf(prior.vmf[i].logC,statsi) \
                    for i, statsi in enumerate(stats)]
        means = [vmfi.mean() for vmfi in self.vmf]
        self.mean = np.hstack(means)
        
        
        
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
        
        


    
        