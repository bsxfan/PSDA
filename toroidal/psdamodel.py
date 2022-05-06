import numpy as np
from numpy.random import randint

from psda.vmf_sampler import sample_uniform
from subsphere.pca import randStiefel, lengthnorm, retract

from psda.vmf_onedim import gvmf, logNormConst



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
        self.splt = d.cumsum()[:-1]
        
        self.L = self.wstack()
        
        
    def update_w(self,R):
        K = self.K
        w = np.array([(Ri*Ki).sum() for Ri, Ki in zip(R,K)])
        return Embedding(lengthnorm(w),K)
        
    def update_K(self,R):
        w, splt = self.w, self.splt
        F = np.hstack([wi*Ri for wi, Ri in zip(w, R)])
        F = retract(F)
        return Embedding(w,np.hsplit(F,splt))
    
    
    def update(self,R,niters):
        w, splt = self.w, self.splt
        for i in range(niters):
            F = np.hstack([wi*Ri for wi, Ri in zip(w, R)])
            K = np.hsplit(retract(F),splt)
            w = lengthnorm(np.array([(Ri*Ki).sum() for Ri, Ki in zip(R,K)]))
        return Embedding(w,K)    
        
    
        
    def embed(self,Z):
        X =  Z @ self.L.T 
        return X
        
    def project(self,X):
        return X @ self.L
        
    def stack(self):
        return np.hstack(self.K)    
    
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
        self.splt = d.cumsum()[:-1]
        logCd = {di:logNormConst(di) for di in np.unique(d)}
        self.vmf = [gvmf(logCd[di],vi,gammai) \
                    for di,gammai, vi in zip(d,gamma,v)]
        self.gamma_v = np.hstack([gammai*vi for gammai,vi in zip(gamma,v)])    

    def unstack(self,Z):
        return np.hsplit(Z,self.splt)


    def sample(self,n):
        return np.hstack([vmfi.sample(n) for vmfi in self.vmf])





class ToroidalPSDA:
    def __init__(self, kappa, m, w, K, gamma, v):
        assert len(w) == len(K) == len(gamma) == len(v)
        
        self.kappa = kappa
        self.D = D = K[0].shape[0]
        self.logCD = logNormConst(D)
        
        self.n = n = len(K)
        self.m = m
        assert 0 <= m <= n
        
        self.E = Embedding(w,K)
        self.gamma = gamma
        self.v = v


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
     
    def sample_data(self, Z=None, labels=None, Y=None, kappa=None, return_mu=False):
        if Z is not None:
            assert self.m > 0
            Z = self.Ez.embed(Z)
            if labels is not None: 
                Z = Z[labels,:]
        else:
            assert self.m == 0
            assert labels is None
            Z = 0
        if Y is not None:
            assert self.m < self.n
            Y = self.Ey.embed(Y)
        else:  
            assert self.m == self.n
            Y = 0
        Mu = Z + Y
        if kappa is None: kappa = self.kappa
        X = Mu if np.isinf(kappa) else gvmf(self.logCD, Mu,self.kappa).sample()    
        if not return_mu: return X
        return X, Mu
    
    
    def sample(self,n_or_labels, ns = None):
        if np.isscalar(n_or_labels):
            n = n_or_labels
            if self.m > 0:
                labels = np.arange(n) if ns is None else randint(ns,size=(n,))                     
            else:
                assert ns is None
        else:
            assert self.m > 0
            n = len(labels)
        if labels is not None:
            ns = labels.max()+1
            Z = self.sample_speakers(ns)
        else:
            Z = None
        if self.m < self.n:
            Y = self.sample_channels(n)
        else:
            Y = None
        X, Mu = self.sample_data(Z,labels,Y, return_mu=True)    
        return X, Y, Z, Mu, labels    
            
            
            
    
    
    
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
        
    def inferZ(self,Xsum):
        """
        Xsum: (ns,D) first-order stats (data sums) for each of ns speakers 
        
        """
        assert self.zprior is not None
        kappa = self.kappa
        return Posterior(self.zprior,kappa*(self.Ez.project(Xsum)))
        
    def inferY(self,X):
        """
        X: (n,D)   data
        """
        assert self.yprior is not None
        kappa = self.kappa
        return Posterior(self.yprior,kappa*self.Ey.project(X))
    
    
    
    def em_iter(self, X, Xsum, wK_iters = 5):
        """
        X: (n,D)  data
        Xsum: (ns,D) first-order stats (data sums) for each of ns speakers 
        """
        Rz = self.inferZ(Xsum).R(Xsum)
        Ry = self.inferY(X).R(X)
        E = self.E.update([*Rz,*Ry], wK_iters)
        return ToroidalPSDA(self.kappa, self.m, E.w, E.K, self.gamma, self.v)
    
    
class Posterior:
    def __init__(self, prior, stats):
        self.prior = prior
        stats = prior.unstack(stats + prior.gamma_v)
        self.vmf = [gvmf(prior.vmf[i].logC,statsi) \
                    for i, statsi in enumerate(stats)]
        means = [vmfi.mean() for vmfi in self.vmf]
        self.mean = np.hstack(means)
        
    def R(self, X):
        """
        X: (n,D)
        """
        Mu = self.mean      # (n,T)
        R = X.T @ Mu        # (D,T)
        return self.prior.unstack(R)
        
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from subsphere.pca import Globe
    
    from nlib.phonexia.embeddings import one_hot
    
    # if False:                # test and demo data synthesis
    #     D = 3
    #     m, n = 1, 2
    #     snr = 5
    #     w = np.array([np.sqrt(snr),1])
    #     d = np.array([1,2])
    #     gamma_z = np.zeros(m)
    #     gamma_y = np.zeros(n-m)
    #     kappa = 200
        
        
    #     model1 = ToroidalPSDA.random(D, d, w, kappa, gamma_z, gamma_y)
        
    #     Z = model1.sample_speakers(100)
    #     X, Mu = model1.sample(Z,return_mu=True)
    
    #     fig = plt.figure()
    #     ax = fig.add_subplot(221, projection='3d')
    #     Globe.plotgrid(ax)
    #     ax.scatter(*Mu.T, color='r', marker='.',label='Mu')
    #     ax.scatter(*X.T, color='g', marker='.',label='X')
    #     ax.legend()
    #     ax.set_title('Model1: zdim = 1, ydim = 2, snr = 5')
        
    #     snr = 1/5
    #     w = np.array([np.sqrt(snr),1])
    #     model2 = ToroidalPSDA.random(D, d, w, kappa, gamma_z, gamma_y)
        
    #     Z = model2.sample_speakers(100)
    #     X, Mu = model2.sample(Z,return_mu=True)
    
    #     ax = fig.add_subplot(222, projection='3d')
    #     Globe.plotgrid(ax)
    #     ax.scatter(*Mu.T, color='r', marker='.',label='Mu')
    #     ax.scatter(*X.T, color='g', marker='.',label='X')
    #     ax.legend()
    #     ax.set_title('Model2: zdim = 1, ydim = 2, snr = 1')
    
    
    #     snr = 5
    #     w = np.array([np.sqrt(snr),1])
    #     d = np.array([2,1])
    #     gamma_z = np.zeros(m)
    #     gamma_y = np.zeros(n-m)
    #     kappa = 200
        
        
    #     model3 = ToroidalPSDA.random(D, d, w, kappa, gamma_z, gamma_y)
        
    #     Z = model3.sample_speakers(100)
    #     X, Mu = model3.sample(Z,return_mu=True)
    
    #     ax = fig.add_subplot(223, projection='3d')
    #     Globe.plotgrid(ax)
    #     ax.scatter(*Mu.T, color='r', marker='.',label='Mu')
    #     ax.scatter(*X.T, color='g', marker='.',label='X')
    #     ax.legend()
    #     ax.set_title('Model3: zdim = 2, ydim = 1, snr = 5')
    
    
    #     m, n = 2, 3
    #     snr = 20
    #     w = np.array([np.sqrt(snr),np.sqrt(snr),1])
    #     d = np.array([1,1,1])
    #     gamma_z = np.zeros(m)
    #     gamma_y = np.zeros(n-m)
    #     kappa = 200
        
        
    #     model4 = ToroidalPSDA.random(D, d, w, kappa, gamma_z, gamma_y)
        
    #     Z = model4.sample_speakers(100)
    #     X, Mu = model4.sample(Z,return_mu=True)
    
    #     ax = fig.add_subplot(224, projection='3d')
    #     Globe.plotgrid(ax)
    #     ax.scatter(*Mu.T, color='r', marker='.',label='Mu')
    #     #ax.scatter(*X.T, color='g', marker='.',label='X')
    #     ax.legend()
    #     ax.set_title('Model4: zdims = [1,1] ydim = 1, snr = 20')
    
    
    
    #     plt.show()
        
        
    # D = 3
    # m, n = 1,2
    # d = np.array([1,2])        
    # gamma_z = np.zeros(m)
    # gamma_y = np.zeros(n-m)
    # kappa = 200

    # snr = 1
    # w = np.array([np.sqrt(snr),1])
    # model = ToroidalPSDA.random(D, d, w, kappa, gamma_z, gamma_y)
    
    # X, Y, Z, Mu, labels = model.sample(20,10)
    # Ze = model.Ez.embed(Z)

    # Ypost = model.inferY(X)


    # fig = plt.figure()
    # ax = fig.add_subplot(121, projection='3d')
    # Globe.plotgrid(ax)
    # ax.scatter(*X.T, color='g', marker='.',label='X')
    # ax.scatter(*Ze.T, color='r', marker='.',label='Ze')
    # ax.legend()
    # ax.set_title(f'zdim = 1, ydim = 2, snr = 1, kappa={kappa}')
    # ax.set_xlim([-1,1])
    # ax.set_ylim([-1,1])
    # ax.set_zlim([-1,1])

    # ax = fig.add_subplot(122)
    # ax.set_aspect('equal', 'box')
    # Yhat = Ypost.mean
    # ax.scatter(*Y.T,color='g',marker = 'o', label='Y')
    # ax.scatter(*Yhat.T,color='r',marker = 'x', label='Yhat')
    # ax.legend()
    # ax.set_title('y posterior means')
    # ax.set_xlim([-1,1])
    # ax.set_ylim([-1,1])
    # ax.grid()    

    # plt.show()

    

    # #  test 2-d z posterior
    # D = 3
    # m, n = 1,2
    # d = np.array([2,1])        
    # gamma_z = np.zeros(m)
    # gamma_y = np.zeros(n-m)
    # kappa = 2

    # snr = 10
    # w = np.array([np.sqrt(snr),1])
    # model = ToroidalPSDA.random(D, d, w, kappa, gamma_z, gamma_y)
    
    # X, Y, Z, Mu, labels = model.sample(1000,10)
    # Ze = model.Ez.embed(Z)



    # fig = plt.figure()
    # ax = fig.add_subplot(121, projection='3d')
    # Globe.plotgrid(ax)
    # ax.scatter(*X.T, color='g', marker='.',label='X')
    # ax.scatter(*Ze.T, color='r', marker='.',label='Ze')
    # ax.legend()
    # ax.set_title(f'zdim = 1, ydim = 2, snr = 1, kappa={kappa}')
    # ax.set_xlim([-1,1])
    # ax.set_ylim([-1,1])
    # ax.set_zlim([-1,1])

    # plabels, counts = one_hot.pack(labels, return_counts=True)
    # L = one_hot.scipy_sparse_1hot_cols(plabels)
    # Zpost = model.inferZ(X,L)

    # ax = fig.add_subplot(122)
    # ax.set_aspect('equal', 'box')
    # Zhat = Zpost.mean
    # ax.scatter(*Z.T,color='g',marker = 'o', label='Y')
    # ax.scatter(*Zhat.T,color='r',marker = 'x', label='Yhat')
    # ax.legend()
    # ax.set_title('z posterior means')
    # # ax.set_xlim([-1,1])
    # # ax.set_ylim([-1,1])
    # ax.grid()    



    # plt.show()


    #  test 1-d z posterior
    D = 3
    m, n = 2,3
    d = np.array([1,1,1])        
    gamma_z = np.zeros(m)
    gamma_y = np.zeros(n-m)
    kappa = 5

    snr = 0.1
    w = np.array([np.sqrt(snr),np.sqrt(snr),1])
    model = ToroidalPSDA.random(D, d, w, kappa, gamma_z, gamma_y)
    
    X, Y, Z, Mu, labels = model.sample(100,10)
    Ze = model.Ez.embed(Z)



    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    Globe.plotgrid(ax)
    ax.scatter(*X.T, color='g', marker='.',label='X')
    ax.scatter(*Ze.T, color='r', marker='.',label='Ze')
    ax.legend()
    ax.set_title(f'zdim = {d[0]}, ydim = {d[1]}, snr = {snr}, kappa={kappa}')
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])

    plabels, counts = one_hot.pack(labels, return_counts=True)
    L = one_hot.scipy_sparse_1hot_cols(plabels)
    # L: (ns,n)    one-hot label matrix
    Zpost = model.inferZ(L@X)

    ax = fig.add_subplot(122)
    ax.set_aspect('equal', 'box')
    Zhat = Zpost.mean
    # ax.plot(Z,Zhat,'.')
    # ax.scatter(*Zhat.T,color='r',marker = 'x', label='Yhat')
    
    ax.scatter(*Zhat.T,color='r', marker = 'x', label='Zhat')
    ax.scatter(*Z.T,color='g', marker = 'o', label='Z')
    
    ax.legend()
    ax.set_title('z posterior means')
    # ax.set_xlim([-1,1])
    # ax.set_ylim([-1,1])
    ax.grid()    



    plt.show()





    
        