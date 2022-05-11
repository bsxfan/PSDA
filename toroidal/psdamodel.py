import numpy as np
from numpy.random import randint
from scipy.optimize import minimize_scalar

from psda.vmf_sampler import sample_uniform
from subsphere.pca import randStiefel, lengthnorm, retract

from psda.vmf_onedim import gvmf, logNormConst

from psda.vmf_map import map_estimate as vmf_map_estimate
from psda.vmf_map import ml_estimate as vmf_ml_estimate
from psda.vmf_map import KappaPrior_KL



class Embedding:
    """
    This is one of the components contained in the model.
    
    This is a function from a product space, where the hidden variables live,
    to the Euclidean representation R^D where the observations live.
    
    Each hidden variable is projected into R^D by one of the factor loading 
    matrices and the function output is the weighted sum of those projections.
    
    This object is used to represent the function from all the hidden variables
    to S^{D-1}, but also for the speaker and channel subsets, in which case the 
    function outputs are inside the ball, not on S^{D-1}. We reach S^{D-1} only 
    if all speaker and channel factors are added. 
    
    
    """
    def __init__(self,w,K):
        """
        w: (n,) weights whose squares sum to either 1, or less than 1
        K: list of length n, containing mutually orthogonal factor loading 
           matrices, in general with differing numbers of columns. Each matrix 
           has D rows.
        """

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
        """
        Given statistics in list R, does an M-step update for w when K is fixed
        """
        K = self.K
        w = np.array([(Ri*Ki).sum() for Ri, Ki in zip(R,K)])
        return Embedding(lengthnorm(w),K)
        
    def update_K(self,R):
        """
        Given statistics in list R, does an M-step update for K when w is fixed
        """
        w, splt = self.w, self.splt
        F = np.hstack([wi*Ri for wi, Ri in zip(w, R)])
        F = retract(F)
        return Embedding(w,np.hsplit(F,splt))
    
    
    def update(self, R, niters):
        """
        Given statistics in list R, does an M-step update for K and w.
        """
        w, splt = self.w, self.splt
        for i in range(niters):
            F = np.hstack([wi*Ri for wi, Ri in zip(w, R)])
            K = np.hsplit(retract(F),splt)
            tilde_w = np.array([(Ri*Ki).sum() for Ri, Ki in zip(R,K)])
            w = lengthnorm(tilde_w)
        Q = w @ tilde_w  # we need this to update kappa
        return Embedding(w,K), Q    
        
    
        
    def embed(self,Z):
        """
        This is the forward function from hidden variables to observation space
        
        input: 
            
            Z: (t,T) horizontally stacked hidden variables, t of them. The sum of
                 hiddeb variable dimensions is T.
                 
        output:
            
            X: (t,D) function output
                 
        """
        X =  Z @ self.L.T 
        return X
        
    def project(self,X):
        """
        project backwards, from observation space to stacked hidden variable 
        space
        
        input: X (t,D)
        output: Z: (t,T)
        
        """
        return X @ self.L
        
    def stack(self):
        """
        returns matrix of horizontally stacked factor loading matrices (F)
        F should have orthonormal columns
        """
        return np.hstack(self.K)    
    
    def wstack(self):
        """
        returns matrix of horizontally stacked weighted factor loading matrices
        """
        return np.hstack([wi*Ki for wi,Ki in zip(self.w,self.K)])

    @classmethod
    def unstack(cls,F,d):
        """
        splits F into a list of matrices, each with dimension as given in d
        """
        assert d.sum() == F.shape[-1]
        return np.hsplit(F,d.cumsum()[:-1])
    
    @classmethod
    def random(cls,w,D,d):
        """
        Generates a random Embedding object, given: 
            weights w, (n,) 
            number of rows, D, common to all matrices
            d: (n,) number of columns of each matrix
            
            we need d.sum() <= D
            
            w does not need to be length-normalized, it is done here
            
            
        """
        assert len(d) == len(w)
        T = d.sum()
        assert T <= D
        w = lengthnorm(w)
        K = cls.unstack(randStiefel(D,T),d)
        return cls(w,K)
    
    
    
    
class Prior:
    """
    This is another component contained in the model, representing a collection
    of hidden variables.
    """
    def __init__(self,gamma,v):
        """
        
        gamma: (n,) VMF concentration for eah of n hidden variables 
            
        v: list of n mean directions, one for each hidden variable
           the dimensionalities can differ
               
        """
        self.d = d = np.array([len(vi) for vi in v])
        self.splt = d.cumsum()[:-1]
        logCd = {di:logNormConst(di) for di in np.unique(d)}
        self.vmf = [gvmf(logCd[di],vi,gammai) \
                    for di,gammai, vi in zip(d,gamma,v)]
        self.gamma_v = np.hstack([vmfi.kmu for vmfi in self.vmf])

    def unstack(self,Z):
        """
        Z: (t,T) set of t stacked hidden variabes
        
        returns a list with n unstacked matrices, each with t rows 
        """
        return np.hsplit(Z,self.splt)


    def sample(self,t):
        """
        returns a matrix, (t,Z) of horizntally stacked, sampled hidden 
        variables
        """
        return np.hstack([vmfi.sample(t) for vmfi in self.vmf])





class ToroidalPSDA:
    """
    Toroidal PSDA model, with sampling, inference (scoring) and learning 
    functionality.
    
    
    
    
    
    """
    def __init__(self, kappa, m, w, K, gamma, v):
        """
        kappa: vmf concentration, for within-speaker noise in S^{D-1}
        
        m: number of hidden speaker variables (can be zero for factor analysis)
        
        w: (n, ) unit-vector of factor weights, one for each hidden variable
                 We need n >= m, so that n is the total number of speaker
                 and channel variables. If 1 <= m = n, there are no channel 
                 variables. The channel variables, if present, contribute
                 structured within-speaker noise in addition to that provided
                 by kappa. The m speaker variables provide between-speaker 
                 variability.
                 
        K: list of n factor laoding matrices, one for each hidden variable

        gamma: (n,) VMF prior concentrations for each of the n hidden variables
        
        v: list of VMF prior mean directions for each of the n hidden variables
         
                 
        """
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
        
        
        
            
            
    def sample_speakers(self,ns):
        """
        Sample ns independent speakers, each represented by the stacked hidden
        speaker variables. The output is in hidden variable product space, not
        in observation space.
        """
        
        assert self.zprior is not None, "this model has no speaker factors"
        Z = self.zprior.sample(ns)
        return Z

    def sample_channels(self,t):
        """
        Sample t independent channels, each represented by the stacked hidden
        channel  variables. The output is in hidden variable product space, not
        in observation space.
        """
        if self.yprior is None: return None
        Y = self.yprior.sample(t)
        return Y
     
    def sample_data(self, Z=None, labels=None, Y=None, kappa=None, return_mu=False):
        """
        Sample observations, given labels and hidden variables.
        
        inputs:
            
            Z: (ns, Tz) stacked speaker variables, for ns speakers, or None if this
                        model does not have speaker variables
            
    
            Y: (t, Ty) stacked channel variables, for t observations, or None if 
                       this model does not have channel variables
    
            labels: (t,) speaker labels in 0..ns-1
                    if None, one sample is drawn for each speaker and/or channel 
                    
            kappa: (optional) can be used to override the model's kappa
    
            return_mu: (optional, default False)
                       if True, return X and Mu, where Mu the sum of projected hidden
                       variables, before adding the final VMF noise. 


        output:
            
            X: (t, D)  observations


        """
        
        if Z is not None:
            assert self.m > 0
            Z = self.Ez.embed(Z)
            if labels is not None: 
                Z = Z[labels,:]
            elif Y is not None:
                assert Z.shape[0] == Y.shape[0]
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
        X = Mu if np.isinf(kappa) else gvmf(self.logCD, Mu, kappa).sample()    
        if not return_mu: return X
        return X, Mu
    
    
    def sample(self, t_or_labels, ns = None):
        """
        General model sampling function for hidden variables, optionally labels
        and observations.
        
        
        inputs:
            
            t_or_labels: 
                if scalar t is provided, there will be t observations
                if speaker labels are provided, t = len(labels)
                if speaker labels are not proved, but ns is provided, 
                   labels are sampled
                if neither labels nor ns are provided, labels are 0..t-1
                
                When there are no speakers, provide t, not labels.
                
            ns: number of speakers
                optional, can be inferred from labels, or if labels are not
                          provided, then ns <- t
                
                
        output: X, Y, Z, Mu, labels
                
            X: (t,D) observations
            Z: (t,Tz) or None
            Y: (t,Ty) or None
            Mu: linear combination of hidden variables on the toroid in 
                observation space
            labels: (t,), as provided or sampled/constructed here    
            
            
        """
        if np.isscalar(t_or_labels):
            t = t_or_labels
            if self.m > 0:
                labels = np.arange(t) if ns is None else randint(ns,size=(t,))                     
            else:
                assert ns is None
        else:
            assert self.m > 0
            t = len(labels)
        if labels is not None:
            ns = labels.max()+1
            Z = self.sample_speakers(ns)
        else:
            Z = None
        if self.m < self.n:
            Y = self.sample_channels(t)
        else:
            Y = None
        X, Mu = self.sample_data(Z,labels,Y, return_mu=True)    
        return X, Y, Z, Mu, labels    
            
            
            
    
    
    
    @classmethod
    def random(cls, D, d, w, kappa, gamma_z = None, gamma_y = None):
        """
        Construct a model with some given and some random parameters.
        """
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
        Compute speaker factor posteriors.

            Xsum: (ns,D) first-order stats (data sums) for each of ns speakers 
        
        """
        assert self.zprior is not None
        kappa = self.kappa
        return Posterior(self.zprior,kappa*(self.Ez.project(Xsum)))
        
    def inferY(self,X):
        """
        Compute channel factor posteriors.
        
            X: (n,D)   data
        """
        assert self.yprior is not None
        kappa = self.kappa
        return Posterior(self.yprior,kappa*self.Ey.project(X))
    
    
    
    def em_iter(self, X, Xsum, wK_iters = 5, zprior = None, 
                                             yprior = None,
                                             kappa_prior = None):
        """
        Do one EM iteration and return a new updated model.
        
            X: (n,D)  data
        
            Xsum: (ns,D) first-order stats (data sums) for each of ns speakers 
            
            zprior: 
                if False, the VMF priors for the z's are not updated
                if None, the VMF priors are ML-updated
                if zprior is a list of priors for the concentrations, do MAP
                   updates
            
            yprior: 
                if False, the VMF priors for the y's are not updated
                if None, the VMF priors are ML-updated
                if yprior is a list of priors for the concentrations, do MAP
                   updates
            
            kappa_prior: 
                if False, kappa is not updated
                if None, kappa is ML-updated
                if kappa_prior a prior for kappa, do a MAP update
        """
        zPost = self.inferZ(Xsum)
        yPost = self.inferY(X)
        Rz = zPost.R(Xsum)
        Ry = yPost.R(X)
        E, Q = self.E.update([*Rz,*Ry], wK_iters)
        
        N = X.shape[0]
        if kappa_prior is not False:
            kappa = self.update_kappa(N,Q, kappa_prior)
        else:
            kappa = self.kappa    
        
        do_zprior = zprior is not False
        do_yprior = yprior is not False
        
        if do_zprior:
            vz, gammaz = zPost.prior_update(zprior)
        else:
            vz = self.v[:self.m]      
            gammaz = self.gamma[:self.m]      
        
        if do_yprior:
            vy, gammay = yPost.prior_update(yprior)
        else:
            vy = self.v[self.m:]      
            gammay = self.gamma[self.m:]
        v = [*vz,*vy]
        gamma = np.hstack([gammaz,gammay])
        
        return ToroidalPSDA(kappa, self.m, E.w, E.K, gamma, v)
    
    
    def ml_update_kappa(self,N,Q):
        logkappa = np.log(self.kappa)
        logCD = self.logCD
        def f(logk):
            k = np.exp(logk)
            return -N*logCD(logk=logk) - k*Q
        res = minimize_scalar(f,[logkappa,logkappa-1])
        logkappa = res.x
        return np.exp(logkappa)
    
    def update_kappa(self, N, Q, kappa_prior = None):
        if kappa_prior is None: return self.ml_update_kappa(N,Q)
        logkappa = np.log(self.kappa)
        logCD = self.logCD
        def f(logk):
            k = np.exp(logk)
            return -N*logCD(logk=logk) - k*Q - kappa_prior.loglh(k)
        res = minimize_scalar(f,[logkappa,logkappa-1])
        logkappa = res.x
        return np.exp(logkappa)
    
class Posterior:
    def __init__(self, prior, stats):
        self.prior = prior
        stats = prior.unstack(stats + prior.gamma_v)
        self.vmf = [gvmf(prior.vmf[i].logC,statsi) \
                    for i, statsi in enumerate(stats)]
        means = [vmfi.mean() for vmfi in self.vmf]
        self.mean = mean = np.hstack(means)
        self.n = mean.shape[0]
        self.sums = [vmfi.mean().sum(axis=0) for vmfi in self.vmf]
        
    def R(self, X):
        """
        X: (n,D)
        """
        Mu = self.mean      # (n,T)
        R = X.T @ Mu        # (D,T)
        return self.prior.unstack(R)
    
    
    def ml_prior_update(self):
        V = []
        Gamma = []
        n = self.n
        for sumi, vmfi in zip(self.sums, self.prior.vmf):
            v, gamma = vmf_ml_estimate(n, sumi, np.log(vmfi.k), vmfi.logC)    
            V.append(v)
            Gamma.append(gamma)
        return V, np.array(Gamma)    
            
    def prior_update(self, kappa_priors = None):
        if kappa_priors is None: return self.ml_prior_update()
        V = []
        Gamma = []
        n = self.n
        for sumi, vmfi, pi in zip(self.sums, self.prior.vmf, kappa_priors):
            v, gamma = vmf_map_estimate(n, sumi, pi, vmfi.logC, np.log(vmfi.kappa))    
            V.append(v)
            Gamma.append(gamma)
        return V, np.array(Gamma)    
        
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





    
        