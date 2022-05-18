import numpy as np
from numpy.random import randint
from scipy.optimize import minimize_scalar

from psda.vmf_sampler import sample_uniform
from subsphere.pca import randStiefel, lengthnorm, retract

from psda.vmf_onedim import gvmf, logNormConst

from psda.vmf_map import map_estimate as vmf_map_estimate
from psda.vmf_map import ml_estimate as vmf_ml_estimate
from psda.vmf_map import KappaPrior_KL, logkappa_asymptote_intersection

from nlib.phonexia.embeddings import one_hot



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
        self.uniform = all(np.atleast_1d(gamma)==0)
        
    def margloglh_term(self):
        return sum([vmfi.logCk for vmfi in self.vmf])

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


    def gammaprior_loglh(self, gamma_prior):
        return sum([pi.loglh(vi.k) for pi, vi in zip(gamma_prior, self.vmf)])


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

        self.hasspeakers = hasspeakers = m>0
        self.haschannels = haschannels = m<n
        
        if hasspeakers:
            self.Ez = Embedding(w[:m],K[:m])
            self.zprior = Prior(gamma[:m],v[:m])
        else:
            self.zprior = None
            self.Ez = None


        if haschannels:
            self.Ey = Embedding(w[m:],K[m:])
            self.yprior = Prior(gamma[m:],v[m:])
        else:
            self.yprior = None
            self.Ey = None
        
    def margloglh_term(self, t, ns=None):
        hasspeakers, haschannels = self.hasspeakers, self.haschannels
        assert hasspeakers or ns is None
        loglh = t*self.logCD(self.kappa)
        if hasspeakers:
            loglh += ns*self.zprior.margloglh_term()
        if haschannels:
            loglh += t*self.yprior.margloglh_term()
        return loglh
    
            
            
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
            assert self.hasspeakers
            Z = self.Ez.embed(Z)
            if labels is not None: 
                Z = Z[labels,:]
            elif Y is not None:
                assert Z.shape[0] == Y.shape[0]
        else:
            assert not self.hasspeakers
            assert labels is None
            Z = 0
        if Y is not None:
            assert self.haschannels
            Y = self.Ey.embed(Y)
        else:  # Y is None
            assert not self.haschannels
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
    def random(cls, D, d, m, w=None, kappa=None, gamma=None):
        """
        Construct a model with some given and some random parameters.
        
        inputs:
            
            D: observation dimensionality

            d: (n,) hidden variable dimensionalities

            m: number of speaker variables: 0 <= m <= n
            
            w: (n,) positive hidden variable combination weights
                    does not need to be length-normed, this will be done here
                    if None, weights are made uniform
                    
            kappa > 0: observation VMF noise concentration
                       if None, will be set to a somewhat concentrated value
                       
            gamma: (n,) concentrations for hidden variables
                        negative values are 'flagged' and will be reset to 
                            somewhat dispersed (smallish positive) values
                        if None all gammas will be treated as flagged and be 
                            reset as above    
                        
                
                
        """
        kappa0 = lambda dim: np.exp(logkappa_asymptote_intersection(dim))
        
        n = len(d)
        assert 0 <= m <= n
        assert d.sum() <= D
        if w is None:
            w = np.ones(n)
        if kappa is None:
            kappa = 10*kappa0(D)
        if gamma is None: gamma = np.full(n, -1.0)
        for i in range(n):    
            if gamma[i] < 0: gamma[i] = kappa0(d[i])/10 
        assert n == len(w) == len(gamma)
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
    
    def margloglh(self, X, Xsum=None, labels = None):
        hasspeakers = self.hasspeakers
        haschannels = self.haschannels
        assert hasspeakers or haschannels
        assert hasspeakers or Xsum is None
        
        ns = None if Xsum is None else Xsum.shape[0]
        t = X.shape[0]
        obj = self.margloglh_term(t,ns)
        
        if hasspeakers:
            if Xsum is None:
                assert labels is not None
                Xsum = sumX(X, labels)
            zPost = self.inferZ(Xsum)
            obj -= zPost.margloglh_term()
            
        if haschannels:
            yPost = self.inferY(X)
            obj -= yPost.margloglh_term()

        return obj 
            
    
    
    def em_iter(self, X, Xsum = None, wK_iters = 5, kappa_prior = None,
                                                    gammaz_prior = None, 
                                                    gammay_prior = None):
        """
        Do one EM iteration and return a new updated model.
        
        Inputs:
        
            X: (n,D)  data
        
            Xsum: (ns,D) first-order stats (data sums) for each of ns speakers 
            
            kappa_prior: 
                if False, kappa is not updated
                if None, kappa is ML-updated
                if kappa_prior a prior for kappa, do a MAP update

            gammaz_prior: 
                if False, the VMF priors for the z's are not updated
                if None, the VMF priors are ML-updated
                if gammaz_prior is a list of priors for the concentrations, do MAP
                   updates
            
            gammay_prior: 
                if False, the VMF priors for the y's are not updated
                if None, the VMF priors are ML-updated
                if gammay_prior is a list of priors for the concentrations, do MAP
                   updates
            
                
                
        returns: updated ToroidalPSDA        
        """
        hasspeakers = self.hasspeakers
        haschannels = self.haschannels
        hasboth = hasspeakers and haschannels
        assert hasspeakers or haschannels
        
        ns = None if Xsum is None else Xsum.shape[0]
        t = X.shape[0]
        obj = self.margloglh_term(t,ns)
        
        
        if kappa_prior is not None and kappa_prior is not False:
            obj += kappa_prior.loglh(self.kappa)
        
        if hasspeakers:
            assert Xsum is not None
            zPost = self.inferZ(Xsum)
            Rz = zPost.R(Xsum)
            obj -= zPost.margloglh_term()
            if gammaz_prior is not None and gammaz_prior is not False:
                obj += self.zprior.gammaprior_loglh(gammaz_prior)
            
        if haschannels:
            yPost = self.inferY(X)
            Ry = yPost.R(X)
            obj -= yPost.margloglh_term()
            if gammay_prior is not None and gammay_prior is not False:
                obj += self.zprior.gammaprior_loglh(gammay_prior)
        
        if hasboth:
            E, Q = self.E.update([*Rz,*Ry], wK_iters)
        elif hasspeakers:
            E, Q = self.E.update(Rz, wK_iters)
        else:  # haschannels
            E, Q = self.E.update(Ry, wK_iters)
        
        
        do_kappa = kappa_prior is not False
        do_zprior = self.hasspeakers and gammaz_prior is not False
        do_yprior = self.haschannels and gammay_prior is not False
        
        if do_kappa:
            N = X.shape[0]
            kappa = self.update_kappa(N,Q, kappa_prior)
        else:
            kappa = self.kappa    
        
        if do_zprior:
            vz, gammaz = zPost.prior_update(gammaz_prior)
        else:
            vz = self.v[:self.m]      
            gammaz = self.gamma[:self.m]      
        
        if do_yprior:
            vy, gammay = yPost.prior_update(gammay_prior)
        else:
            vy = self.v[self.m:]      
            gammay = self.gamma[self.m:]

        v = [*vz,*vy]
        gamma = np.hstack([gammaz,gammay])
        return ToroidalPSDA(kappa, self.m, E.w, E.K, gamma, v), obj
    
    
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
    
    
    @classmethod
    def TrainEM(cls, initial_model, niters, X, Xsum = None,  
                                      kappa_prior = None,
                                      gammaz_prior = None, 
                                      gammay_prior = None,
                                      quiet = False):
        
        model = initial_model
        obj = []

        if not model.hasspeakers or model.zprior.uniform: 
            assert gammaz_prior is False or gammaz_prior is None
            gammaz_prior = False
        
        if not model.haschannels or model.yprior.uniform: 
            assert gammay_prior is False or gammay_prior is None
            gammay_prior = False
        
        
        for i in range(niters):
            model, mllh = model.em_iter(X, Xsum, 
                                  kappa_prior = kappa_prior,
                                  gammaz_prior = gammaz_prior, 
                                  gammay_prior = gammay_prior)
            if not quiet:
                print(f"em {i}: {mllh}")
            obj.append(mllh)    
        return model, obj        
        
    
class Scoring:
    def __init__(self, model: ToroidalPSDA, fast = False):
        kappa = model.kappa
        self.m = m = model.m
        self.fast = fast
        assert m >= 1
        self.zprior = zprior = model.zprior
        self.uniform = uniform  = zprior.uniform
        self.P = P = kappa*model.Ez.L      # (D, Tz),   kappa*wi*Ki, stacked
        self.Plst = Plst = self.unstack(P) # [(D, di)], kappa*wi*Ki, unstacked
        if not uniform:
            nvec = zprior.unstack(zprior.gamma_v)
            self.Pn = np.array([Pi @ nveci \
                                 for nveci,Pi in zip(nvec,Plst)])      #(m,D)  
            self.nnormsq = np.array([nveci @ nveci for nveci in nvec]) #(m,)
            self.priorden = self.sumlogC(self.nnormsq)          # scalar
        else:    
            self.priorden = self.sumlogC(np.zeros(model.m))     # scalar
            
    def sumlogC(self, ksq):
        """
        input: ksq: k**2   (m, t)
        output: (t,)
        """
        m, fast = self.m, self.fast
        vmf = self.zprior.vmf
        logk = np.log(ksq)/2
        if m==1: return vmf[0].logC(logk=logk, fast=fast)
        logk = np.atleast_2d(logk)
        s = vmf[0].logC(logk=logk[0,:], fast=fast)
        for i in range(1,m):
            s += vmf[i].logC(logk=logk[i,:], fast=fast)
        return s    


    def logC(self, logk, i):
        return self.zprior.vmf[i].logC(logk=logk, fast=self.fast)


    
    def unstack(self, M):
        return self.zprior.unstack(M)
            
        
    def side(self,Xsum):
        """
        Xsum: (t,D)
        """
        return Side(self,Xsum)
        

class Side:
    def __init__(self, sc:Scoring, Xsum:np.ndarray):
        """
        Xsum: (t, D)
        """
        self.sc = sc
        self.t = Xsum.shape[0]
        self.m = sc.m
        self.uniform = uniform = sc.uniform
        self.XP = XP = sc.unstack(Xsum @ sc.P)             # [(t,di)]
        self.normsq = np.array([(XPi**2).sum(axis=-1) \
                                for XPi in XP])           # (m,t)
        if not uniform:
            self.normsq0 = self.normsq + 2*(sc.Pn @ Xsum.T)      # (m,t) + (m,t)
            self.normsq = self.normsq0 + sc.nnormsq.reshape(-1,1)# (m,t) + (m,1) 
        else:
            self.normsq0 = self.normsq                           # (m,t)    
        self.num = sc.sumlogC(self.normsq)                       # (t,)


    def postdenMatrix(self,rhs,i):
        left = self.normsq[i,:].reshape(-1,1)
        right = rhs.normsq0[i,:]
        ksq = 2*(self.XP[i] @ rhs.XP[i].T) + left + right
        logk = np.log(ksq) / 2
        return self.sc.logC(logk,i)

    def postden(self,rhs,i):
        left = self.normsq[i,:]
        right = rhs.normsq0[i,:]
        ksq = (self.XP[i] * rhs.XP[i]).sum(axis=-1)*2 + left + right
        logk = np.log(ksq) / 2
        return self.sc.logC(logk,i)



    def llrMatrix(self,rhs):
        llr = self.num.reshape(-1,1) + rhs.num - self.sc.priorden
        for i in range(self.m):
            llr -= self.postdenMatrix(self,rhs,i)
        return llr    
        
    def llr(self,rhs):
        llr = (self.num + rhs.num - self.sc.priorden).ravel()
        for i in range(self.m):
            llr -= self.postden(rhs,i)
        return llr    
            
    
    
class Posterior:
    """
    Stores parameters for the VMF posteriors for a set of hidden variables. 
    - makes available some posterior expectations 
    - can do M-step update for the hidden variable priors
    """
    def __init__(self, prior:Prior, stats:np.ndarray):
        self.prior = prior
        stats = prior.unstack(stats + prior.gamma_v)
        self.vmf = [gvmf(prior.vmf[i].logC,statsi) \
                    for i, statsi in enumerate(stats)]
        means = [vmfi.mean() for vmfi in self.vmf]
        self.mean = mean = np.hstack(means)
        self.n = mean.shape[0]
        self.sums = [vmfi.mean().sum(axis=0) for vmfi in self.vmf]
        
        
    def margloglh_term(self):
        return sum([vmfi.logCk.sum() for vmfi in self.vmf])
        
        
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
            
    def prior_update(self, gamma_priors = None):
        if gamma_priors is None: return self.ml_prior_update()
        V = []
        Gamma = []
        n = self.n
        for sumi, vmfi, pi in zip(self.sums, self.prior.vmf, gamma_priors):
            v, gamma = vmf_map_estimate(n, sumi, pi, vmfi.logC, np.log(vmfi.k))    
            V.append(v)
            Gamma.append(gamma)
        return V, np.array(Gamma)    
    


def sumX(X,labels):
    if labels is not None:
        plabels, counts = one_hot.pack(labels, return_counts=True)
        L = one_hot.scipy_sparse_1hot_cols(plabels)
        Xsum = L @ X
    else:
        Xsum = None
    return Xsum    

    
def train_ml(d, m, niters, X, labels = None, uniformz = False, uniformy = False, 
             initial_model = None, quiet=False):
    """
    """
    D = X.shape[-1]
    n = len(d)
    assert d.sum() <= D

    Xsum = sumX(X,labels)         


    if initial_model is None:
        gamma = np.full(n,-1.0)
        if m>0 and uniformz:
            gamma[:m] = 0
        if m<n and uniformy:
            gamma[m:] = 0
        model = ToroidalPSDA.random(D,d,m,gamma=gamma)
    else:
        model = initial_model

    model, obj = ToroidalPSDA.TrainEM(model, niters, X, Xsum, quiet=quiet)
    return model, obj
    

def train_map(d, m, niters, X, labels = None, 
              kappa_prior = None ,
              gammaz_prior = None,
              gammay_prior = None,
              quiet=False):
    """
    inputs:
        
        d: (n,) hidden factor dims 

        m: number of speaker factors

        niters: number of EM iterations

        X: (t,D) training data

        labels: speaker labels, 
                None if there are no speaker factors
                
        kappa_prior: prior on VMF concentration, for D-dim observation noise
                     can be None for ML estimate of this parameter
                     
        gammaz_prior: list of priors on VMF concentrations for the speaker
                      factors 
                      if None, zero concentrations are enforced, for uniform
                               distributions
                
        gammay_prior: list of priors on VMF concentrations for the channel
                      factors 
                      if None, zero concentrations are enforced, for uniform
                               distributions
    """

    D = X.shape[-1]
    n = len(d)
    assert d.sum() <= D

    Xsum = sumX(X,labels)         

    kappa = None if kappa_prior is None else kappa_prior.rep

    if m == 0: assert gammaz_prior is None
    if m == n: assert gammay_prior is None
    gamma = np.zeros(n)
    if gammaz_prior is not None:
        gamma[:m] = [pi.rep for pi in gammaz_prior]
    if gammay_prior is not None:
        gamma[m:] = [pi.rep for pi in gammay_prior]

    model = ToroidalPSDA.random(D,d,m,kappa=kappa,gamma=gamma)
    model, obj = ToroidalPSDA.TrainEM(model, niters, X, Xsum, kappa_prior, 
                                 gammaz_prior, gammay_prior, quiet)
    return model, obj
        
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from subsphere.pca import Globe
    
    
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
    kappa = 5

    snr = 0.1
    w = np.array([np.sqrt(snr),np.sqrt(snr),1])
    model = ToroidalPSDA.random(D, d, m, w, kappa)
    
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





    
        