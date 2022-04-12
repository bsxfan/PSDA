"""
MixPSDA: Probabilistic Spherical Discriminant Analysis

"""
import numpy as np
from numpy import ndarray

from psda.vmf import VMF, compose, decompose, LogNormConst


rng = np.random.default_rng()



class MixPSDA:
    """
    Probabilistic Spherical Discriminant Analysis Model

    """

    def __init__(self,
                 p_i:float,
                 within_concentration:float,
                 between_distr:VMF):
        """
        model = MixPSDA(w, VMF(mu, b))

                w,b > 0
                mu (dim, ) is a lengh-normalized speaker mean


                or

        model = MixPSDA(w, VMF(mean))

                w,b > 0,
                mean (dim,) is speaker mean inside (not on) unit hypersphere

                or

        model = MixPSDA.em(means,counts) # see the documention for em()



        """
        self.p_i = p_i
        self.w = w = within_concentration
        self.between = between = between_distr
        self.b = b = between.k
        self.dim = between.dim
        self.mu = between.mu
        self.bmu = between.kmu
        self.logC = logC = between.logC
        self.logCb = logC(b)
        self.logCw = logC(w)

        self.ncomp = len(p_i)
        assert self.mu.shape[0] == self.ncomp, "Expecting {self.ncomp} means."
        self.tied_w = self.w.size < self.ncomp
        self.tied_b = self.b.size < self.ncomp

    def save(self,fname):
        import h5py
        with h5py.File(fname,'w') as h5:
            h5["pi"] = self.p_i
            h5["w"] = self.w
            self.between.save_to_h5(h5,"between")

    @classmethod
    def load(cls,fname):
        import h5py
        with h5py.File(fname,'r') as h5:
            w = np.asarray(h5["w"])
            p_i = np.asarray(h5["pi"])
            w = np.atleast_1d(w)
            p_i = np.atleast_1d(p_i)
            between = VMF.load_from_h5(h5,"between")
        return cls(p_i,w,between)

    # def marg_llh(self, data_sum, count):
    #     """
    #     Computes the marginal log-likelihood log P(X | same speaker), where
    #     z is integrated out. We use: log P(X | z) P(z) / P(z | X)

    #     Returns a vector of independent calculations if data_sum is a matrix
    #     and count is a vector.
    #     """
    #     post = self.zposterior(data_sum)
    #     return self.logCw*count + self.logCb - post.logCk


    def sample_speakers(self, n):
        return self.between.sample(n)


    def sample(self, speakers, component_labels, labels):
        ww = self.w if np.isscalar(self.w) else self.w[component_labels]
        within = VMF(speakers, ww, self.logC)
        return within.sample(labels)


    def prep(self, X: ndarray, counts: ndarray=None):
        """
        Does some precomputation for fast computation of a matrix of LLR
        scores.

            X: vector or matrix of observations (in rows)
               Each row can represent a single observation, or multiple
               observations. For single observation, the row must be on the
               unit hypersphere. For multiple observations, the row must be the
               sum of observations, which can be anywhere in R^d.

               To be clear, when doing multi-enroll trials, the enrollment
               embeddings must be summed, not averaged. Do not length-norm
               again after summing.


            returns: a Side that contains precomputed stuff for fast scoring.
                     The Side provides a method for scoring against another
                     Side.
        """
        if counts is None:
            counts = np.ones(X.shape[0])
        return Side(self,X.astype(np.float64),counts)



    def llr_matrix(self, enroll:ndarray, test:ndarray) -> ndarray:
        """
        Convenience method. See MixPSDA.prep() for details.
        """
        return self.prep(enroll).llr_matrix(self.prep(test))

    def llr_vector(self, enroll:ndarray, test:ndarray) -> ndarray:
        """
        Convenience method. See MixPSDA.prep() for details.
        """
        return self.prep(enroll).llr_vector(self.prep(test))


    @classmethod
    def em(cls, means: ndarray, counts: ndarray,
           niters = 10, w0 = 1.0, quiet = False,
           psda_init = None, fname_tmp=None):
        """
        Trains a MixPSDA model from data.

            means: (n, dim) the means of each of the n classes available for training

            counts: (n,) the number of examples of each class

            niters: the number of EM iterations to run

            w0>0: (optional) initial guess for within-class concentration

        returns: the trained model as a MixPSDA object


        """
        assert counts.min() > 1, "all speakers need at least 2 observations"
        means = means.astype(np.float64)
        ns, dim = means.shape
        assert len(counts) == ns
        total = counts.sum()
        psda = psda_init or cls.em_init(means,w0)
        if not quiet:
            print(f"em init  : 0 B =",psda.b, "W =",psda.w,"mu =",psda.between.mu.ravel()[:6])
        obj = []
        llh0 = 0
        for i in range(niters):
            psda, llh = psda.em_iter(means,counts)
            if fname_tmp is not None:
                psda.save(fname_tmp.format(iter=i))

            impr = llh - llh0; llh0 = llh
            if not quiet:
                print(f"em iter {i}: {impr}","B =",psda.b, "W =",psda.w,"mu =",psda.between.mu.ravel()[:6])
            obj.append(llh)
        return psda, obj

    @classmethod
    def em_init(cls, means, w0=None, b0=None,ncomp=None):
        """
        Invoked by em
        """
        if ncomp is None:
            assert not (w0 is None and b0 is None)
            ncomp = np.atleast_1d(w0).size if b0 is None else np.atleast_1d(b0).size

        pi0 = np.ones(ncomp)/ncomp
        w0 = rng.uniform(100,1000,size=ncomp).astype(float)

        norms, means = decompose(means)
        assert all(norms < 1), "Invalid means"
        ii = np.random.choice(np.arange(means.shape[0]),size=len(w0),replace=False)
        # between1 = VMF.max_likelihood(means)
        b0 = np.ones_like(w0)

        between = VMF(means[ii], b0)
        return MixPSDA(pi0, w0, between)

    # def zposterior(self, data_sum:ndarray):
    def zposterior(self, means:ndarray, counts:ndarray):
        """
        Computes the hidden variable posterior, given the sufficient statistics
        (sum of the observations).

        If mutiple sums (each sum is a vector) are supplied, multiple posteriors
        are computed.

        The posterior(s) one or more are returned in a single VMF object.
        """

        data_sum = compose(counts, means)                 # s x dim

        w = self.w if np.isscalar(self.w) else self.w[:,None,None]
        # (k x 1 x 1)*(k x s x d) + (k x 1 x d)
        theta = w*data_sum[None,...] + self.bmu[:,None,:]
        post = VMF(theta)

        m,n = counts.size, counts.sum()
        # y_exp = post.mean()                             # m x s x dim
        r_tilde = np.log(self.p_i) + self.logCb           # m x 1
        r_tilde = r_tilde[:,None] + np.atleast_1d(self.logCw)[:,None]*counts # m x s
        r_tilde -= post.logCk                           # m x s

        # normalize over components
        log_r = r_tilde - np.logaddexp.reduce(r_tilde,axis=0,keepdims=True)
        gamma = np.exp(log_r)                         # m x s
        assert np.all(np.isfinite(gamma)), "Check gammas!"
        return post, log_r

    def llh(self, means, counts):
        post, log_r = self.zposterior(means,counts)                # m x s x dim
        m,n = counts.size, counts.sum()

        llh = self.logCw*n + self.logCb*m - post.logCk.sum(axis=-1)
        llh += m*np.log(self.p_i) - log_r.sum(axis=1)
        assert np.allclose(llh,llh[0]), "Candidate's formula fail!"
        return llh[0]

    def em_iter(self, means, counts):
        """
        Invoked by em

        m components
        s speakers
        d dimensional embeddings

        returns:
            a new updated MixPSDA
            marginal log-likelihood (em objective)
        """

        p_i, w, b, mu = self.p_i, self.w, self.b, self.mu

        y_post, log_r = self.zposterior(means, counts)
        y_exp = y_post.mean()                             # m x s x dim
        gamma = np.exp(log_r)                             # m x s

        y_bar = np.sum(gamma[:,:,None]*y_exp, axis=1)     # m x dim (spk s summed out)
        y_bar /= gamma.sum(axis=1,keepdims=True)          # m x dim

        pi_new = gamma.sum(axis=1)/gamma.sum()            # m

        if self.tied_b:
            b_new, mu_new = decompose(y_bar)
            b_new = np.atleast_1d(b_new)
            b_new = self.logC.rhoinv(b_new@pi_new)
        else:
            # between_new = VMF.max_likelihood(y_bar, logC=self.logC)
            b_new, mu_new = decompose(y_bar)
            b_new = np.atleast_1d(b_new)
            b_new = self.logC.rhoinv(b_new)
        between_new = VMF(mu_new, b_new,self.logC)

        if self.tied_w:
            warg = np.sum(gamma[:,:,None]*y_exp, axis=0)
            warg = np.sum(compose(counts,means)*warg)/counts.sum()
        else:
            # warg = ((gamma*counts)*(y_exp*means).sum(axis=-1)).sum(axis=-1)
            warg = (gamma*(y_exp*means).sum(axis=-1))@counts
            warg /= (gamma@counts)

        assert np.all(0 < warg) and np.all(warg < 1)
        w_new = self.logC.rhoinv(warg)

        newmod = MixPSDA(pi_new, w_new, between_new)
        llh = newmod.llh(means,counts)
        return newmod, llh


    def __repr__(self):
        return f"MixPSDA(dim={self.dim}, b={self.b}, w={self.w})"

def atleast2(means, counts):
    ok = counts > 1
    return means[ok,:], counts[ok]



class Side:
    """
    Represents a trial side, for one or more observations. When two trial sides
    are scored against each other, one containing m and the other n observations
    an (m,n) llr score matrix is produced.

    """

    def __init__(self, psda:MixPSDA, X: ndarray, counts: ndarray):
        """
        This constructor is invoked by psda.prep(X), see the docs of MixPSDA.
        """
        self.psda = psda
        self.X = X
        self.counts = counts

        self.yi1norm2 = np.sum(X**2,axis=1,keepdims=True)*psda.w**2 + X@(2*psda.bmu.T*psda.w) + psda.b**2
        logr1 = counts[:,None]*psda.logCw + psda.logCb + np.log(psda.p_i)
        logr1 -= psda.logC(self.yi1norm2)
        self.logr1 = np.logaddexp.reduce(logr1, axis=1)


    def llr_matrix(self,rhs):
        """
        Scores the one or more (m) trial sides contained in self against
        all (n) of the trial side(s) in rhs. Returns an (m,n) matrix of
        LLR scores.

        """

        yi3norm2 = self.yi1norm2[:,None,:] + rhs.yi1norm2[None,:,:] - self.psda.b[None,None,:]**2
        yi3norm2 += 2*self.psda.w[None,None,:]*np.sum(self.X[:,None,:]*rhs.X[None,:,:],axis=-1,keepdims=True)

        logr3  = (self.counts[:,None] + rhs.counts)[:,:,None]*self.psda.logCw[None,None,:]
        logr3 += (self.psda.logCb + np.log(self.psda.p_i))[None,None,:]
        logr3 -= self.psda.logC(yi3norm2)
        logr3 = np.logaddexp.reduce(logr3, axis=-1)

        return logr3 - self.logr1[:,None] - rhs.logr1


    def llr_vector(self, rhs):
        """
        Scores the n trial sides contained in self against the respective n
        sides in the rhs. Returns an (n,) vector of LLR scores. If one of the
        sides has a single trial and the other multiple trials, broadcasting
        will be done in the usual way.
        """

        yi3norm2 = self.yi1norm2 + rhs.yi1norm2 - self.psda.b**2
        yi3norm2 += 2*np.sum(self.X*rhs.X,axis=-1,keepdims=True)*self.psda.w

        logr3  = (self.counts + rhs.counts)[:,None]*self.psda.logCw
        logr3 += (self.psda.logCb + np.log(self.psda.p_i))
        logr3 -= self.psda.logC(yi3norm2)
        logr3 = np.logaddexp.reduce(logr3, axis=-1)

        return logr3 - self.logr1 - rhs.logr1
