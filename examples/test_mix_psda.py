
import os,sys

import numpy as np
from numpy.random import randn, randint

import matplotlib.pyplot as plt

from pyllr import quick_eval
from psda.mix_psda import VMF, MixPSDA, decompose, atleast2

rng = np.random.default_rng()



def generate_psda(dim,ns,ntrain,ntest):
    from psda.psda import PSDA

    b, w = 50, 100  # within, between concentrations

    # set up model to sample from
    norm, mu = decompose(randn(dim))
    model0 = PSDA(w, VMF(mu, b))
    print(f"true     : 0","B =",b, "W =",w,"mu =",mu.ravel()[:6])

    print(f'sampling {ns} training speakers')
    Z = model0.sample_speakers(ns)
    labels = randint(ns,size=(ntrain,))
    uu, labels, counts = np.unique(labels, return_inverse=True, return_counts=True)

    # sample training data
    print(f'sampling {ntrain} training data')
    Xtrain = model0.sample(Z, labels)

    # generate some test data
    #nt = 10000
    print(f'sampling {ns*2} test speakers')
    Z1 = model0.sample_speakers(ns)
    Z2 = model0.sample_speakers(ns)
    labels1 = randint(ns,size=(ntest,))
    labels2 = randint(ns,size=(ntest,))

    print(f'sampling {ntest*3} test data')
    Enroll = model0.sample(Z1, labels1)  # enrollment embeddings
    Test1 = model0.sample(Z1, labels1)   # target test embeddings
    Test2 = model0.sample(Z2, labels2)   # nnotar test embeddings

    return Xtrain, labels, Enroll, Test1, Test2


def generate_mix_psda(m, dim,ns,ntrain,ntest):

    p_i = np.ones(m)/m
    norm, mu = decompose(randn(m,dim))
    # if m>1:
    #     mu[1] = -mu[0]

    if m == 1:
        b = np.array([50])
        w = np.array([100])
    else:
        b = rng.uniform(20,100,size=m)
        w = rng.uniform(50,300,size=m)

    print(f"true     : 0","B =",b, "W =",w,"mu =",mu.ravel()[:6])
    print()

    model0 = MixPSDA(p_i, w, VMF(mu, b))

    labels = randint(ns,size=(ntrain,))
    uu, labels, counts = np.unique(labels, return_inverse=True, return_counts=True)
    onehot = labels[:,None] == uu
    # filter out speakers with < 2 utterances
    onehot = onehot[:,onehot.sum(axis=0) > 1]
    labels = labels[onehot.sum(axis=1) > 0]
    uu, labels, counts = np.unique(labels, return_inverse=True, return_counts=True)
    onehot = labels[:,None] == uu

    spk2comp = np.random.choice(range(m),size=ns,replace=True)
    # component_labels = np.asarray([spk2comp[spk] for spk in labels])

    print(f'sampling {ntrain} training data')
    Z = model0.sample_speakers(spk2comp)
    Xtrain = model0.sample(Z, spk2comp, labels)


    # # generate some test data
    print(f'sampling {ns*2} test speakers')
    spk2comp1 = np.random.choice(range(m),size=ns,replace=True)
    Z1 = model0.sample_speakers(spk2comp1)
    spk2comp2 = np.random.choice(range(m),size=ns,replace=True)
    Z2 = model0.sample_speakers(spk2comp2)

    labels1 = randint(ns,size=(ntest,))
    labels2 = randint(ns,size=(ntest,))
    Enroll = model0.sample(Z1, spk2comp1, labels1)
    Test1 = model0.sample(Z1, spk2comp1, labels1)
    Test2 = model0.sample(Z2, spk2comp2, labels2)

    return Xtrain, labels, Enroll, Test1, Test2




if __name__ == "__main__":

    dim = 2
    ns = 500  # number of training speakers
    ntrain = 10000  # numer of training examples
    ntest = 10000
    m = 3   # number of VMF components

    # Xtrain, labels, Enroll, Test1, Test2 = generate_psda(dim,ns,ntrain,ntest)
    Xtrain, labels, Enroll, Test1, Test2 = generate_mix_psda(m,dim,ns,ntrain,ntest)
    uu, labels, counts = np.unique(labels, return_inverse=True, return_counts=True)

    onehot = labels[:,None] == uu
    onehot = onehot[:,onehot.sum(axis=0) > 1]
    ii = onehot.sum(axis=1) > 0
    onehot = onehot[ii]
    Xtrain = Xtrain[ii]
    onesoft = onehot/onehot.sum(axis=0)

    labels = onehot.argmax(axis=1)
    counts = onehot.sum(axis=0)
    uu = np.unique(labels)

    # =================================================================

    # these are the 1st-order stats required by the em traning
    means = onesoft.T @ Xtrain

    p_i = np.ones(m)/m
    mm = np.random.randn(m,dim)
    bb = np.random.rand(m)*0.1 # np.ones_like(b)*1/0.11
    ww = np.random.rand(m)*100+300
    model1 = MixPSDA(p_i, ww, VMF(mm, bb))
    #model1 = model0
    model, obj = MixPSDA.em(means, counts, niters=20, w0=None, psda_init=model1)

    # =================================================================
    # =================================================================

    E = model.prep(Enroll)
    T1 = model.prep(Test1)
    T2 = model.prep(Test2)

    tar = E.llr_vector(T1)
    non = E.llr_vector(T2)

    tarc = np.sum(Enroll*Test1,axis=-1)
    nonc = np.sum(Enroll*Test2,axis=-1)

    fig,axes = plt.subplots(2,1,figsize=(12,12))
    axes[0].hist(non,bins=100,label='non',alpha=0.85,density=True)
    axes[0].hist(tar,bins=100,label='tar',alpha=0.85,density=True)
    axes[0].legend()

    eer,cllr,mincllr = quick_eval.tarnon_2_eer_cllr_mincllr(tar,non)
    axes[0].set_title(f"EER={eer:1.3%}, CLLR={cllr:1.3f}, minCLLR={mincllr:1.3f}")

    axes[1].hist(nonc,bins=100,label='nonc',alpha=0.85,density=True)
    axes[1].hist(tarc,bins=100,label='tarc',alpha=0.85,density=True)
    axes[1].legend()
    #axes[1].sharex(axes[0])
    eer,cllr,mincllr = quick_eval.tarnon_2_eer_cllr_mincllr(tarc,nonc)
    axes[1].set_title(f"EER={eer:1.3%}, CLLR={cllr:1.3f}, minCLLR={mincllr:1.3f}")


    plt.figure()
    plt.plot(non,nonc,'.',label='non')
    plt.plot(tar,tarc,'.',label='tar')
    plt.xlabel("PSDA")
    plt.ylabel("Cosine")

    eer,cllr,mincllr = quick_eval.tarnon_2_eer_cllr_mincllr(tar,non)
    plt.title(f"EER={eer:1.3%}, CLLR={cllr:1.3f}, minCLLR={mincllr:1.3f}")


    if 'plot' in sys.argv:
        cmap = plt.get_cmap('Spectral')
        cc = [cmap(s/ns) for s in labels]

        plt.figure()

        if dim == 2:
            x,y = Xtrain.T
            plt.scatter(x,y,color=cc,marker='o')
            # for spk in np.unique(labels):
            #     ii = labels==spk
            #     for k in np.unique(component_labels[ii]):
            #         jj = np.logical_and(ii,component_labels==k)
            #         plt.scatter(Xtrain[jj][:,0],Xtrain[jj][:,1],color=cmap(spk/ns), marker="x^+o."[k%4])

            for m in model.between.mu:
                plt.arrow(0,0,*m,color='r')
            plt.axis('square')
            plt.xlim(-1.2,1.2)
            plt.ylim(-1.2,1.2)
            plt.grid()
            plt.show()

        elif dim==3:

            cc = [cmap(s/ns) for s in labels]
            x,y,z = Xtrain.T

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x,y,z,color=cc,marker='o')

            for mi in model.between.mu:
                ax.quiver(0,0,0,*mi,color='r')



    plt.show()
