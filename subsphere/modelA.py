import numpy as np
from numpy.random import randn

from subsphere.pca import UnitSphere, Subsphere, Globe, retract
from psda.vmf import VMF

class ModelA:
    def __init__(self,wdim, zdim, kappaw, kappax):
        self.kappax = kappax
        self.wdim = wdim
        self.zdim = zdim
        self.D = D = wdim + zdim
        F = retract(randn(D,D))
        self.W, self.B = F[:,:wdim], F[:,wdim:]
        
        self.zprior = VMF(zdim)  # uniform
        # muw = np.zeros(wdim+1)   # 1 signal weight, and wdim noise weights
        # muw[0] = 1
        muw = retract(np.ones(wdim+1))
        self.wprior = VMF(muw,kappaw)  # concentrates at low noise state
        
        
        
    def sample_speakers(self, n):
        return self.zprior.sample(n)
    
    
    def sample(self, speakers, labels):
        n = len(labels)
        SN = self.wprior.sample(n)
        signal = SN[:,0]
        Noise = SN[:,1:]
        Z = signal.reshape(-1,1)*speakers[labels,:]
        Mu = Z@self.B.T + Noise@self.W.T
        return VMF(Mu,self.kappax).sample()
        
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap
    
    
    ns = 3
    n = 200
    labels = np.arange(ns).repeat(n)
    model = ModelA(1,2,20,1000)
    
    # Z = model.sample_speakers(ns)
    theta = np.linspace(0,ns/(ns+1)*2*np.pi,ns)
    Z = np.vstack([np.cos(theta),np.sin(theta)]).T
    Mu = Z@model.B.T    
    X = model.sample(Z, labels)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Globe.plotgrid(ax)
    ax.scatter(*X.T, color='b', marker='.',label='embeddings')
    ax.scatter(*Mu.T, color='r', marker='o',label='speakers')
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.legend()
    fig.show()    
    
    
    
    
    
    