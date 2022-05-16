import numpy as np

from toroidal.psdamodel import ToroidalPSDA, train_ml, train_map, \
                               KappaPrior_KL


import matplotlib.pyplot as plt
from subsphere.pca import Globe

from nlib.phonexia.embeddings import one_hot



def train_kappa(d, m, niters, X, labels, kappa, pseudo_count):
    dim = X.shape[-1]
    kappa_prior = KappaPrior_KL(dim,kappa,pseudo_count)
    gammaz_prior = [KappaPrior_KL.assign(di, 1.0, 1/100) for di in d[:m]]
    gammay_prior = [KappaPrior_KL.assign(di, 1.0, 1/100) for di in d[m:]]
    model1, obj1 = train_map(d, m, niters, X, labels, kappa_prior, 
                             gammaz_prior, gammay_prior)
    model2, obj2 = train_ml(d, m, niters, X, labels, initial_model = model1)
    return model1, obj1, model2, obj2


def plotdata(ax,title,X,Z,model):
    Globe.plotgrid(ax)
    ax.scatter(*X.T, color='g', marker='.',label='X')
    Ze = model.Ez.embed(Z)
    ax.scatter(*Ze.T, color='r', marker='.',label='Ze')
    ax.legend()
    ax.set_title(title)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])

D = 3
m, n = 1,2
d = np.array([2,1])        
#gamma_z = np.zeros(m)
gamma_z = np.full(m,5.0)
#gamma_y = np.zeros(n-m)
gamma_y = np.full(n-m,1)
kappa = 2000
snr = 10
w0 = np.array([np.sqrt(snr),1])
gamma = np.hstack([gamma_z,gamma_y])
model0 = ToroidalPSDA.random(D, d, m, w0, kappa, gamma)

# generate training data
print('sampling training data')
Xtr, Ytr, Ztr, Mutr, labels = model0.sample(1000,100)

# let's look at it
fig = plt.figure()
ax = fig.add_subplot(221, projection='3d')
plotdata(ax,f'Training Data (kappa={model0.kappa})',Xtr,Ztr,model0)


niters = 50
model1, obj1, model2, obj2 = train_kappa(d, m, niters, Xtr, labels, 
                                         kappa = 2, pseudo_count = 1000)
ax = fig.add_subplot(222, projection='3d')
X, Y, Z, Mu, labels = model1.sample(1000,100)
plotdata(ax,f'MAP Model 1 (kappa={model1.kappa})',X,Z,model1)

ax = fig.add_subplot(224, projection='3d')
X, Y, Z, Mu, labels = model2.sample(1000,100)
plotdata(ax,f'ML Model 2 (kappa={model2.kappa})',X,Z,model2)



plt.show()
