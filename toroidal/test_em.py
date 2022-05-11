import numpy as np

from toroidal.psdamodel import ToroidalPSDA

import matplotlib.pyplot as plt
from subsphere.pca import Globe

from nlib.phonexia.embeddings import one_hot


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
model0 = ToroidalPSDA.random(D, d, w0, kappa, gamma_z, gamma_y)

# generate training data
X, Y, Z, Mu, labels = model0.sample(1000,100)
plabels, counts = one_hot.pack(labels, return_counts=True)
L = one_hot.scipy_sparse_1hot_cols(plabels)
Xsum = L @ X

# let's look at it
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
Globe.plotgrid(ax)
ax.scatter(*X.T, color='g', marker='.',label='X')
Ze = model0.Ez.embed(Z)
ax.scatter(*Ze.T, color='r', marker='.',label='Ze')
ax.legend()
ax.set_title(f'Training Data\nzdim = 1, ydim = 2, snr = 1, kappa={kappa}')
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
#plt.show()


snr = 1
w = np.array([np.sqrt(snr),1])
#gamma_z_init = gamma_z 
gamma_z_init = np.full(m,1)
gamma_y_init = np.full(n-m,10)
#gamma_y_init = gamma_y 
model = ToroidalPSDA.random(D, d, w, kappa/5, gamma_z_init, gamma_y_init)
for i in range(50):
    print(f"em iter {i}: w = {model.E.w}, kappa = {model.kappa}")
    model = model.em_iter(X, Xsum)
print(f"\ncf true   w = {model0.E.w}, kappa = {model0.kappa}")

# generate data from trained model
X, Y, Z, Mu, labels = model.sample(1000,100)
#plabels, counts = one_hot.pack(labels, return_counts=True)

# let's look at it
ax = fig.add_subplot(122, projection='3d')
Globe.plotgrid(ax)
ax.scatter(*X.T, color='g', marker='.',label='X')
Ze = model.Ez.embed(Z)
ax.scatter(*Ze.T, color='r', marker='.',label='Ze')
ax.legend()
ax.set_title('New data from trained model')
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
plt.show()






