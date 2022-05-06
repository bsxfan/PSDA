import numpy as np

from toroidal.psdamodel import ToroidalPSDA

import matplotlib.pyplot as plt
from subsphere.pca import Globe

from nlib.phonexia.embeddings import one_hot


D = 3
m, n = 1,2
d = np.array([2,1])        
gamma_z = np.zeros(m)
gamma_y = np.zeros(n-m)
kappa = 200
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
model = ToroidalPSDA.random(D, d, w, kappa, gamma_z, gamma_y)
for i in range(5):
    print(f"em iter {i}: w = {model.E.w}")
    model = model.em_iter(X, Xsum)
print(f"\ncf true w:     {model0.E.w}")

# generate data from trained model
X, Y, Z, Mu, labels = model0.sample(1000,100)
plabels, counts = one_hot.pack(labels, return_counts=True)

# let's look at it
ax = fig.add_subplot(122, projection='3d')
Globe.plotgrid(ax)
ax.scatter(*X.T, color='g', marker='.',label='X')
Ze = model0.Ez.embed(Z)
ax.scatter(*Ze.T, color='r', marker='.',label='Ze')
ax.legend()
ax.set_title('New data from trained model')
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
plt.show()






