import numpy as np

from toroidal.psdamodel import ToroidalPSDA, train_ml, train_map, \
                               KappaPrior_KL


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
gamma = np.hstack([gamma_z,gamma_y])
model0 = ToroidalPSDA.random(D, d, m, w0, kappa, gamma)

# generate training data
print('sampling training data')
X, Y, Z, Mu, labels = model0.sample(1000,100)
plabels, counts = one_hot.pack(labels, return_counts=True)
L = one_hot.scipy_sparse_1hot_cols(plabels)
Xsum = L @ X

# let's look at it
fig = plt.figure()
ax = fig.add_subplot(221, projection='3d')
Globe.plotgrid(ax)
ax.scatter(*X.T, color='g', marker='.',label='X')
Ze = model0.Ez.embed(Z)
ax.scatter(*Ze.T, color='r', marker='.',label='Ze')
ax.legend()
ax.set_title(f'Training Data\nzdim = {d[0]}, ydim = {d[1]}, snr = {snr}, kappa={kappa}')
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
#plt.show()


# model = ToroidalPSDA.random(D, d, m)
# for i in range(50):
#     print(f"em iter {i}: w = {model.E.w}, kappa = {model.kappa}")
#     model = model.em_iter(X, Xsum)
# print(f"\ncf true   w = {model0.E.w}, kappa = {model0.kappa}")

#dh = np.array([1,2])        
print("ML training")
niters = 50
model = train_ml(d, m, niters, X, labels)



# generate data from trained model
X, Y, Z, Mu, labels = model.sample(1000,100)
#plabels, counts = one_hot.pack(labels, return_counts=True)

# let's look at it
ax = fig.add_subplot(222, projection='3d')
ax.set_title("Samples from ML model")
Globe.plotgrid(ax)
ax.scatter(*X.T, color='g', marker='.',label='X')
Ze = model.Ez.embed(Z)
ax.scatter(*Ze.T, color='r', marker='.',label='Ze')
ax.legend()
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])


print("MAP training")
niters = 50
kappa_prior = KappaPrior_KL.assign(D,1,1)
gamma_z_prior = [KappaPrior_KL.assign(d[0],1,100)]
gamma_y_prior = [KappaPrior_KL.assign(d[1],1,1)]
model = train_map(d, m, niters, X, labels,kappa_prior,
                  gamma_z_prior, gamma_y_prior)

# generate data from trained model
X, Y, Z, Mu, labels = model.sample(1000,100)
#plabels, counts = one_hot.pack(labels, return_counts=True)

# let's look at it
ax = fig.add_subplot(223, projection='3d')
ax.set_title("Samples from MAP model")
Globe.plotgrid(ax)
ax.scatter(*X.T, color='g', marker='.',label='X')
Ze = model.Ez.embed(Z)
ax.scatter(*Ze.T, color='r', marker='.',label='Ze')
ax.legend()
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])



plt.show()






