import numpy as np
from numpy.linalg import solve
from numpy.random import randn

from subsphere.pca import Globe
from psda.vmf import VMF, decompose


def rsolve(num, den):
    return solve(den.T,num.T).T

def lengthnorm(x):
    return decompose(x)[1]

import matplotlib.pyplot as plt


def quad(a,b,c):
    delta = np.sqrt(b**2 - 4*a*c)
    return (-b+delta)/(2*a), (-b-delta)/(2*a) 


def invert_affine(Y, A, b):
    Z = solve(A,Y)
    c = -solve(A,b)
    Aq = (Z**2).sum(axis=0)
    Bq = 2*c@Z
    Cq = c@c - 1
    alpha1, alpha2 = quad(Aq, Bq, Cq)
    X1 = alpha1*Z + c.reshape(-1,1)
    X2 = alpha2*Z + c.reshape(-1,1)
    return X1, X2

# fig = plt.figure()
# ax1 = fig.add_subplot(121, projection='3d')
# ax2 = fig.add_subplot(122, projection='3d')
# Globe.plotgrid(ax1)
# Globe.plotgrid(ax2)

# n, dim = 300, 3
# X = VMF(dim).sample(n)
# ax1.scatter(*X.T, color='g', marker='x',label='X (uniform)')

# R = np.diag([1.0,5,50])/30#randn(3,3)
# Ylin = X@R
# Y = lengthnorm(X@R)
# ax2.scatter(*Ylin.T, label='Ylin = R @ X',marker='.')
# ax2.scatter(*Y.T, label='Y = lengthnorm(R @ X)',marker='.')

# Xhat = lengthnorm(rsolve(Y,R))
# ax1.scatter(*Xhat.T, color='r', marker='.',label='X = lengthnorm(solve(R,Y))')
# ax1.legend()
# ax2.legend()


# ax1.set_xlim([-1,1])
# ax1.set_ylim([-1,1])
# ax1.set_zlim([-1,1])

# ax2.set_xlim([-1,1])
# ax2.set_ylim([-1,1])
# ax2.set_zlim([-1,1])

n = 50
A = np.eye(3) #randn(3,3)
#A = randn(3,3)
b = np.array([0,0,-1.5]) #np.array([0,0,-1])
mu = np.array([0,0,1.0]) #VMF(3).sample().ravel()
X = VMF(mu,100).sample(n).T

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
Globe.plotgrid(ax1)
ax1.scatter(*X, color='g', marker='x',label='X')



Ylin = A@X + b.reshape(-1,1)

Y = lengthnorm((A@X + b.reshape(-1,1)).T).T
ax1.scatter(*Y, color='b', marker='o',label='Y')
ax1.scatter(*Ylin, color='c', marker='.',label='Ylin')



X1, X2 = invert_affine(Y, A, b)


ax1.scatter(*X1, color='r', marker='.',label='X solution 1')
ax1.scatter(*X2, color='m', marker='.',label='X solution 2')




Y1 = lengthnorm((A@X1 + b.reshape(-1,1)).T).T
Y2 = lengthnorm((A@X2 + b.reshape(-1,1)).T).T


#print(X)
print("X agreement:\n",(X*X1).sum(axis=0))
print((X*X2).sum(axis=0))

print("\nY agreement:\n",(Y*Y1).sum(axis=0))
print((Y*Y2).sum(axis=0))



ax1.set_xlim([-1,1])
ax1.set_ylim([-1,1])
ax1.set_zlim([-1,1])
ax1.legend()







