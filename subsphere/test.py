import numpy as np
from numpy.random import randn

from subsphere.pca import retract, lengthnorm
from psda.vmf import VMF

F = retract(randn(4,4))
J, K = F[:,:2], F[:,2:]
#a,b = VMF(2).sample().ravel()

x = VMF(4).sample().ravel()

z = lengthnorm(J.T@x)
y = -lengthnorm(K.T@x)
a, b = z@(J.T@x), y@(K.T@x) 

hatx = a*(J@z) + b*(K@y)

print(x)
print(hatx)