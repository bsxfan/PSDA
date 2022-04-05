import numpy as np

from oldpsda import vmf as oldvmf
from psda import vmf as newvmf
from psda import besseli 


import matplotlib.pyplot as plt

dim = 100
oldC = oldvmf.LogNormConst(dim)
newC = newvmf.LogNormConst(dim)

nu = dim/2-1
oldlogI = oldvmf.LogBesselI(nu)
newlogI = besseli.LogBesselI(nu)

plt.figure()
k = np.linspace(100,150,200)
plt.plot(k, oldC(k), label='old')
plt.plot(k, newC(k), '--', label='new')
plt.grid()
plt.legend()
plt.xlabel('kappa')
plt.ylabel('Log Cnu')
plt.title("Hier's die fokkop")
plt.show()

plt.figure()
k = np.linspace(100,150,200)
plt.plot(k, oldlogI(k), label='old')
plt.plot(k, newlogI.log_iv(k), '--', label='new')
plt.grid()
plt.legend()
plt.xlabel('kappa')
plt.ylabel('Log Inu')
plt.title("Hier's die fokkop")
plt.show()