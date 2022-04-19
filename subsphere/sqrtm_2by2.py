"""
https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix#A_general_formula
"""
import numpy as np
from numpy.random import randn

from scipy.linalg import eigh, svd


def sqrtm_eigh(C):
    """
    Symmetric matrix square root of a symmetric, positive definite 
    matrix. The result is symmetric positive definite.
    """
    e, V = eigh(C) # C = (V*e) @ V.T
    assert all(e>=0), "input must be positive semi-definite"
    return (V * np.sqrt(e)) @ V.T


def sqrtm_trace(M, upper = True):
    a,b,c,d = M.ravel()
    if upper:
        c = b
    else:
        b = c
    tr = a + d
    det = a*d - b*c
    s = np.sqrt(det)
    t = np.sqrt(tr+2*s)
    return (tr+2*s)/t
 

X = randn(2,3)
M = X @ X.T

print(sqrtm_eigh(M).trace(), sqrtm_trace(M))
