import numpy as np
from scipy.interpolate import UnivariateSpline as spl

def _gaussian(x, a, b, c):
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))

def _multi_gaussian(x, params):
    template = np.zeros_like(x)
    for param in params:
        template += gaussian_pulse(x, *param)
    return template

def _monomial(x,a,n):
    return a*x**n

def _polynomial(x,params):
    template = np.zeros_like(x)
    for param in params:
        template += poly(x, *param)
    return template

def _binary_search(nknots,x,y,maxiter=20):
    """
    Binary search for the smoothness parameter "s" to get a spline with nknots.
    """
    hi,low = 7, -10
    for _ in range(maxiter):
        s = 10**((hi+low)/2)
        ss = spl(x,y,k=3,s=s)
        if len(ss.get_knots()) < nknots:
            hi = np.log10(s)
        elif len(ss.get_knots()) > nknots:
            low = np.log10(s)
        else:
            return s,ss
    return s,ss
