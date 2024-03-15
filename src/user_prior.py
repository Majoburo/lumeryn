import numpy as np
from eryn.prior import uniform_dist

def init_prior(wl,flux,eflux):

    sigmamin = (wl[1] - wl[0]) * 5
    sigmamax = (wl[-1] - wl[0]) / 20
    # If positive or negative wanna increase the range by 20%.
    mx = max(np.max(flux)*1.2,np.max(flux)*0.8)
    mn = min(np.min(flux)*1.2,np.min(flux)*0.8)
    # Don't wanna let gaussians have too little amplitude.
    minamp = -np.median(np.sqrt(eflux))/3

    priors = {
        "gauss": {
            0: uniform_dist(mn - mx, minamp),  # amplitude
            1: uniform_dist(wl[0] + sigmamin, wl[-1] - sigmamin),  # mean
            2: uniform_dist(sigmamin, sigmamax)  # variance
        },
        "knots": {
            0: uniform_dist(wl[0]+sigmamin, wl[-1]-sigmamin),
            1: uniform_dist(mn, mx),
        },
        "edges": {
            0: uniform_dist(mn, mx),
            1: uniform_dist(mn, mx),
        }
    }
    return priors
