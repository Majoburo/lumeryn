import numpy as np
import utils
import matplotlib.pyplot as plt

knots = np.array([[450,1.3],[550,1.2]])
edges = np.array([[1.1,1.4]])
wl = np.linspace(400, 600, 1000)
interp_model = utils.get_spline(knots,edges,wl)
params = [[-0.3,450,2], # amp, mean, width
          [-0.2,530,1]
        ]
gaussians = utils._multi_gaussian(wl,params)
amp_noise = 0.05
eflux = amp_noise*np.ones_like(wl)
noise = eflux*np.random.randn(len(wl))
flux = gaussians + interp_model(wl) + noise
plt.plot(wl,flux)
plt.show()
np.savetxt("testdata2.dat",list(zip(wl,flux,eflux)))

