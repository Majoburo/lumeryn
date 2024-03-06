import numpy as np
import configparser
from scipy.signal import find_peaks,peak_prominences,peak_widths,savgol_filter
from eryn.prior import uniform_dist
from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.moves import GaussianMove,GroupMove
from eryn.backends import HDFBackend
from spec_likelihood import SpectraLikelihood
import utils


class LumerynSpecFitter:
    def __init__(self, wl, flux, eflux, prior_conf=None, sampler_conf=None):
        if prior_conf is None:
            prior_conf = 'prior.ini'
            print(f"[lumeryn]: Loading default priors from {prior_conf}")
        if sampler_conf is None:
            sampler_conf = 'sampler.ini'
            print(f"[lumeryn]: Loading default sampler params from {sampler_conf}")
        prior_par = configparser.ConfigParser()
        mcmc_par = configparser.ConfigParser()
        prior_par.read(prior_conf)
        mcmc_par.read(sampler_conf)
        self.prior_par = prior_par
        self.mcmc_par = mcmc_par
        mcmc_par = self.mcmc_par['eryn']
        self.ntemps = int(mcmc_par['ntemps'])
        self.nwalkers = int(mcmc_par['nwalkers'])
        self.wl = wl
        self.flux = flux
        self.eflux= eflux

    def generate_initial_guess(self, nameplot=None, sfwindow=50, pixwidthmin=20, pixwidthmax=400, prominence=None, knots_continuum=6):
        """
        Initializes the MCMC for spectral fitting of absorption lines and continuum.
        For the gaussians:
        - Smooths the data using a Savitzky-Golay filter. Specify window size and pixel width min and max.
        - Identifies peaks and their parameters.
        For the continuum:
        - Smooths the data to a spline with "n" defined knots to give the continuum spline a start.
        """
        wl = self.wl
        flux = self.flux

        if prominence is None:
            prominence = np.median(np.sqrt(eflux) / flux) /4  # Using half of signal/noise as prominence enough to id.

        smooth_flux = savgol_filter(flux, sfwindow, 3)
        apeak, _ = find_peaks(-smooth_flux, width=[pixwidthmin, pixwidthmax], prominence=prominence)
        prominences = peak_prominences(-smooth_flux, apeak)[0]
        results_half = peak_widths(-smooth_flux, apeak, rel_height=0.5)
        half_height, x1, x2 = results_half[1:]
        widths = wl[np.rint(x2).astype(int)] - wl[np.rint(x1).astype(int)]

        s, ss = utils._binary_search(knots_continuum, wl, flux)

        if nameplot is not None:
            plt.plot(wl, flux, label="data", color="lightskyblue")
            plt.plot(wl, smooth_flux)
            plt.plot(wl[apeak], flux2[apeak], "x")
            plt.vlines(x=wl[apeak], ymin=smooth_flux[apeak], ymax=smooth_flux[apeak] + prominences, color='black')
            plt.hlines(-half_height, wl[np.rint(x1).astype(int)], wl[np.rint(x2).astype(int)], color="C2")
            plt.plot(wl, ss(wl), label='continuum start', color='yellow')
            plt.ylabel("flux")
            plt.xlabel("wavelength")
            plt.legend()
            #plt.show()
            plt.savefig(f"{nameplot}.pdf")
        self._xpeak = wl[apeak]
        self._ypeak = prominences
        self._wpeak = widths
        self._knots = ss

        return

    def _init_eryn_prior(self):
        wl = self.wl
        flux = self.flux


        varmin = (wl[1] - wl[0]) * 5
        varmax = (wl[-1] - wl[0]) / 5

        mx = np.max(flux)
        mn = np.min(flux)

        self.priors = {
            "gauss": {
                0: uniform_dist(mn - mx, 0),  # amplitude
                1: uniform_dist(wl[0] + varmin, wl[-1] - varmin),  # mean
                2: uniform_dist(varmin, varmax)  # variance
            },
            "knots": {
                0: uniform_dist(wl[0], wl[-1]),
                1: uniform_dist(mn, mx),
            },
            "edges": {
                0: uniform_dist(mn, mx),
                1: uniform_dist(mn, mx),
            }
        }
        return

    def initialize_chains(self):
        self._init_eryn_prior()
        wl = self.wl
        flux = self.flux
        if not hasattr(self,'_xpeak'):
            self.generate_initial_guess()
        xpeak = self._xpeak
        ypeak = self._ypeak
        wpeak = self._wpeak
        knots = self._knots
        prior_par = self.prior_par
        mcmc_par = self.mcmc_par['eryn']
        ntemps = self.ntemps
        nwalkers = self.nwalkers
        priors = self.priors

        branches = prior_par.sections()
        ndims = {}
        coords = {}
        indxs = {}
        groups = {}
        nleaves_max = {}
        nleaves_min = {}


        for branch in branches:
            ndims[branch]  = int(prior_par[branch]['dim'])
            nleaves_max[branch]  = int(prior_par[branch]['nmax'])
            nleaves_min[branch]  = int(prior_par[branch]['nmin'])
            coords[branch] = np.zeros((ntemps, nwalkers, int(prior_par[branch]['nmax']), int(prior_par[branch]['dim'])))
            indxs[branch]  = np.zeros((ntemps, nwalkers, int(prior_par[branch]['nmax'])), dtype=bool)
            groups[branch] = np.arange(coords[branch].shape[0] * coords[branch].shape[1]).reshape(coords[branch].shape[:2])[:, :, None]
            groups[branch] = np.repeat(groups[branch], coords[branch].shape[2], axis=-1)

        def init_check(coords):
            """Checking the initial values are within the priors
            """
            for name in prior_par.sections():
                ndim = int(prior_par[name]['dim'])
                nleaf = int(prior_par[name]['nmax'])
                prior_bounds = np.array([ [priors[name][n].min_val, priors[name][n].max_val] for n in range(ndim)])
                mn,mx = prior_bounds.T
                mask = np.logical_and.reduce((coords[name] > mn) & (coords[name] < mx),axis=-1)
                if not mask.all():
                    print(f"[lumeryn]: error, some {name} initial conditions outside priors")
                    print(f"[lumeryn]: priors: {prior_bounds}")
                    print("[lumeryn]: bad coords:")
                    print(coords[name][~mask])
            return

        for w in range(nwalkers):
            for t in range(ntemps):
                # gauss
                for nn in range(int(prior_par["gauss"]["nmax"])):
                    i = np.random.randint(0, len(ypeak))
                    coords["gauss"][t, w, nn, 0] = -ypeak[i] + 1e-6 * np.random.randn()
                    coords["gauss"][t, w, nn, 1] = xpeak[i] + 1e-6 * np.random.randn()
                    coords["gauss"][t, w, nn, 2] = wpeak[i] + 1e-6 * np.random.randn()

                for nn in range(int(prior_par["knots"]["nmax"])):
                    i = np.random.randint(1, len(knots.get_knots()) - 1)
                    coords["knots"][t, w, nn, 0] = knots.get_knots()[i] + 1e-2 * np.random.randn()
                    coords["knots"][t, w, nn, 1] = knots(knots.get_knots())[i] + 1e-2 * np.random.randn()

                for nn in range(int(prior_par["edges"]["nmax"])):
                    coords["edges"][t, w, nn, 0] = knots(knots.get_knots())[0] + 1e-2 * np.random.randn()
                    coords["edges"][t, w, nn, 1] = knots(knots.get_knots())[-1] + 1e-2 * np.random.randn()

        init_check(coords)

        # turn on at least this many knots in each temp and walker
        indxs['knots'][:, :, :int(prior_par['knots']['nmin'])] = True
        # turn on at least this many gaussians in each temp and walker
        indxs['gauss'][:, :, :int(prior_par['gauss']['nmin'])] = True

        for name, inds_temp in indxs.items():
            inds_fix = np.where(np.sum(inds_temp, axis=-1) == 0)

            for ind1, ind2 in zip(inds_fix[0], inds_fix[1]):
                inds_temp[ind1, ind2, 0] = True

        coords_in = {name: coords[name][indxs[name]] for name in coords}
        groups_in = {name: groups[name][indxs[name]] for name in groups}
        self.coords_in = coords_in
        self.groups_in = groups_in
        self.indxs = indxs
        self.coords = coords
        self.ndims = ndims
        self.nleaves_max = nleaves_max
        self.nleaves_min = nleaves_min
        return
    def fit(self,backendname=None):
        if backendname is None:
            print("[lumeryn]: No backend file provided, not saving chains.")
        wl = self.wl
        flux = self.flux
        if not hasattr(self,'coords_in'):
            self.initialize_chains()
        prior_par = self.prior_par
        mcmc_par = self.mcmc_par['eryn']
        ntemps = self.ntemps
        nwalkers = self.nwalkers
        priors = self.priors
        coords_in = self.coords_in
        groups_in = self.groups_in
        indxs = self.indxs
        coords = self.coords

        branches = prior_par.sections()

        speclike = SpectraLikelihood(wl, flux, eflux, wltol=.05, kind='linear')

        ll_eval = speclike.evaluate([coords_in['gauss'], coords_in["knots"], coords_in["edges"]], [groups_in['gauss'], groups_in["knots"], groups_in["edges"]])

        print('llh shape = ', ll_eval.shape)
        print('ll_eval = ', ll_eval)

        log_prob = ll_eval.reshape(ntemps, nwalkers)

        betas  = np.linspace(1.0, 0.0, ntemps)
        factor = float(mcmc_par['factor'])
        cov    = {
            "gauss": np.diag(np.ones(3)) * factor,
            "knots": np.diag(np.ones(2)) * factor,
            "edges": np.diag(np.ones(2)) * factor
        }

        moves = GaussianMove(cov)

        if backendname is not None:
            backend = HDFBackend(backendname + ".h5")
        else:
            backend = None

        ensemble = EnsembleSampler(
            nwalkers,
            self.ndims,  # assumes ndim_max
            speclike.evaluate,
            priors,
            args=None,
            tempering_kwargs=dict(betas=betas,adaptive=False),
            nbranches=len(self.nleaves_max),
            branch_names=list(self.nleaves_max.keys()),
            nleaves_max=self.nleaves_max,
            nleaves_min=self.nleaves_min,
            provide_groups=True,
            update_iterations=1,
            plot_iterations=-1,
            moves=moves,
            rj_moves=True,
            vectorize=True,
            backend=backend,
        )
        #import pdb
        #pdb.set_trace()

        print('[lumeryn]:  * Started sampling ...')
        state = State(coords, log_like=log_prob, betas=betas, blobs=None, inds=indxs)

        burnin = int(mcmc_par['burnin'])
        nsteps = int(mcmc_par['nsteps'])
        ensemble.run_mcmc(state, nsteps, burn=burnin, progress=True, thin_by=1)

        print('[lumeryn]:  * Finished sampling!')

if __name__ == "__main__":
    wl,flux,eflux = np.loadtxt("testdata.dat").T
    lum = LumerynSpecFitter(wl,flux,eflux)
    # These two following routines will be run by default when calling LumerynSpecFitter.fit(), but are accesible and have many useful options.
    lum.generate_initial_guess()
    lum.initialize_chains(backendname = "testdata")
    ensemble = lum.fit()

