def get_max(ensemble,param):
    """
    Returns the max-likelihood sample.
    """
    amax = np.argmax(ensemble.get_log_like().flatten())
    samp = ensemble.get_chain()[param]
    evals, temps, walker, nleaves,dim = samp.shape
    samp = samp.reshape(evals*temps*walker,nleaves,dim)[amax]
    naninds = np.logical_not(np.isnan(samp))
    nleaves = np.sum(naninds[:,0])
    return samp[naninds].reshape(nleaves,dim)

def get_clean_best_nleave_chain(ensemble, param, temp=0):
    """
    Returns only the chains with the same number of leaves for "param" as the max-likelihood one.
    """
    amax = np.argmax(ensemble.get_log_like())
    best_nleaves = ensemble.get_nleaves()[param].flatten()[amax]
    mask = ensemble.get_nleaves()[param][:,temp,:] == best_nleaves
    sparams = ensemble.get_chain()[param][:,temp,:][mask]

    naninds = np.logical_not(np.isnan(sparams))
    num,_,dim = sparams.shape
    return sparams[naninds].reshape(num,best_nleaves,dim)


def color_interp(rgb_0, rgb_f, steps):
    """
    Gives good colors for the dif temps on the chains.
    """
    r, g, b = rgb_0
    re, ge, be = rgb_f
    return [
        "#{:02X}{:02X}{:02X}".format(
            int((re - r) / steps * i + r),
            int((ge - g) / steps * i + g),
            int((be - b) / steps * i + b),
        )
        for i in range(steps)
    ]

label="trace"
outdir="."

def trace_plots(ll, samples_burnt, param):
    # trace plots
    _, ntemps, nwalkers, _, ndim = samples_burnt.shape
    fig, axes = plt.subplots(ndim + 1, figsize=(10, 7 / 3 * (ndim + 1)), sharex=True)
    if names not None:
        keylist = names
    #keylist = list(priors.keys())
    #labels = keylist
    #ntemps = ntemps
    #nwalkers = nwalkers
    csteps = color_interp((0, 10, 200), (200, 10, 0), ntemps)
    for j in range(ndim):
        ax = axes[j]
        ax.set_xlim(0, samples_burnt.shape[0])
        ax.set_ylabel(keylist[j])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        for t in reversed(range(ntemps)):
            for k in range(nwalkers):
                ax.plot(samples_burnt[:, t, k, 0, j], c=csteps[t], alpha=0.2)

    for k in range(nwalkers):
        axes[-1].plot(ll[:, 0, k])
    axes[-1].set_ylabel("log pos (T=1)")
    axes[-1].yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    plt.savefig(os.path.join(outdir, f"{label}_trace.png"))
    plt.close()

def plot_best(wl,flux):
    plt.plot(wl, flux, label="data", color="lightskyblue")
    for i in range(100):
        gaussians = utils.multi_gaussian(wl, gauss[i])
        interp_mod = utils.get_spline(knots[i],edges[i],wl)
        plt.plot(wl,interp_mod(wl)+gaussians,alpha=0.02,color="orange")
    gaussians = combine_gaussians(wl,get_max(ensemble,'gauss'))
    interp_mod = get_spline(get_max(ensemble,'knots'),get_max(ensemble,'edges'),wl)
    plt.plot(wl,interp_mod(wl)+gaussians,color="black")
    plt.show()
    return
