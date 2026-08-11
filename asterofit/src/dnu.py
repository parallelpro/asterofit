import numpy as np
from scipy.optimize import linear_sum_assignment, curve_fit

__all__ = ['get_model_Dnu', 'get_obs_Dnu']


def get_model_Dnu(mod_freq, mod_l, Dnu, numax, mod_n=None):
    
    """
    Calculate model Dnu (and epsilon) around numax, by fitting mode frequency
    vs. radial order n (a straight line, `freq = (n + eps) * Dnu`) restricted
    to l=0 modes and weighted by a Gaussian envelope centered on numax -- so
    only modes near numax (where the star is actually observed to oscillate)
    dominate the fit, the same way `Dnu`/`eps` would be measured from data.

    ----------
    Input:
    mod_freq: array_like[Nmode_mod]
        model's mode frequency (all degrees; only l=0 entries are used)
    mod_l: array_like[Nmode_mod]
        model's mode degree
    Dnu: float
        the star's (observed) large separation in muHz -- only used to set
        the Gaussian envelope width via the numax-width scaling relation,
        not fit directly
    numax: float
        the frequency of maximum power in muHz -- the envelope's center

    ----------
    Optional input:
    mod_n: array_like[Nmode_mod] or None
        model's radial order, for the l=0 modes. If None, falls back to
        `arange(n_l0_modes)` -- fine as a relative ordering (the fit doesn't
        care about the absolute n), but only meaningful if the l=0 modes in
        `mod_freq`/`mod_l` are already contiguous/ordered by n with no gaps.

    ----------
    Return:
    mod_Dnu, mod_eps: float, float
        The fitted large separation and offset (`(mod_n[i] + mod_eps) *
        mod_Dnu == mod_freq_l0[i]`, approximately). Both NaN if fewer than 3
        l=0 modes fall within the Gaussian envelope (weight > 1e-100).

    """

    # width estimates based on Yu+2018, Lund+2017, Li+2020
    width_slope, width_intercept = 0.9638, -1.7145
    width = np.exp(width_slope*np.log(numax) + width_intercept)

    # assign n
    l0_mask = mod_l==0
    mod_freq_l0 = np.copy(mod_freq)[l0_mask]
    sort_idx = np.argsort(mod_freq_l0)
    mod_freq_l0 = mod_freq_l0[sort_idx]
    mod_n = np.arange(len(mod_freq_l0)) if (mod_n is None) else np.copy(mod_n)[l0_mask][sort_idx]
    # print(mod_n, mod_freq_l0)
    # sigma = 1/np.exp(-(mod_freq_l0-numax)**2./(2*width**2.))
    weight = np.exp(-(mod_freq_l0-numax)**2./(2*width**2.))
    significant_weight_mask = weight>1e-100
    if np.sum(significant_weight_mask)>2:
        fit_coeffs, _, _, _, _ = np.polyfit(mod_n[significant_weight_mask], mod_freq_l0[significant_weight_mask], 1, w=weight[significant_weight_mask], full=True)
        mod_Dnu, mod_eps = fit_coeffs[0], fit_coeffs[1]/fit_coeffs[0]
    else:
        mod_Dnu, mod_eps = np.nan, np.nan


    return mod_Dnu, mod_eps


def get_obs_Dnu(obs_freq, obs_efreq=None, Dnu_guess=None, ifReturnEpsilon=False):
    
    """
    Calculate observed Dnu (and, optionally, epsilon) from l=0 radial modes,
    by fitting `freq = (n + eps) * Dnu` via least squares (`scipy.curve_fit`)
    -- unlike `get_model_Dnu`, every mode is used (no numax-weighting), and
    the radial order n is inferred from the frequencies themselves (assuming
    they're already sorted and roughly evenly spaced by `Dnu_guess`) rather
    than taken as an input, since real observations aren't labeled with n.

    ----------
    Input:
    obs_freq: array_like[Nmode_obs]
        radial (l=0) mode frequency, in frequency order.

    ----------
    Optional input:
    obs_efreq: array_like[Nmode_obs] or None
        mode frequency uncertainty, used as the fit's sigma. Defaults to
        uniform (unweighted) uncertainty if not given.
    Dnu_guess: float or None
        An initial guess for Dnu, used only to infer each mode's radial
        order n from gaps in `obs_freq` (a mode more than ~1.5*Dnu_guess past
        the previous one is assumed to skip an order). Defaults to the
        median frequency spacing in `obs_freq`.
    ifReturnEpsilon: bool, default False
        If True, also return the fitted epsilon and its uncertainty.

    ----------
    Return:
    obs_Dnu, obs_eDnu: float, float
        The fitted large separation and its 1-sigma uncertainty.
    obs_eps, obs_eeps: float, float -- only if ifReturnEpsilon=True
        The fitted epsilon offset and its 1-sigma uncertainty.

    """
    if obs_efreq is None: obs_efreq = np.ones(len(obs_freq))
    if Dnu_guess is None: Dnu_guess = np.median(np.diff(np.sort(obs_freq)))

    Nmode = len(obs_freq)

    mode_n = np.arange(Nmode) + np.floor(obs_freq[0]/Dnu_guess)
    for mode_idx in range(len(mode_n)-1):
        n_correction = np.round((obs_freq[mode_idx+1]-obs_freq[mode_idx])/Dnu_guess)-1
        mode_n[(mode_idx+1):] = mode_n[(mode_idx+1):] + n_correction

    # ## 1 - all modes without curvature
    def dnu_model(xdata, Dnu, eps):
        return (xdata + eps )*Dnu
    # print(obs_freq, obs_efreq,mode_n)
    popt, pcov = curve_fit(dnu_model, mode_n, obs_freq, sigma=obs_efreq)
    perr = np.diag(pcov)**0.5

    obs_Dnu, obs_eDnu = popt[0], perr[0]
    obs_eps, obs_eeps = popt[1], perr[1]

    if ifReturnEpsilon:
        return obs_Dnu, obs_eDnu, obs_eps, obs_eeps
    else:
        return obs_Dnu, obs_eDnu
