import numpy as np
from scipy.optimize import linear_sum_assignment, curve_fit

__all__ = ['get_model_Dnu', 'get_obs_Dnu']


def get_model_Dnu(mod_freq, mod_l, Dnu, numax, mod_n=None):
    
    """
    Calculate model Dnu around numax, by fitting freq vs n with 
    Gaussian envelope around numax weighted data points, or with
    observational uncertainty weighted data points (if obs_freq, 
    obs_efreq and obs_l are set).

    ----------
    Input:
    mod_freq: array_like[Nmode_mod]
        model's mode frequency
    mod_l: array_like[Nmode_mod]
        model's mode degree
    Dnu: float
        the p-mode large separation in muHz
    numax: float
        the frequency of maximum power in muHz

    ----------
    Optional input:
    obs_freq: array_like[Nmode_obs]
        observation's mode frequency
    obs_efreq: array_like[Nmode_obs]
        observation's mode frequency uncertainty
    obs_l: array_like[Nmode_obs]
        observation's mode degree

    ----------
    Return:
    mod_Dnu: float

    """

    # width estimates based on Yu+2018, Lund+2017, Li+2020
    k, b = 0.9638, -1.7145
    width = np.exp(k*np.log(numax) + b)

    # assign n
    l0 = mod_l==0
    mod_freq_l0 = np.copy(mod_freq)[l0]
    sidx = np.argsort(mod_freq_l0) 
    mod_freq_l0 = mod_freq_l0[sidx]
    mod_n = np.arange(len(mod_freq_l0)) if (mod_n is None) else np.copy(mod_n)[l0][sidx]
    # print(mod_n, mod_freq_l0)
    # sigma = 1/np.exp(-(mod_freq_l0-numax)**2./(2*width**2.))
    weight = np.exp(-(mod_freq_l0-numax)**2./(2*width**2.))
    idx = weight>1e-100
    if np.sum(idx)>2:
        p, _, _, _, _ = np.polyfit(mod_n[idx], mod_freq_l0[idx], 1, w=weight[idx], full=True)
        mod_Dnu, mod_eps = p[0], p[1]/p[0]
    else:
        mod_Dnu, mod_eps = np.nan, np.nan 


    return mod_Dnu, mod_eps


def get_obs_Dnu(obs_freq, obs_efreq=None, Dnu_guess=None, ifReturnEpsilon=False):
    
    """
    Calculate observed Dnu from l=0 radial modes.

    ----------
    Input:
    obs_freq: array_like[Nmode_obs]
        radial mode frequency

    ----------
    Optional input:
    obs_efreq: array_like[Nmode_obs]
        mode frequency uncertainty

    Dnu_guess: float
        An initial guess for the Dnu

    ----------
    Return:
    obs_Dnu: float

    """
    if obs_efreq is None: obs_efreq = np.ones(len(obs_freq))
    if Dnu_guess is None: Dnu_guess = np.median(np.diff(np.sort(obs_freq)))

    Nmodes = len(obs_freq)
    
    mode_n = np.arange(Nmodes) + np.floor(obs_freq[0]/Dnu_guess)
    for im in range(len(mode_n)-1):
        add = np.round((obs_freq[im+1]-obs_freq[im])/Dnu_guess)-1
        mode_n[(im+1):] = mode_n[(im+1):] + add 

    # ## 1 - all modes without curvature
    def func1(xdata, Dnu, epsp):
        return (xdata + epsp )*Dnu
    # print(obs_freq, obs_efreq,mode_n)
    popt, pcov = curve_fit(func1, mode_n, obs_freq, sigma=obs_efreq)
    perr = np.diag(pcov)**0.5

    obs_Dnu, obs_eDnu = popt[0], perr[0]
    obs_eps, obs_eeps = popt[1], perr[1]

    if ifReturnEpsilon:
        return obs_Dnu, obs_eDnu, obs_eps, obs_eeps
    else:
        return obs_Dnu, obs_eDnu
