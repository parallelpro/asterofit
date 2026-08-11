import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ['get_surface_correction', 'surface_params_dict']

surface_params_dict = {'cubic': ['surf_a3', 'surf_corr_at_numax'],
                       'combined': ['surf_a1', 'surf_a3', 'surf_corr_at_numax', 'surf_corr_at_1p1_numax'],
                       'kjeldsen': ['surf_a', 'surf_b', 'surf_corr_at_numax', 'surf_corr_at_1p1_numax']}


def get_surface_correction(obs_freq, obs_l, mod_freq, mod_l, mod_inertia, mod_acoustic_cutoff,
                            formula='cubic', if_full_output=False, Dnu=None, numax=None, ):
    """
    Correct model mode frequencies for the surface term -- the near-surface
    model deficiencies that make raw model frequencies systematically diverge
    from observations at high frequency -- by fitting one of three empirical
    formulas to the l=0 (radial) modes and applying it to every mode.

    l=0 model and observed frequencies are matched by closest frequency
    (linear sum assignment, same idea as `.matching.match_modes` but done
    locally here since only l=0 modes are needed) to build the regression
    target `residual = obs_freq_l0 - mod_freq_l0`, then a small weighted
    linear regression (via the normal equations, not `np.linalg.lstsq`) is
    solved for the formula's free coefficients:
    - 'cubic': Ball & Gizon (2014) cubic-only term,
      `delta_freq = a3 * (freq/acoustic_cutoff)^3 / inertia`.
    - 'combined': Ball & Gizon (2014) combined inverse + cubic term,
      `delta_freq = (a1*(freq/acoustic_cutoff)^-1 + a3*(freq/acoustic_cutoff)^3) / inertia`.
    - 'kjeldsen': Kjeldsen et al. (2008) power law,
      `delta_freq = a*(freq/inertia)^b`, fit in log space and only usable when
      at least 3 l=0 modes have `residual < 0` (a precondition of the log fit);
      note this is currently NaN-producing even when that precondition holds,
      since the log fit takes `log()` of those same negative residuals --
      pre-existing in the original implementation, not something callers can
      currently work around.
    If the regression fails (e.g. a singular matrix) or (kjeldsen) too few
    modes qualify, the correction coefficients are set to zero (i.e. no
    correction applied) and a message is printed.

    `delta_freq` is then evaluated for *every* mode (not just l=0) using its
    own frequency/inertia, and added to `mod_freq` -- so non-radial modes get
    corrected too, extrapolating the l=0-only fit, per the physical assumption
    that the surface term depends on frequency and inertia but not degree.

    ----------
    Input:
    obs_freq, obs_l: array-like[Nmode_obs]
        Observed mode frequency and angular degree; only the l=0 entries are
        used (there must be at least one, and for 'kjeldsen' at least 3 with
        a negative model-minus-observed sense to satisfy its precondition).
    mod_freq, mod_l, mod_inertia: array-like[Nmode_mod]
        Model mode frequency, angular degree, and (weighted) mode inertia.
    mod_acoustic_cutoff: float
        The model's acoustic cutoff frequency (one value per model, not per
        mode) -- used to non-dimensionalize frequency in 'cubic'/'combined'.
    formula: 'cubic' | 'combined' | 'kjeldsen', default 'cubic'
        Which surface-correction formula to fit.

    ----------
    Optional input:
    if_full_output: bool, default False
        If True, also return `surface_params` (see Output).
    Dnu, numax: float or None
        If both given (and `if_full_output=True`), `surface_params` also
        includes the correction interpolated to numax (and to 1.1*numax, for
        'combined'/'kjeldsen') -- otherwise those entries are NaN.

    ----------
    Output:
    new_mod_freq: array-like[Nmode_mod]
        `mod_freq` with the fitted surface correction added, for every mode.
    surface_params: array-like, only if if_full_output=True
        The fitted coefficients (`surf_a3`/`surf_corr_at_numax` for 'cubic';
        `surf_a1, surf_a3, ...` for 'combined'; `surf_a, surf_b, ...` for
        'kjeldsen' -- see `surface_params_dict` for the exact names/order),
        followed by the correction at numax (and 1.1*numax where applicable).

    """
    if not (formula in ['cubic', 'combined', 'kjeldsen']):
        raise ValueError('formula must be one of ``cubic``, ``combined`` and ``kjeldsen''. ')

    new_mod_freq = np.array(mod_freq)

    if (np.sum(np.isin(mod_l, 0))) :
        # if correction is needed, first we use l=0 modes to derive correction factors
        # if obs_l don't have a 0, well I am not expecting this!
        obs_freq_l0 = obs_freq[obs_l==0]
        new_mod_freq_l0 = new_mod_freq[mod_l==0]
        mod_inertia_l0 = mod_inertia[mod_l==0]

        # because we don't know n_p or n_g from observation (if we do that will save tons of effort here)
        # we need to assign each obs mode with a model mode
        # this can be seen as a linear sum assignment problem, also known as minimun weight matching in bipartite graphs
        cost = np.abs(obs_freq_l0.reshape(-1,1) - new_mod_freq_l0)
        matched_obs_idx, matched_mod_idx = linear_sum_assignment(cost)
        obs_freq_l0 = obs_freq_l0[matched_obs_idx]
        new_mod_freq_l0 = new_mod_freq_l0[matched_mod_idx]
        mod_inertia_l0 = mod_inertia_l0[matched_mod_idx]

        # regression
        residual = obs_freq_l0-new_mod_freq_l0
        # # avoid selecting reversed models
        # if (np.abs(np.median(np.diff(np.sort(obs_freq_l0)))) > np.abs(np.median(np.diff(np.sort(mod_freq_l0)))) ) :
        #     return None

        if formula == 'combined':
            inv_freq_term = (new_mod_freq_l0/mod_acoustic_cutoff)**-1. / mod_inertia_l0
            cubic_freq_term = (new_mod_freq_l0/mod_acoustic_cutoff)**3. / mod_inertia_l0
            design_matrix_T = np.array([inv_freq_term, cubic_freq_term])
            design_matrix = design_matrix_T.T
            residual = residual.reshape(-1,1)

            # apply corrections
            try:
                fit_coeff = np.dot(np.dot(np.linalg.inv(np.dot(design_matrix_T,design_matrix)), design_matrix_T), residual)
                fit_coeff = fit_coeff.reshape(-1)
                delta_freq = (fit_coeff[0]*(new_mod_freq/mod_acoustic_cutoff)**-1.  + fit_coeff[1]*(new_mod_freq/mod_acoustic_cutoff)**3. ) / mod_inertia
                new_mod_freq += delta_freq

                if ((numax != None) & (Dnu != None)):
                    surf_corr_at_numax = np.interp(numax, new_mod_freq, delta_freq)
                    surf_corr_at_1p1_numax = np.interp(1.1*numax, new_mod_freq, delta_freq)
                    surface_params = np.concatenate([fit_coeff, [surf_corr_at_numax, surf_corr_at_1p1_numax]])
                else:
                    surface_params = np.concatenate([fit_coeff, [np.nan, np.nan]])

            except:
                fit_coeff = np.zeros(4)
                print('An exception occurred when correcting surface effect using combined form.')
                # pass

        if formula == 'cubic':
            cubic_freq_term = (new_mod_freq_l0/mod_acoustic_cutoff)**3. / mod_inertia_l0
            design_matrix_T = np.array([cubic_freq_term])
            design_matrix = design_matrix_T.T
            residual = residual.reshape(-1,1)

            # apply corrections
            try:
                fit_coeff = np.dot(np.dot(np.linalg.inv(np.dot(design_matrix_T,design_matrix)), design_matrix_T), residual)
                fit_coeff = fit_coeff.reshape(-1)
                delta_freq = ( fit_coeff[0]*(new_mod_freq/mod_acoustic_cutoff)**3. ) / mod_inertia
                new_mod_freq += delta_freq

                if ((numax != None) & (Dnu != None)):
                    surf_corr_at_numax = np.interp(numax, new_mod_freq, delta_freq)
                    surface_params = np.concatenate([fit_coeff, [surf_corr_at_numax]])
                else:
                    surface_params = np.concatenate([fit_coeff, [np.nan]])

            except:
                fit_coeff = np.zeros(1)
                print('An exception occurred when correcting surface effect using cubic form.')
                # pass

        if formula == 'kjeldsen':
            if np.sum(residual<0)>3:
                negative_residual_mask = residual<0
                const_term = np.ones(len(new_mod_freq_l0[negative_residual_mask]))
                log_freq_term = np.log(new_mod_freq_l0[negative_residual_mask]/mod_inertia_l0[negative_residual_mask])
                design_matrix_T = np.array([const_term, log_freq_term])
                design_matrix = np.swapaxes(design_matrix_T, 0, 1)
                residual = np.log(residual[negative_residual_mask]).reshape(-1,1)

                # apply corrections
                try:
                    fit_coeff = np.dot(np.dot(np.linalg.inv(np.dot(design_matrix_T,design_matrix)), design_matrix_T), residual)
                    fit_coeff = fit_coeff.reshape(-1)
                    fit_coeff[0] = np.exp(fit_coeff[0])
                    delta_freq = ( fit_coeff[0]*(new_mod_freq/mod_inertia)**fit_coeff[1] )
                    new_mod_freq += delta_freq

                    if ((numax != None) & (Dnu != None)):
                        surf_corr_at_numax = np.interp(numax, new_mod_freq, delta_freq)
                        surf_corr_at_1p1_numax = np.interp(1.1*numax, new_mod_freq, delta_freq)
                        surface_params = np.concatenate([fit_coeff, [surf_corr_at_numax, surf_corr_at_1p1_numax]])
                    else:
                        surface_params = np.concatenate([fit_coeff, [np.nan, np.nan]])


                except:
                    fit_coeff = np.zeros(2)
                    print('An exception occurred when correcting surface effect using kjeldsen form.')
                    # pass
            else:
                fit_coeff = np.zeros(2)
                print('Using kjeldsen form, not enough (at least 3) modes with negative difference.')
                # pass
    if if_full_output:
        return new_mod_freq, surface_params
    else:
        return new_mod_freq
