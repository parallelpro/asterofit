import numpy as np

__all__ = ['compute_chi2_seismic', 'compute_chi2_total']


def compute_chi2_seismic(diff_freq, star, config) -> dict:
    '''
    Combine per-mode squared frequency residuals (already matched, selected, and
    collapsed across all tracks -- see .scan_tracks.compute_star_track_result) into
    per-degree and total seismic chi2, adding model systematic uncertainty and
    low-order-mode regularization as configured.

    Pure function of its inputs. Returns a dict keyed by the same
    'chi2_seismic*_l{l}' names process_star_results used to write directly, plus the
    summed 'chi2_seismic'.

    Fixes a bug present in the original single-loop version: `for degree_idx in
    range(obs_N_l_uniq): ... for degree_idx, l in enumerate(obs_l_uniq): ...`
    shadowed the outer loop's `degree_idx`/`l` with an inner loop that re-ran over
    every degree on every outer iteration. Nmode/Nnreg/Nreg (used as the
    reduced-chi2 divisor) were computed from the outer loop's degree but only the
    last outer pass's writes survived -- so every degree's chi2 ended up divided
    by the *last* degree's mode count, not its own, whenever
    `if_reduce_seis_chi2=True` (the project default) and a star has more than one
    observed l. Looping once over `enumerate(obs_l_uniq)` removes the duplication
    and the bug with it.

    ----------
    Input:
    diff_freq: array-like[Nmodel, Nmode_obs]
        Squared observed-minus-model frequency residuals, already matched,
        selected, and collapsed across all tracks for this star (i.e.
        `StarResults['diff_freq_sc']` if if_correct_surface, else
        `StarResults['diff_freq']`, after `StarResults.collapse()`).
    star: StarObs
        This star's observational data -- `obs_l`/`obs_l_uniq`/`obs_e_freq`
        (to build the per-degree masks and weights) and `mod_e_freq` (if
        add_model_error_method is 2 or 3).
    config: ScanConfig
        Which systematic-error/regularization behavior to use (see its
        docstring's ".process_star_results" fields section) and the
        reduce-chi2 toggles.

    ----------
    Output:
    result: dict
        'chi2_seismic_l{l}': array-like[Nmodel], for every observed degree l
            -- the (possibly regularized, possibly model-error-inflated)
            reduced chi2 for that degree, normalized by its own mode count.
        'chi2_seismic_obs_l{l}' (and, depending on config,
        'chi2_seismic_obs_mod_l{l}'/'..._reg_l{l}'/'..._nreg_l{l}'): the
            intermediate quantities `chi2_seismic_l{l}` is built from --
            see the branches below for exactly which are present.
        'chi2_seismic': array-like[Nmodel]
            Sum of `chi2_seismic_l{l}` over every observed degree -- the
            column `compute_chi2_total` reads as this star's seismic chi2.

    '''
    obs_e_freq = star.obs_e_freq
    obs_l = star.obs_l
    obs_l_uniq = star.obs_l_uniq
    obs_l_masks = [obs_l==l for l in obs_l_uniq]

    result = {}
    for degree_idx, l in enumerate(obs_l_uniq):
        l_mask = obs_l_masks[degree_idx]
        Nnreg = np.sum(l_mask)-config.Nreg if config.if_reduce_seis_chi2 else 1
        Nreg = config.Nreg if config.if_reduce_seis_reg_chi2 else 1
        Nmode = np.sum(l_mask) if config.if_reduce_seis_chi2 else 1

        if config.if_add_model_error:
            if config.if_regularize:
                # if_regularize is documented (params.py) as "not properly implemented
                # yet" and is never enabled in this project's configs. `mod_e_freq`
                # below is intentionally left undefined, exactly as in the original --
                # only the *_nreg/*_reg variants are computed in this branch -- so this
                # still raises NameError if ever reached, unchanged from before.
                if config.add_model_error_method == 1:
                    mod_e_freq_nreg = np.percentile(np.mean(diff_freq[:,obs_l_masks[degree_idx][config.Nreg:]], axis=1), config.rescale_percentile)
                    mod_e_freq_reg = np.percentile(np.mean(diff_freq[:,obs_l_masks[degree_idx][:config.Nreg]], axis=1), config.rescale_percentile)
                elif config.add_model_error_method == 2:
                    mod_e_freq_nreg, mod_e_freq_reg = star.mod_e_freq, star.mod_e_freq
                else: # config.add_model_error_method == 3:
                    error_ratio = star.mod_e_freq/np.min(obs_e_freq[obs_l_masks[degree_idx][config.Nreg:]])
                    error_ratio = error_ratio if error_ratio>1 else 1
                    mod_e_freq_nreg = (error_ratio**2. - 1)**0.5 * obs_e_freq[obs_l_masks[degree_idx][config.Nreg:]]
                    error_ratio = star.mod_e_freq/np.min(obs_e_freq[obs_l_masks[degree_idx][:config.Nreg]])
                    error_ratio = error_ratio if error_ratio>1 else 1
                    mod_e_freq_reg = (error_ratio**2. - 1)**0.5 * obs_e_freq[obs_l_masks[degree_idx][:config.Nreg]]
                result['chi2_seismic_obs_mod_nreg_l{:0.0f}'.format(l)] = np.sum(diff_freq[:,obs_l_masks[degree_idx][config.Nreg:]]/(obs_e_freq[obs_l_masks[degree_idx][config.Nreg:]]**2.0 + mod_e_freq_nreg**2.0), axis=1) / Nnreg
                result['chi2_seismic_obs_mod_reg_l{:0.0f}'.format(l)] = np.sum(diff_freq[:,obs_l_masks[degree_idx][:config.Nreg]]/(obs_e_freq[obs_l_masks[degree_idx][:config.Nreg]]**2.0 + mod_e_freq_reg**2.0), axis=1) / Nreg
                result['chi2_seismic_obs_mod_l{:0.0f}'.format(l)] = np.sum(diff_freq[:,l_mask]/(obs_e_freq[l_mask]**2.0 + mod_e_freq**2.0), axis=1) / Nmode
                result['chi2_seismic_obs_l{:0.0f}'.format(l)] = np.sum(diff_freq[:,l_mask]/obs_e_freq[l_mask]**2.0, axis=1) / Nmode
                result['chi2_seismic_l{:0.0f}'.format(l)] = (1-config.weight_reg) * result['chi2_seismic_obs_mod_nreg_l{:0.0f}'.format(l)] + config.weight_reg * result['chi2_seismic_obs_mod_reg_l{:0.0f}'.format(l)]
            else:
                if config.add_model_error_method == 1:
                    mod_e_freq = np.percentile(np.mean(diff_freq[:,l_mask], axis=1), config.rescale_percentile)
                elif config.add_model_error_method == 2:
                    mod_e_freq = star.mod_e_freq
                else: # config.add_model_error_method == 3:
                    error_ratio = star.mod_e_freq/np.min(obs_e_freq[l_mask])
                    error_ratio = error_ratio if error_ratio>1 else 1
                    mod_e_freq = (error_ratio**2. - 1)**0.5 * obs_e_freq[l_mask]
                result['chi2_seismic_obs_mod_l{:0.0f}'.format(l)] = np.sum(diff_freq[:,l_mask]/(obs_e_freq[l_mask]**2.0 + mod_e_freq**2.0), axis=1) / Nmode
                result['chi2_seismic_obs_l{:0.0f}'.format(l)] = np.sum(diff_freq[:,l_mask]/obs_e_freq[l_mask]**2.0, axis=1) / Nmode
                result['chi2_seismic_l{:0.0f}'.format(l)] = result['chi2_seismic_obs_mod_l{:0.0f}'.format(l)]
        else:
            if config.if_regularize:
                result['chi2_seismic_obs_nreg_l{:0.0f}'.format(l)] = np.sum(diff_freq[:,obs_l_masks[degree_idx][config.Nreg:]]/(obs_e_freq[obs_l_masks[degree_idx][config.Nreg:]]**2.0), axis=1) / Nnreg
                result['chi2_seismic_obs_reg_l{:0.0f}'.format(l)] = np.sum(diff_freq[:,obs_l_masks[degree_idx][:config.Nreg]]/(obs_e_freq[obs_l_masks[degree_idx][:config.Nreg]]**2.0), axis=1) / Nreg
                result['chi2_seismic_obs_l{:0.0f}'.format(l)] = np.sum(diff_freq[:,l_mask]/obs_e_freq[l_mask]**2.0, axis=1) / Nmode
                result['chi2_seismic_l{:0.0f}'.format(l)] = (1-config.weight_reg) * result['chi2_seismic_obs_nreg_l{:0.0f}'.format(l)] + config.weight_reg * result['chi2_seismic_obs_reg_l{:0.0f}'.format(l)]
            else:
                result['chi2_seismic_obs_l{:0.0f}'.format(l)] = np.sum(diff_freq[:,l_mask]/obs_e_freq[l_mask]**2.0, axis=1) / Nmode
                result['chi2_seismic_l{:0.0f}'.format(l)] = result['chi2_seismic_obs_l{:0.0f}'.format(l)]

    result['chi2_seismic'] = np.sum([result['chi2_seismic_l{:0.0f}'.format(l)] for l in obs_l_uniq], axis=0)
    return result


def compute_chi2_total(chi2_classical, chi2_seismic, config) -> np.ndarray:
    '''
    Combine classical and seismic chi2 into the single 'chi2' column
    `output_results()` ranks and weights models by -- if only one of
    if_classical/if_seismic is set, the other input is ignored entirely
    (and may be None).

    ----------
    Input:
    chi2_classical, chi2_seismic: array-like[Nmodel] or None
        Per-model chi2 from the classical and seismic constraints,
        respectively (e.g. `StarResults['chi2_classical']`/`['chi2_seismic']`
        after `process_star_results()` has collapsed and computed them).
        Either may be None if the corresponding `config` flag is False.
    config: ScanConfig
        `if_classical`/`if_seismic` select which input(s) to use;
        `weight_classical`/`weight_seismic` scale them before combining.

    ----------
    Output:
    chi2: array-like[Nmodel]
        `weight_classical*chi2_classical` if only if_classical,
        `weight_seismic*chi2_seismic` if only if_seismic, or their weighted
        sum if both.

    '''
    if config.if_classical and not config.if_seismic:
        return np.array(config.weight_classical * chi2_classical, dtype=float)
    elif (not config.if_classical) and config.if_seismic:
        return np.array(config.weight_seismic * chi2_seismic, dtype=float)
    else:
        return np.array(config.weight_classical * chi2_classical + config.weight_seismic * chi2_seismic, dtype=float)


if __name__ == "__main__":
    # a tiny, self-contained smoke test demonstrating the Nmode fix: two l-degrees
    # with different mode counts should each be normalized by their own count.
    try:
        from .scan_tracks import StarObs, ScanConfig
    except ImportError:
        from scan_tracks import StarObs, ScanConfig

    obs_l = np.array([0., 0., 0., 2., 2., 2., 2., 2.])
    obs_e_freq = np.full(8, 0.1)
    diff_freq = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])  # 1 model x 8 modes

    star = StarObs(obs_e_freq=obs_e_freq, obs_l=obs_l, obs_l_uniq=np.array([0., 2.]))
    config = ScanConfig(
        if_classical=False, if_classical_independent=True, if_seismic=True,
        if_correct_surface=False, surface_correction_formula='cubic',
        require_negative_surface_correction=False,
        require_absolute_surface_correction_increase_with_nu=False,
        col_mode_freq='mode_freq', col_mode_l='mode_l', col_mode_n=None,
        col_mode_inertia='mode_inertia', col_acoustic_cutoff='acoustic_cutoff',
        observables=[], estimators=[],
        if_add_model_error=False, if_regularize=False, if_reduce_seis_chi2=True,
    )

    result = compute_chi2_seismic(diff_freq, star, config)
    expected_l0 = np.sum(diff_freq[0, obs_l==0] / obs_e_freq[obs_l==0]**2.0) / 3  # 3 l=0 modes
    expected_l2 = np.sum(diff_freq[0, obs_l==2] / obs_e_freq[obs_l==2]**2.0) / 5  # 5 l=2 modes
    print('chi2_seismic_obs_l0:', result['chi2_seismic_obs_l0'], 'expected:', expected_l0)
    print('chi2_seismic_obs_l2:', result['chi2_seismic_obs_l2'], 'expected:', expected_l2)
    assert np.allclose(result['chi2_seismic_obs_l0'], expected_l0)
    assert np.allclose(result['chi2_seismic_obs_l2'], expected_l2)
    print('OK: each degree normalized by its own mode count.')
