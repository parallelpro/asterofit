from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from .matching import match_modes
    from .dnu import get_model_Dnu
    from .surface_correction import get_surface_correction
except ImportError:
    # allow `python3 scan_tracks.py` directly (e.g. for the self-check below), not just `-m`
    from matching import match_modes
    from dnu import get_model_Dnu
    from surface_correction import get_surface_correction

__all__ = ['ScanConfig', 'StarObs', 'TrackArrays', 'extract_track_arrays', 'compute_star_track_result']


@dataclass
class ScanConfig:
    '''
    Static, run-level settings needed by the pure kernels in .scan_tracks,
    .process_star_results, and .output_results -- no observational data.
    Built once per grid run (from the validated `params` dict, in
    `grid.__init__`) and shared by all three, so those functions never need
    `self`.

    ----------
    Fields used by .scan_tracks (this module):
    if_classical, if_classical_independent: bool
        Whether to score models against classical observables at all, and
        (if so) whether their uncertainties are independent (True) or given
        as a covariance matrix (False) -- see `StarObs.e_obs_params` vs.
        `.cinv_obs_params`.
    if_seismic: bool
        Whether to match/score individual mode frequencies at all.
    if_correct_surface: bool
        Whether to apply a surface-correction formula before matching.
    surface_correction_formula: 'cubic' | 'combined' | 'kjeldsen'
        Which formula -- see `.surface_correction.get_surface_correction`.
    require_negative_surface_correction: bool
        If True, discard models whose surface correction is positive for any
        mode (physically, the correction should always be negative).
    require_absolute_surface_correction_increase_with_nu: bool
        If True, discard models whose l=0 surface correction doesn't grow in
        magnitude with frequency (the expected qualitative behavior).
    col_mode_freq, col_mode_l, col_mode_inertia, col_acoustic_cutoff: str
        Column names in the track table for model mode frequency, angular
        degree, mode inertia, and (per-model, not per-mode) acoustic cutoff.
    col_mode_n: str or None
        Column name for model radial order n, or None to fall back to
        `arange(len(mode_l))` per model (see `extract_track_arrays`).
    observables, estimators: list[str]
        Classical observable column names (compared to `StarObs.obs_params`)
        and estimator column names (the quantities saved for surviving
        models) in the track table.
    Nsurface: int
    surface_estimators: list[str]
        Number of, and column names for, the surface-correction coefficients
        (only meaningful if if_correct_surface; see `surface_params_dict`).

    ----------
    Fields used by .output_results:
    estimators_to_summary, estimators_to_plot: list[str]
        Columns included in the per-star summary tables/corner plots,
        respectively (a superset/subset of `estimators` -- see
        `summarize_star_results`).

    ----------
    Fields used by .process_star_results (chi2 systematic-error modeling and weighting):
    if_add_model_error: bool
    add_model_error_method: 1 | 2 | 3
        Whether/how to add a model systematic uncertainty term when combining
        seismic chi2 -- see `compute_chi2_seismic`.
    rescale_percentile: float
        Used by add_model_error_method 1.
    if_regularize: bool
    Nreg: int
        Whether/how many low-order modes get a separately-weighted chi2
        contribution. `if_regularize=True` is not fully implemented (see
        `compute_chi2_seismic`'s docstring) and unused by every config in
        this project.
    if_reduce_seis_chi2, if_reduce_seis_reg_chi2: bool
        Whether to divide seismic chi2 (respectively, the regularized part)
        by the number of modes contributing to it.
    weight_classical, weight_seismic, weight_reg: float
        Multiplicative weights combining classical+seismic chi2, and
        regularized+non-regularized seismic chi2, respectively.

    '''

    if_classical: bool
    if_classical_independent: bool
    if_seismic: bool
    if_correct_surface: bool
    surface_correction_formula: str
    require_negative_surface_correction: bool
    require_absolute_surface_correction_increase_with_nu: bool
    col_mode_freq: str
    col_mode_l: str
    col_mode_n: Optional[str]
    col_mode_inertia: str
    col_acoustic_cutoff: str
    observables: list
    estimators: list
    Nsurface: int = 0
    surface_estimators: list = field(default_factory=list)
    # used by .output_results (sample/quantile summaries for output_results)
    estimators_to_summary: list = field(default_factory=list)
    estimators_to_plot: list = field(default_factory=list)
    # used by .process_star_results (chi2 systematic-error modeling and weighting)
    if_add_model_error: bool = False
    add_model_error_method: int = 2
    rescale_percentile: float = 0
    if_regularize: bool = False
    Nreg: int = 0
    if_reduce_seis_chi2: bool = True
    if_reduce_seis_reg_chi2: bool = True
    weight_classical: float = 1
    weight_seismic: float = 1
    weight_reg: float = 1


@dataclass
class StarObs:
    '''
    One star's observational data -- everything the pure kernels in
    .scan_tracks and .process_star_results need about a star. Built once per
    star in `grid.read_data()`
    from whichever of its fields apply (governed by `ScanConfig.if_classical`/
    `if_classical_independent`/`if_seismic`/`if_add_model_error`); fields that
    don't apply are left `None`.

    ----------
    Fields (seismic; if_seismic):
    obs_freq, obs_e_freq, obs_l: array-like[Nmode_obs]
        Observed mode frequency, frequency uncertainty, and angular degree.
    obs_l_uniq: array-like
    obs_N_l_uniq: int
        The unique angular degrees observed, and how many there are.
    Dnu, numax: float
        Large frequency separation and frequency of maximum power.
    mod_e_freq: float or None
        Per-star model systematic frequency uncertainty, only set if
        `if_add_model_error` with `add_model_error_method` 2 or 3 (see
        `compute_chi2_seismic`) -- otherwise None even if if_seismic.

    ----------
    Fields (classical; if_classical):
    obs_params: array-like[Nobservable]
        Observed classical values, in the order of `ScanConfig.observables`.
    e_obs_params: array-like[Nobservable] or None
        1-sigma uncertainties, only set if if_classical_independent.
    cinv_obs_params: array-like[Nobservable, Nobservable] or None
        Inverse covariance matrix, only set if not if_classical_independent.

    '''

    obs_freq: Optional[np.ndarray] = None
    obs_e_freq: Optional[np.ndarray] = None
    obs_l: Optional[np.ndarray] = None
    obs_l_uniq: Optional[np.ndarray] = None
    obs_N_l_uniq: Optional[int] = None
    Dnu: Optional[float] = None
    numax: Optional[float] = None
    obs_params: Optional[np.ndarray] = None
    e_obs_params: Optional[np.ndarray] = None
    cinv_obs_params: Optional[np.ndarray] = None
    mod_e_freq: Optional[float] = None


@dataclass
class TrackArrays:
    '''
    Per-model quantities that depend only on the track, not on which star is
    being scored -- built once per track by `extract_track_arrays` and reused
    for every star, instead of re-extracting them once per (star, model) pair.

    ----------
    Fields:
    Nmodel: int
        Number of models (rows) in this track.
    mod_params_all: array-like[Nmodel, Nobservable] or None
        Classical observable columns (`ScanConfig.observables`), only set if
        if_classical.
    mode_freq_all, mode_l_all, mode_n_all: list[array-like], length Nmodel, or None
        Per-model mode frequency/angular degree/radial order arrays (ragged --
        each model can have a different number of modes), only set if
        if_seismic. `mode_n_all` falls back to `arange(len(mode_l))` per model
        when `ScanConfig.col_mode_n` is None.
    mode_inertia_all: list[array-like], length Nmodel, or None
    acoustic_cutoff_all: array-like[Nmodel] or None
        Per-model mode inertia (ragged, like the arrays above) and (scalar
        per model, not per mode) acoustic cutoff frequency; only set if
        if_seismic and if_correct_surface.

    '''

    Nmodel: int
    mod_params_all: Optional[np.ndarray] = None
    mode_freq_all: Optional[list] = None
    mode_l_all: Optional[list] = None
    mode_n_all: Optional[list] = None
    mode_inertia_all: Optional[list] = None
    acoustic_cutoff_all: Optional[np.ndarray] = None


def extract_track_arrays(track_table, config: ScanConfig) -> TrackArrays:
    '''
    Extract the per-model arrays once per track, instead of once per (star, model)
    pair -- these values are identical for every star being scored against this track.

    ----------
    Input:
    track_table: table-like[Nmodel]
        One track's models, as returned by the user-supplied `read_models`
        (must support `track_table[column_name]` and `len(track_table)`).
    config: ScanConfig
        Determines which columns get extracted (`if_classical`/`if_seismic`/
        `if_correct_surface` gate the classical/seismic/surface-correction
        arrays respectively) and which column names to read them from.

    ----------
    Output:
    track: TrackArrays
        See `TrackArrays`'s docstring for exactly which fields get populated
        under which `config` flags (the rest are left None).

    '''
    Nmodel = len(track_table)
    track = TrackArrays(Nmodel=Nmodel)

    if config.if_classical:
        track.mod_params_all = np.array([track_table[col] for col in config.observables]).T.reshape(Nmodel, -1)

    if config.if_seismic:
        freq_column = track_table[config.col_mode_freq]
        l_column = track_table[config.col_mode_l]
        track.mode_freq_all = [np.array(freq_column[model_idx]) for model_idx in range(Nmodel)]
        track.mode_l_all = [np.array(l_column[model_idx]) for model_idx in range(Nmodel)]
        if config.col_mode_n is not None:
            n_column = track_table[config.col_mode_n]
            track.mode_n_all = [np.array(n_column[model_idx]) for model_idx in range(Nmodel)]
        else:
            track.mode_n_all = [np.arange(len(l)) for l in track.mode_l_all]

        if config.if_correct_surface:
            inertia_column = track_table[config.col_mode_inertia]
            track.mode_inertia_all = [np.array(inertia_column[model_idx]) for model_idx in range(Nmodel)]
            track.acoustic_cutoff_all = np.asarray(track_table[config.col_acoustic_cutoff])

    return track


def compute_star_track_result(track: TrackArrays, star: StarObs, config: ScanConfig) -> dict:
    '''
    Score one star against every model in one track: classical chi2, matched seismic
    frequencies/Dnu/surface-correction, and the boolean selection mask. Pure function
    of its inputs -- no shared/instance state read or mutated.

    Classical: chi2 (weighted sum-of-squares, or the full covariance form if
    `not if_classical_independent`) is computed for every model at once;
    models with chi2 < 16 (4-sigma) pass. Seismic: only for models that
    passed the classical cut (or all models, if if_classical is False) --
    `.matching.match_modes` pairs this model's mode frequencies with the
    star's observed ones (skipped if the model has fewer modes than
    observed), `.dnu.get_model_Dnu` fits the model Dnu/epsilon from the
    matched frequencies, and (if if_correct_surface, and only once Dnu is
    within 15% of the star's Dnu and the model has l=0 modes)
    `.surface_correction.get_surface_correction` is applied and the whole
    matching+Dnu step repeated on the corrected frequencies. A model's
    seismic mask bit requires Dnu within 15% of the star's Dnu (and, if
    if_correct_surface, a finite corrected Dnu, plus whichever of
    `require_negative_surface_correction`/
    `require_absolute_surface_correction_increase_with_nu` are set).

    ----------
    Input:
    track: TrackArrays
        This track's per-model arrays (from `extract_track_arrays`).
    star: StarObs
        This star's observational data.
    config: ScanConfig
        Which of the above to compute, and how (see its docstring).

    ----------
    Output:
    result: dict
        'keep_mask': array-like[Nmodel], bool
            True for models that passed both the classical and seismic cuts
            (whichever apply) -- the caller (`grid.scan_tracks`) uses this to
            select which models' columns actually get saved.
        'chi2_classical': array-like[Nmodel], only if if_classical.
        'diff_freq', 'mod_freq': array-like[Nmodel, Nmode_obs]
        'Dnu_freq', 'eps': array-like[Nmodel]
            Only if if_seismic. Squared frequency residuals and matched model
            frequencies (Nmodel are all NaN/unset for models that didn't reach
            the matching step); fitted model Dnu/epsilon.
        'diff_freq_sc', 'mod_freq_sc', 'Dnu_freq_sc', 'eps_sc',
        'surface_parameters': as above, for the surface-corrected frequencies,
            plus the fitted surface-correction coefficients
            (array-like[Nmodel, Nsurface]) -- only if if_correct_surface.

    '''
    Nmodel = track.Nmodel
    result = {}

    if config.if_classical:
        if config.if_classical_independent:
            chi2_classical = np.sum((star.obs_params-track.mod_params_all)**2.0/(star.e_obs_params**2.0), axis=1)
        else:
            param_diff = track.mod_params_all[:,:] - star.obs_params[None,:]
            chi2_classical = np.einsum('ij,jk,ik->i', param_diff, star.cinv_obs_params, param_diff)
        idx_classical = chi2_classical < 16. # 4-sigma
        result['chi2_classical'] = chi2_classical
    else:
        idx_classical = np.ones(Nmodel, dtype=bool) # all

    if config.if_seismic:
        obs_freq = star.obs_freq
        obs_e_freq = star.obs_e_freq
        obs_l = star.obs_l
        obs_l_unique = star.obs_l_uniq
        Nmode = len(obs_freq)

        # initialize the variables to calculate & save
        Dnu_freq = np.zeros(Nmodel, dtype=float) + np.nan
        eps = np.zeros(Nmodel, dtype=float) + np.nan
        diff_freq = np.zeros((Nmodel, Nmode), dtype=float) + np.nan
        mod_freq = np.zeros((Nmodel, Nmode), dtype=float) + np.nan
        if config.if_correct_surface:
            surface_parameters = np.zeros((Nmodel,config.Nsurface), dtype=float) + np.nan
            Dnu_freq_sc = np.zeros(Nmodel, dtype=float) + np.nan
            eps_sc = np.zeros(Nmodel, dtype=float) + np.nan
            diff_freq_sc = np.zeros((Nmodel, Nmode), dtype=float) + np.nan
            mod_freq_sc = np.zeros((Nmodel, Nmode), dtype=float) + np.nan

        for model_idx in np.arange(0, Nmodel)[idx_classical]:
            # get 1) Dnu from frquencies, 2) squared differences
            mode_freq = track.mode_freq_all[model_idx]
            mode_l = track.mode_l_all[model_idx]
            mode_n = track.mode_n_all[model_idx]

            if len(mode_freq) < len(obs_freq) : continue

            obs_freq_matched, _, _, mode_freq_matched, mode_l_matched, mode_n_matched = match_modes(obs_freq, obs_e_freq, obs_l, mode_freq, mode_l, mode_n)

            if len(mode_freq_matched) < len(obs_freq): continue

            Dnu_freq[model_idx], eps[model_idx] = get_model_Dnu(mode_freq_matched, mode_l_matched, star.Dnu, star.numax, mode_n_matched)
            diff_freq[model_idx, :] = (obs_freq_matched-mode_freq_matched)**2.0
            mod_freq[model_idx, :] = mode_freq_matched

            # get 1) Dnu, 2) squared differences,
            # but for the surface correction version, if there is any
            if (config.if_correct_surface) & \
                (np.abs((Dnu_freq[model_idx]-star.Dnu)/star.Dnu)<0.15 ) & \
                (np.sum(np.isin(mode_l, 0))) :

                mode_inertia = track.mode_inertia_all[model_idx]
                acoustic_cutoff = track.acoustic_cutoff_all[model_idx]

                mode_freq_sc, surface_parameters[model_idx,:] = get_surface_correction(obs_freq, obs_l, \
                                                                mode_freq, mode_l, \
                                                                mode_inertia, acoustic_cutoff, \
                                                                formula=config.surface_correction_formula, \
                                                                if_full_output=True, \
                                                                Dnu=star.Dnu, numax=star.numax)

                obs_freq_matched, _, _, mode_freq_sc_matched, mode_l_sc_matched, mode_n_sc_matched = match_modes(obs_freq, obs_e_freq, obs_l, mode_freq_sc, mode_l, mode_n)
                Dnu_freq_sc[model_idx], eps_sc[model_idx] = get_model_Dnu(mode_freq_sc_matched, mode_l_sc_matched, star.Dnu, star.numax, mode_n_sc_matched)
                diff_freq_sc[model_idx, :] = (obs_freq_matched-mode_freq_sc_matched)**2.0
                mod_freq_sc[model_idx, :] = mode_freq_sc_matched

        idx_seismic = (np.abs((Dnu_freq-star.Dnu)/star.Dnu)<0.15 )
        if config.if_correct_surface: idx_seismic = idx_seismic & np.isfinite(Dnu_freq_sc)
        if config.if_correct_surface & config.require_negative_surface_correction:
            idx_seismic = idx_seismic & (np.sum(mod_freq_sc - mod_freq > 0, axis=1) == 0 )
        if config.if_correct_surface & config.require_absolute_surface_correction_increase_with_nu & (0 in obs_l_unique):
            idx_upper_nu = np.argmax(obs_freq[obs_l==0])
            idx_lower_nu = np.argmin(obs_freq[obs_l==0])
            diff_freq_l0 = mod_freq_sc[:, obs_l==0] - mod_freq[:, obs_l==0]
            idx_seismic = idx_seismic & (np.abs(diff_freq_l0[:, idx_upper_nu]) - np.abs(diff_freq_l0[:, idx_lower_nu]) > 0)

        result['diff_freq'] = diff_freq
        result['mod_freq'] = mod_freq
        result['Dnu_freq'] = Dnu_freq
        result['eps'] = eps
        if config.if_correct_surface:
            result['diff_freq_sc'] = diff_freq_sc
            result['mod_freq_sc'] = mod_freq_sc
            result['Dnu_freq_sc'] = Dnu_freq_sc
            result['eps_sc'] = eps_sc
            result['surface_parameters'] = surface_parameters
    else:
        idx_seismic = True

    result['keep_mask'] = idx_classical & idx_seismic
    return result


if __name__ == "__main__":
    # a tiny, self-contained smoke test -- no CSVs, no grid instance required.
    from astropy.table import Table

    rng = np.random.default_rng(0)
    Nmodel = 4
    ls_full = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    ns_full = np.array([17, 18, 19, 17, 18, 19, 17, 18, 19])

    mode_freq_col = np.empty(Nmodel, dtype=object)
    mode_l_col = np.empty(Nmodel, dtype=object)
    for model_idx in range(Nmodel):
        n_modes = len(ls_full) - (model_idx % 2)  # keep it genuinely ragged
        mode_freq_col[model_idx] = 1200 + model_idx*15 + ns_full[:n_modes]*58 + ls_full[:n_modes]*6.5 + rng.normal(0, 0.05, n_modes)
        mode_l_col[model_idx] = ls_full[:n_modes]

    track_table = Table({
        'Teff': 5700. + rng.normal(0, 50, Nmodel),
        'luminosity': 1.0 + rng.normal(0, 0.05, Nmodel),
        'mode_freq': mode_freq_col,
        'mode_l': mode_l_col,
    })

    config = ScanConfig(
        if_classical=True, if_classical_independent=True, if_seismic=True,
        if_correct_surface=False, surface_correction_formula='cubic',
        require_negative_surface_correction=False,
        require_absolute_surface_correction_increase_with_nu=False,
        col_mode_freq='mode_freq', col_mode_l='mode_l', col_mode_n=None,
        col_mode_inertia='mode_inertia', col_acoustic_cutoff='acoustic_cutoff',
        observables=['Teff', 'luminosity'], estimators=['Teff', 'luminosity'],
    )
    star = StarObs(
        obs_freq=np.array(track_table['mode_freq'][1]), obs_e_freq=np.full(len(track_table['mode_freq'][1]), 0.05),
        obs_l=np.array(track_table['mode_l'][1]), obs_l_uniq=np.unique(track_table['mode_l'][1]),
        obs_N_l_uniq=len(np.unique(track_table['mode_l'][1])),
        Dnu=58., numax=1200 + 1*15 + 18*58.,
        obs_params=np.array([float(track_table['Teff'][1]), float(track_table['luminosity'][1])]),
        e_obs_params=np.array([50., 0.05]),
    )

    track = extract_track_arrays(track_table, config)
    result = compute_star_track_result(track, star, config)
    print('keep_mask (models kept):', result['keep_mask'])
    print('Dnu_freq:', result['Dnu_freq'])
    print('chi2_classical:', result['chi2_classical'])
