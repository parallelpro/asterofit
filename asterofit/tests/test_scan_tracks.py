import numpy as np
import pytest

from asterofit.src.scan_tracks import ScanConfig, StarObs, extract_track_arrays, compute_star_track_result
from asterofit.tests.fixtures import OBSERVABLES, ESTIMATORS, make_track, make_stars, star_obs_from_dict


def base_config(**overrides):
    defaults = dict(
        if_classical=True, if_classical_independent=True, if_seismic=True,
        if_correct_surface=False, surface_correction_formula='cubic',
        require_negative_surface_correction=False,
        require_absolute_surface_correction_increase_with_nu=False,
        col_mode_freq='mode_freq', col_mode_l='mode_l', col_mode_n='mode_n',
        col_mode_inertia='mode_inertia', col_acoustic_cutoff='acoustic_cutoff',
        observables=OBSERVABLES, estimators=ESTIMATORS,
    )
    defaults.update(overrides)
    return ScanConfig(**defaults)


def test_extract_track_arrays_shapes():
    track = make_track(seed=0, Nmodel=6)
    config = base_config()
    arrays = extract_track_arrays(track, config)

    assert arrays.Nmodel == 6
    assert arrays.mod_params_all.shape == (6, len(OBSERVABLES))
    assert len(arrays.mode_freq_all) == 6
    assert len(arrays.mode_l_all) == 6
    assert len(arrays.mode_n_all) == 6
    # genuinely ragged -- not every model has the same mode count
    assert len({len(f) for f in arrays.mode_freq_all}) > 1


def test_col_mode_n_none_falls_back_to_index():
    track = make_track(seed=0, Nmodel=4)
    config = base_config(col_mode_n=None)
    arrays = extract_track_arrays(track, config)

    for mode_l, mode_n in zip(arrays.mode_l_all, arrays.mode_n_all):
        assert np.array_equal(mode_n, np.arange(len(mode_l)))


def test_compute_star_track_result_matches_true_model():
    # observations built directly from model 2 of this exact track -> a known,
    # (near-)exact match at that model.
    track = make_track(seed=100, Nmodel=6)
    target = 2
    obs_l = np.array(track['mode_l'][target])
    obs_freq = np.array(track['mode_freq'][target])
    dnu_true = 58 + target * 0.5

    star = StarObs(
        obs_freq=obs_freq, obs_e_freq=np.full(len(obs_freq), 0.05), obs_l=obs_l.astype(float),
        obs_l_uniq=np.unique(obs_l), obs_N_l_uniq=len(np.unique(obs_l)),
        Dnu=dnu_true, numax=1200 + target*15 + 18.5*dnu_true,
        obs_params=np.array([float(track['Teff'][target]), float(track['FeH'][target]), float(track['luminosity'][target])]),
        e_obs_params=np.array([50., 0.05, 0.05]),
    )
    config = base_config()
    arrays = extract_track_arrays(track, config)
    result = compute_star_track_result(arrays, star, config)

    assert result['chi2_classical'][target] == pytest.approx(0.0, abs=1e-9)
    assert result['Dnu_freq'][target] == pytest.approx(dnu_true, abs=0.1)
    assert result['keep_mask'][target]


def test_compute_star_track_result_rejects_far_classical_mismatch():
    track = make_track(seed=0, Nmodel=6)
    config = base_config()
    arrays = extract_track_arrays(track, config)

    star = StarObs(
        obs_freq=np.array(track['mode_freq'][0]), obs_e_freq=np.full(len(track['mode_freq'][0]), 0.05),
        obs_l=np.array(track['mode_l'][0]).astype(float), obs_l_uniq=np.unique(track['mode_l'][0]),
        obs_N_l_uniq=len(np.unique(track['mode_l'][0])), Dnu=58., numax=2300.,
        obs_params=np.array([9000., 5.0, 100.0]),  # nowhere near any model in the track
        e_obs_params=np.array([50., 0.05, 0.05]),
    )
    result = compute_star_track_result(arrays, star, config)
    assert not np.any(result['keep_mask'])


def test_classical_covariance_equals_independent_when_diagonal():
    # mathematically, a diagonal covariance matrix with variance=e_obs_params**2
    # must give the same chi2 as the independent-uncertainty formula.
    track = make_track(seed=1, Nmodel=6)
    stars = make_stars(seed=42, Nstar=1)

    config_indep = base_config(if_classical_independent=True)
    config_cov = base_config(if_classical_independent=False)

    arrays_indep = extract_track_arrays(track, config_indep)
    arrays_cov = extract_track_arrays(track, config_cov)

    star_indep = star_obs_from_dict(stars[0], if_classical_independent=True)
    star_cov = star_obs_from_dict(stars[0], if_classical_independent=False)

    result_indep = compute_star_track_result(arrays_indep, star_indep, config_indep)
    result_cov = compute_star_track_result(arrays_cov, star_cov, config_cov)

    assert np.allclose(result_indep['chi2_classical'], result_cov['chi2_classical'])


def test_if_classical_false_scans_every_model():
    # regression test: idx_classical used to be the bare Python `True`, which
    # under array[True] fancy-indexing wrapped the whole array in a new axis
    # instead of iterating scalars -- silently corrupting the per-model scan.
    track = make_track(seed=2, Nmodel=5)
    stars = make_stars(seed=42, Nstar=1)
    config = base_config(if_classical=False)
    arrays = extract_track_arrays(track, config)
    star = star_obs_from_dict(stars[0], if_classical=False)

    result = compute_star_track_result(arrays, star, config)
    assert result['keep_mask'].shape == (5,)
    assert 'chi2_classical' not in result
    # at least one model should have been individually scored (finite Dnu_freq),
    # proving the loop iterated real integer indices, not a single corrupted one
    assert np.sum(np.isfinite(result['Dnu_freq'])) >= 1


def test_surface_correction_produces_expected_keys():
    track = make_track(seed=3, Nmodel=6)
    stars = make_stars(seed=42, Nstar=1)
    config = base_config(
        if_correct_surface=True, Nsurface=2, surface_estimators=['surf_a3', 'surf_corr_at_numax'],
        require_negative_surface_correction=True,
        require_absolute_surface_correction_increase_with_nu=True,
    )
    arrays = extract_track_arrays(track, config)
    star = star_obs_from_dict(stars[0])

    result = compute_star_track_result(arrays, star, config)
    for key in ['diff_freq_sc', 'mod_freq_sc', 'Dnu_freq_sc', 'eps_sc', 'surface_parameters']:
        assert key in result
    assert result['surface_parameters'].shape == (arrays.Nmodel, 2)
