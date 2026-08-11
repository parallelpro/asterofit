import numpy as np
import pytest

from asterofit.src.scan_tracks import ScanConfig, StarObs
from asterofit.src.process_star_results import compute_chi2_seismic, compute_chi2_total
from asterofit.tests.fixtures import OBSERVABLES, ESTIMATORS


def base_config(**overrides):
    defaults = dict(
        if_classical=True, if_classical_independent=True, if_seismic=True,
        if_correct_surface=False, surface_correction_formula='cubic',
        require_negative_surface_correction=False,
        require_absolute_surface_correction_increase_with_nu=False,
        col_mode_freq='mode_freq', col_mode_l='mode_l', col_mode_n='mode_n',
        col_mode_inertia='mode_inertia', col_acoustic_cutoff='acoustic_cutoff',
        observables=OBSERVABLES, estimators=ESTIMATORS,
        if_add_model_error=False, if_regularize=False, if_reduce_seis_chi2=True,
        weight_classical=1, weight_seismic=1,
    )
    defaults.update(overrides)
    return ScanConfig(**defaults)


@pytest.mark.parametrize("n_l0,n_l1,n_l2", [(3, 5, 8), (5, 5, 3), (10, 2, 4)])
def test_chi2_seismic_normalizes_each_degree_by_its_own_mode_count(n_l0, n_l1, n_l2):
    '''
    Regression test for a bug where an inner loop shadowed the outer loop's
    `il`/`l`, so every degree's reduced chi2 ended up divided by whichever
    degree happened to be *last* in obs_l_uniq instead of its own mode count.
    Parametrized over several (unequal, and one equal) mode-count combinations
    to make sure the fix holds regardless of which degree is "last".
    '''
    rng = np.random.default_rng(0)
    obs_l = np.concatenate([np.zeros(n_l0), np.ones(n_l1), np.full(n_l2, 2)])
    obs_e_freq = rng.uniform(0.05, 0.15, len(obs_l))
    diff_freq = rng.uniform(0.1, 10.0, (1, len(obs_l)))  # 1 model x Nmode

    star = StarObs(obs_e_freq=obs_e_freq, obs_l=obs_l, obs_l_uniq=np.array([0., 1., 2.]))
    config = base_config()

    result = compute_chi2_seismic(diff_freq, star, config)

    for l, n in zip([0., 1., 2.], [n_l0, n_l1, n_l2]):
        idx = obs_l == l
        expected = np.sum(diff_freq[0, idx] / obs_e_freq[idx]**2.0) / n
        assert result['chi2_seismic_obs_l{:0.0f}'.format(l)][0] == pytest.approx(expected)

    expected_total = sum(
        result['chi2_seismic_l{:0.0f}'.format(l)][0] for l in [0., 1., 2.]
    )
    assert result['chi2_seismic'][0] == pytest.approx(expected_total)


def test_chi2_seismic_reduce_flag_off_uses_unreduced_chi2():
    obs_l = np.array([0., 0., 0., 1., 1.])
    obs_e_freq = np.full(5, 0.1)
    diff_freq = np.array([[1., 2., 3., 4., 5.]])

    star = StarObs(obs_e_freq=obs_e_freq, obs_l=obs_l, obs_l_uniq=np.array([0., 1.]))
    config = base_config(if_reduce_seis_chi2=False)

    result = compute_chi2_seismic(diff_freq, star, config)
    expected_l0 = np.sum(diff_freq[0, obs_l==0] / obs_e_freq[obs_l==0]**2.0)  # divided by 1, not 3
    assert result['chi2_seismic_obs_l0'][0] == pytest.approx(expected_l0)


def test_chi2_total_weighting():
    chi2_classical = np.array([2.0, 4.0])
    chi2_seismic = np.array([3.0, 6.0])

    only_classical = compute_chi2_total(chi2_classical, chi2_seismic, base_config(if_classical=True, if_seismic=False))
    assert np.array_equal(only_classical, chi2_classical)

    only_seismic = compute_chi2_total(chi2_classical, chi2_seismic, base_config(if_classical=False, if_seismic=True))
    assert np.array_equal(only_seismic, chi2_seismic)

    both = compute_chi2_total(chi2_classical, chi2_seismic, base_config(if_classical=True, if_seismic=True, weight_classical=2, weight_seismic=0.5))
    assert np.array_equal(both, 2*chi2_classical + 0.5*chi2_seismic)
