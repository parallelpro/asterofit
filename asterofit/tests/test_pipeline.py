import os

import numpy as np
import pandas as pd

import asterofit
from asterofit.tests.fixtures import make_track, read_models


def build_star_csvs(tmp_path, target_track_model=2, sort_modes_by=None):
    '''Write samples.csv/modes.csv for 4 synthetic stars: three with seismic
    data at different l-degree coverage, one with none at all.'''
    truth = make_track(seed=100, Nmodel=6)
    base_freq = np.array(truth['mode_freq'][target_track_model])
    base_l = np.array(truth['mode_l'][target_track_model])
    dnu_true = 58 + target_track_model * 0.5
    numax_true = 1200 + target_track_model * 15 + 18.5 * dnu_true

    l_masks = [
        np.isin(base_l, [0, 1, 2]),
        np.isin(base_l, [0, 2]),
        np.isin(base_l, [0, 1]),
    ]
    kics = ['1001', '1002', '1003', '1004']

    star_rows, mode_rows = [], []
    for i, kic in enumerate(kics):
        star_rows.append(dict(
            KIC=kic,
            Teff=float(truth['Teff'][target_track_model]), e_Teff=50.,
            FeH=float(truth['FeH'][target_track_model]), e_FeH=0.05,
            luminosity=float(truth['luminosity'][target_track_model]), e_luminosity=0.05,
            Dnu=dnu_true, e_Dnu=0.5, numax=numax_true, e_numax=5.0,
        ))
        if kic == '1004':
            continue  # deliberately no seismic rows for this star
        mask = l_masks[i]
        for f, l in zip(base_freq[mask], base_l[mask]):
            mode_rows.append(dict(KIC=kic, l=int(l), fc=f, e_fc=0.05))

    samples_path = tmp_path / 'samples.csv'
    modes_path = tmp_path / 'modes.csv'
    pd.DataFrame(star_rows).to_csv(samples_path, index=False)
    modes = pd.DataFrame(mode_rows)
    if sort_modes_by is not None:
        modes = modes.sort_values(['KIC', sort_modes_by], kind='stable')
    modes.to_csv(modes_path, index=False)
    return str(samples_path), str(modes_path)


def make_params(tmp_path, samples_path, modes_path, Nthread, executor_class=None):
    outdir = str(tmp_path / 'output') + '/'
    params = {
        "observables": ['Teff', 'FeH', 'luminosity'],
        "estimators": ['star_age', 'star_mass', 'Teff', 'luminosity'],
        "estimators_to_plot": ['star_age', 'star_mass'],
        "filepath_stellar_params": samples_path,
        "col_starIDs": 'KIC',
        "filepath_stellar_freqs": modes_path,
        "col_obs_freq": "fc", "col_obs_e_freq": "e_fc", "col_obs_l": "l",
        "col_obs_Dnu": "Dnu", "col_obs_numax": "numax",
        "if_plot": False, "if_data": False, "Nthread": Nthread,
        "filepath_output": outdir,
        "if_classical": True, "if_classical_independent": True, "weight_classical": 1,
        "if_seismic": True, "weight_seismic": 1,
        "col_mode_freq": "mode_freq", "col_mode_l": "mode_l", "col_mode_n": "mode_n",
        "col_mode_inertia": "mode_inertia", "col_acoustic_cutoff": "acoustic_cutoff",
        "if_reduce_seis_chi2": True,
        "if_correct_surface": False,
        "if_add_model_error": False,
    }
    if executor_class is not None:
        params["executor_class"] = executor_class
    return params


def run_pipeline(tmp_path, Nthread, executor_class=None, sort_modes_by=None):
    samples_path, modes_path = build_star_csvs(tmp_path, sort_modes_by=sort_modes_by)
    params = make_params(tmp_path, samples_path, modes_path, Nthread, executor_class=executor_class)
    tracks = [0, 1, 2]
    g = asterofit.grid(read_models, tracks, params)
    g.run()
    return g


def test_pipeline_runs_end_to_end(tmp_path):
    g = run_pipeline(tmp_path, Nthread=1)

    assert g.Nstar == 4
    for istar in range(3):  # stars with seismic data
        chi2 = g.star_results[istar]['chi2']
        assert len(chi2) > 0
        assert np.all(np.isfinite(chi2))

    # star with no seismic rows at all: still processed, just empty frequency arrays
    assert len(g.obs_freq[3]) == 0


def assert_star_results_match(g1, g2):
    assert g1.Nstar == g2.Nstar
    for istar in range(g1.Nstar):
        for key in g1.keys[istar]:
            v1 = np.asarray(g1.star_results[istar][key], dtype=float)
            v2 = np.asarray(g2.star_results[istar][key], dtype=float)
            assert v1.shape == v2.shape
            assert np.allclose(v1, v2, equal_nan=True)


def test_nthread_1_and_2_give_identical_results(tmp_path_factory):
    g1 = run_pipeline(tmp_path_factory.mktemp("nt1"), Nthread=1)
    g2 = run_pipeline(tmp_path_factory.mktemp("nt2"), Nthread=2)
    assert_star_results_match(g1, g2)


def test_executor_class_is_swappable(tmp_path_factory):
    # proves executor_class is a real seam, not just a ProcessPoolExecutor
    # in disguise: swap in ThreadPoolExecutor and check results still match
    # the Nthread=1 baseline.
    from concurrent.futures import ThreadPoolExecutor

    g1 = run_pipeline(tmp_path_factory.mktemp("nt1"), Nthread=1)
    g_threaded = run_pipeline(tmp_path_factory.mktemp("nt2-threaded"), Nthread=2, executor_class=ThreadPoolExecutor)
    assert_star_results_match(g1, g_threaded)


def test_mode_table_row_order_does_not_change_results(tmp_path_factory):
    '''
    Regression test for a bug where the row order of the observed mode table
    changed the seismic chi2. `match_modes` returns the matched modes grouped
    by ascending l, but `compute_chi2_seismic` masks those columns with
    `obs_l` as read from the file -- so a table sorted by frequency, where the
    degrees interleave, had each degree's residuals divided by another
    degree's uncertainties and summed into the wrong degree's chi2. The
    natural way to write a peakbagging table (one row per mode, in frequency
    order) hit it; a table that happened to be grouped by l did not.
    '''
    by_l = run_pipeline(tmp_path_factory.mktemp("by_l"), Nthread=1)
    by_freq = run_pipeline(tmp_path_factory.mktemp("by_freq"), Nthread=1,
                           sort_modes_by='fc')

    # guard: the frequency-sorted table must genuinely interleave degrees,
    # otherwise this test passes without exercising anything
    modes = pd.read_csv(by_freq.filepath_stellar_freqs)
    star_modes = modes[modes['KIC'] == 1001]['l'].to_numpy()
    assert not np.array_equal(star_modes, np.sort(star_modes, kind='stable'))

    assert_star_results_match(by_l, by_freq)
