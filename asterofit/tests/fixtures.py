'''
Shared synthetic-data builders for the asterofit test suite. Not a test module
itself (no test_* names) -- pytest won't try to collect anything from here.
'''

import numpy as np
from astropy.table import Table

from asterofit.src.scan_tracks import StarObs

OBSERVABLES = ['Teff', 'FeH', 'luminosity']
ESTIMATORS = ['star_age', 'star_mass', 'Teff', 'luminosity']


def to_ragged_object_column(list_of_arrays):
    '''
    Build a genuinely 1D array-of-arrays object column. Plain
    `np.array(list_of_arrays, dtype=object)` silently collapses into a 2D float
    array when every sub-array has the same length -- real track data (different
    acoustic cutoff per model) never has that problem, but synthetic fixtures can
    accidentally construct exactly that uniform case, so build it by explicit
    per-slot assignment instead.
    '''
    col = np.empty(len(list_of_arrays), dtype=object)
    for i, a in enumerate(list_of_arrays):
        col[i] = a
    return col


def make_track(seed, Nmodel=6):
    '''A synthetic evolutionary track: Nmodel models, each with a genuinely
    ragged (different mode count) set of p-modes across l=0,1,2.'''
    rng = np.random.default_rng(seed)
    data = {}
    data['Teff'] = 5700. + rng.normal(0, 80, Nmodel)
    data['FeH'] = rng.normal(0, 0.1, Nmodel)
    data['luminosity'] = 1.0 + rng.normal(0, 0.1, Nmodel)
    data['star_age'] = np.linspace(1e9, 5e9, Nmodel)
    data['star_mass'] = np.linspace(0.9, 1.3, Nmodel)
    data['acoustic_cutoff'] = 4500. + rng.normal(0, 100, Nmodel)

    ls_full = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    ns_full = np.array([16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20])

    mode_freq_list, mode_l_list, mode_n_list, mode_inertia_list = [], [], [], []
    for i in range(Nmodel):
        n_modes = len(ls_full) - (i % 3)  # keep the mode count genuinely ragged
        ls_pattern = ls_full[:n_modes]
        ns_pattern = ns_full[:n_modes]

        base = 1200 + i * 15
        dnu = 58 + i * 0.5
        freqs = base + ns_pattern * dnu + ls_pattern * 6.5 + rng.normal(0, 0.05, len(ls_pattern))
        inertia = rng.uniform(0.5, 2.0, len(ls_pattern))
        mode_freq_list.append(freqs)
        mode_l_list.append(ls_pattern.copy())
        mode_n_list.append(ns_pattern.copy())
        mode_inertia_list.append(inertia)

    data['mode_freq'] = to_ragged_object_column(mode_freq_list)
    data['mode_l'] = to_ragged_object_column(mode_l_list)
    data['mode_n'] = to_ragged_object_column(mode_n_list)
    data['mode_inertia'] = to_ragged_object_column(mode_inertia_list)

    return Table(data)


def make_stars(seed, target_track_model=2, Nstar=3, truth_seed=100):
    '''
    Synthetic "observed" stars whose frequencies are drawn from one specific
    model of an independent track (`truth_seed`), so scoring them against
    `make_track(seed=truth_seed)` gives an exact, known-correct match at
    `target_track_model` -- useful for asserting chi2 is near zero / Dnu_freq
    is near the true Dnu for that model.
    '''
    truth = make_track(truth_seed, Nmodel=max(6, target_track_model + 1))
    base_freq = np.array(truth['mode_freq'][target_track_model])
    base_l = np.array(truth['mode_l'][target_track_model])

    l_masks = [
        np.isin(base_l, [0, 1, 2]),
        np.isin(base_l, [0, 2]),
        np.isin(base_l, [0, 1]),
    ]
    dnu_true = 58 + target_track_model * 0.5
    numax_true = 1200 + target_track_model * 15 + 18.5 * dnu_true

    stars = []
    for istar in range(Nstar):
        mask = l_masks[istar % len(l_masks)]
        f = base_freq[mask].copy()
        l = base_l[mask].copy()
        e = np.full(len(f), 0.05)
        stars.append(dict(obs_freq=f, obs_e_freq=e, obs_l=l.astype(float),
                           Dnu=dnu_true, numax=numax_true,
                           Teff=float(truth['Teff'][target_track_model]),
                           FeH=float(truth['FeH'][target_track_model]),
                           luminosity=float(truth['luminosity'][target_track_model])))
    return stars


def star_obs_from_dict(s, if_classical=True, if_classical_independent=True, if_seismic=True):
    '''Build a StarObs from one of make_stars()'s plain dicts.'''
    return StarObs(
        obs_freq=s['obs_freq'] if if_seismic else None,
        obs_e_freq=s['obs_e_freq'] if if_seismic else None,
        obs_l=s['obs_l'] if if_seismic else None,
        obs_l_uniq=np.unique(s['obs_l']) if if_seismic else None,
        obs_N_l_uniq=len(np.unique(s['obs_l'])) if if_seismic else None,
        Dnu=s['Dnu'] if if_seismic else None,
        numax=s['numax'] if if_seismic else None,
        obs_params=np.array([s['Teff'], s['FeH'], s['luminosity']]) if if_classical else None,
        e_obs_params=(np.array([50., 0.05, 0.05])
                      if (if_classical and if_classical_independent) else None),
        cinv_obs_params=(np.linalg.inv(np.diag(np.array([50., 0.05, 0.05]) ** 2))
                          if (if_classical and not if_classical_independent) else None),
    )


def read_models(track_id):
    '''
    A `read_models` callable in the shape grid.__init__ expects (one argument,
    the track "path"). Defined at module level (not nested) so it stays
    picklable for multiprocessing.Pool workers.
    '''
    return make_track(seed=track_id)
