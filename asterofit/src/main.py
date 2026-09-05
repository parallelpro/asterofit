from warnings import resetwarnings
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from astropy.io import ascii
from astropy.table import Table, Column
import h5py

# from scipy.spatial import distance
# from scipy.special import logsumexp

from .tools import return_2dmap_axes, plot_seis_echelles, plot_parameter_distributions
from .surface_correction import surface_params_dict
from .star_results import StarResults, concat
from .defaults import compulsory_global_params, default_global_params
from .scan_tracks import ScanConfig, StarObs, extract_track_arrays, compute_star_track_result
from .process_star_results import compute_chi2_seismic, compute_chi2_total
from .output_results import summarize_star_results

__all__ = ['grid']


class grid:
    """

    Estimate stellar parameters by comparing a grid of evolutionary tracks
    against observed classical (Teff, [Fe/H], luminosity, ...) and/or seismic
    (individual mode frequencies) constraints, for one or more stars at once.

    Usage:
    >>> g = grid(read_models, tracks, params)
    >>> g.run()

    `run()` drives a three-stage pipeline, each stage optionally split across
    `self.Nthread` worker processes/threads:
    1. scan_tracks     -- score every model in every track against every star's
                           observations (classical chi2, seismic mode matching
                           and Dnu, optional surface correction); keep only the
                           models that pass loose classical/seismic cuts.
    2. process_star_results -- collapse the surviving per-track models into
                           flat per-star arrays and compute the final seismic
                           and combined chi2.
    3. output_results  -- write corner plots, summary tables, echelle diagrams,
                           and/or an HDF5 dump per star under `filepath_output`.

    After `run()` returns, `self.star_results` holds one `StarResults` object
    per star (indexed the same way as `self.starIDs`) with every column listed
    in `self.keys[istar]` -- this is the same data that gets written to
    `data.h5` when `if_data=True`.

    The actual per-model scoring math lives in the pure, `self`-free functions
    in `.scan_tracks`/`.process_star_results`/`.output_results` (each named
    after, and called from, the `grid` method of the same name below); this
    class is the orchestration layer around them (config validation, data
    loading, track/star iteration, parallelism, and file output).

    """

    def __init__(self, read_models, tracks, params):

        """
        Validate and store the run configuration, then load the observational
        data (via `read_data()`) so the instance is ready for `run()`.

        ----------
        Input:
        read_models: function reference
            The function takes one argument 'track_path' which is one element
            of `tracks`, and returns a structured array, or a table-like object
            (like the one defined in 'astropy'). The track column of an
            observable/estimate 'i' defined in `params['estimators']` /
            `params['observables']` should be able to be called with
            'track_table[i]'.
            Return `None` to skip a track -- useful when a track file is
            missing or fails to load; `scan_tracks()` will silently skip it.
        tracks: array-like[Ntrack,]
            A list containing the paths (or any other per-track identifier
            `read_models` understands) of all tracks to scan.
        params: dict
            Run configuration. Must contain every key listed in
            `defaults.compulsory_global_params` (observables, estimators,
            estimators_to_plot, filepath_stellar_params, col_starIDs); any key
            missing from `defaults.default_global_params` falls back to its
            default there. See `params.py` for a fully-documented example of
            every supported key (classical/seismic toggles, column-name
            mappings, surface-correction settings, output settings, ...).
            Mutated in place: missing defaulted keys are inserted before every
            key is set as an attribute on `self` (so e.g. `params['if_seismic']`
            becomes `self.if_seismic`).

        ----------
        Sets (in addition to one attribute per `params` key):
        self.Nestimate, self.Nobservable, self.Ntrack: int
            Convenience counts derived from `estimators`, `observables`, `tracks`.
        self.starIDs, self.Nstar, and (depending on if_classical/if_seismic)
        self.obs_params, self.e_obs_params/self.cinv_obs_params, self.Dnu,
        self.numax, self.obs_freq, self.obs_e_freq, self.obs_l, self.obs_l_uniq,
        self.obs_N_l_uniq, self.star_obs:
            Loaded by `read_data()` -- see that method's docstring.
        self.estimators_to_summary: array-like[str]
            `estimators` plus the seismic/surface-correction derived columns
            (Dnu_freq, eps, ...) that get written to the per-star summary
            tables in `output_results()`.
        self.config: ScanConfig
            The static, run-level settings needed by the pure kernels in
            `.scan_tracks`/`.process_star_results`/`.output_results` --
            everything above, repackaged into one plain object so those
            functions don't need `self`.

        """

        self.read_models = read_models
        self.tracks = np.array(tracks)

        # set up params and pass them into class attributes
        for param_name in compulsory_global_params :
            if not (param_name in list(params.keys()) ):
                raise ValueError('parameter {:s} not set.'.format(param_name))

        for param_name in default_global_params:
            if not (param_name in list(params.keys()) ):
                params[param_name] = default_global_params[param_name]

        for param_name in params.keys():
            setattr(self, param_name, params[param_name])

        # handy numbers
        self.Nestimate = len(self.estimators)
        self.Nobservable = len(self.observables)
        self.Ntrack = len(self.tracks)

        # read in data
        self.read_data()

        # set up output dir
        if not os.path.exists(self.filepath_output): os.mkdir(self.filepath_output)

        # seismic
        self.estimators_to_summary = np.array(self.estimators)
        if self.if_seismic:
            self.estimators_to_summary = np.concatenate([self.estimators_to_summary, ['Dnu_freq', 'eps']])

            # surface corrections
            if self.if_correct_surface:
                self.surface_estimators = surface_params_dict[self.surface_correction_formula]
                self.Nsurface = len(self.surface_estimators)
                self.estimators_to_summary = np.concatenate([self.estimators_to_summary, ['Dnu_freq_sc', 'eps_sc'], self.surface_estimators])

                if self.surface_correction_formula == 'prescribed':
                    if self.surface_prescription is None or len(np.atleast_1d(self.surface_prescription)) != 8:
                        raise ValueError("surface_correction_formula 'prescribed' needs "
                                         "'surface_prescription' (the 8 power-law parameters).")
                    if None in (self.col_model_numax, self.col_model_teff, self.col_model_feh):
                        raise ValueError("surface_correction_formula 'prescribed' needs "
                                         "'col_model_numax'/'col_model_teff'/'col_model_feh'.")
                    self.surface_prescription = np.asarray(self.surface_prescription, dtype=float)

        # static, run-level settings for the (self-free) pure kernels in .scan_tracks, .process_star_results, and .output_results
        self.config = ScanConfig(
            if_classical=self.if_classical,
            if_classical_independent=self.if_classical_independent,
            if_seismic=self.if_seismic,
            if_correct_surface=self.if_correct_surface,
            surface_correction_formula=self.surface_correction_formula,
            require_negative_surface_correction=self.require_negative_surface_correction,
            require_absolute_surface_correction_increase_with_nu=self.require_absolute_surface_correction_increase_with_nu,
            col_mode_freq=self.col_mode_freq,
            col_mode_l=self.col_mode_l,
            col_mode_n=self.col_mode_n,
            col_mode_inertia=self.col_mode_inertia,
            col_acoustic_cutoff=self.col_acoustic_cutoff,
            observables=self.observables,
            estimators=self.estimators,
            Nsurface=getattr(self, 'Nsurface', 0),
            surface_estimators=getattr(self, 'surface_estimators', []),
            surface_prescription=self.surface_prescription,
            col_model_numax=self.col_model_numax,
            col_model_teff=self.col_model_teff,
            col_model_feh=self.col_model_feh,
            if_add_model_error=self.if_add_model_error,
            add_model_error_method=self.add_model_error_method,
            rescale_percentile=self.rescale_percentile,
            if_regularize=self.if_regularize,
            Nreg=self.Nreg,
            if_reduce_seis_chi2=self.if_reduce_seis_chi2,
            if_reduce_seis_reg_chi2=self.if_reduce_seis_reg_chi2,
            weight_classical=self.weight_classical,
            weight_seismic=self.weight_seismic,
            weight_reg=self.weight_reg,
            estimators_to_summary=self.estimators_to_summary,
            estimators_to_plot=self.estimators_to_plot,
        )

        return


    def read_data(self):
        """
        Load the observational data for every star and bundle it for the scan
        kernel. Called once from `__init__`; not meant to be called directly.

        ----------
        Reads:
        self.filepath_stellar_params: csv
            One row per star, indexed by `self.col_starIDs`. If `if_classical`,
            must have a column per `self.observables` entry, plus either
            'e_<observable>' columns (if `if_classical_independent`) or
            'c_<obs_i>_<obs_j>' covariance columns for every pair (otherwise).
            If `if_seismic`, must also have `self.col_obs_Dnu`/`col_obs_numax`
            columns, and (if `if_add_model_error` with method 2 or 3) a
            `self.col_model_error` column.
        self.filepath_stellar_freqs: csv
            Only read if `if_seismic`. One row per observed mode, with columns
            `self.col_starIDs`/`col_obs_freq`/`col_obs_e_freq`/`col_obs_l`.
            Stars with no matching rows get empty frequency arrays rather than
            raising -- see `output_results()`'s handling of 'too_few_models'.

        ----------
        Sets:
        self.starIDs, self.Nstar
            Star identifiers (as strings) and their count, in file order --
            every other per-star array in this class is indexed the same way.
        self.obs_params, self.e_obs_params (if if_classical_independent)
        or self.c_obs_params/self.cinv_obs_params (otherwise)
            Classical observables and their 1-sigma uncertainties or
            (inverse) covariance matrix, one row/matrix per star.
        self.Dnu, self.numax, self.obs_freq, self.obs_e_freq, self.obs_l,
        self.obs_l_uniq, self.obs_N_l_uniq (if if_seismic)
            Per-star scalars (Dnu, numax) and per-star ragged arrays (mode
            frequency/uncertainty/degree, and the unique degrees observed).
        self.star_obs: list[StarObs]
            One `StarObs` per star, bundling whichever of the above apply
            (classical and/or seismic) into the plain, `self`-free structure
            `.scan_tracks`/`.process_star_results` actually consume. Purely additive: the parallel
            arrays above are untouched, since `process_star_results()` and
            `output_results()` still read them directly.

        """
        # read in starIDs:
        data_stellar_params = pd.read_csv(self.filepath_stellar_params)
        data_stellar_params[self.col_starIDs] = data_stellar_params[self.col_starIDs].astype('str')
        self.starIDs = data_stellar_params[self.col_starIDs].to_numpy()
        self.Nstar = len(self.starIDs)

        # read in stellar params
        if self.if_classical:
            observables = self.observables
            e_observables = ['e_'+s for s in self.observables]
            self.obs_params = data_stellar_params[observables].to_numpy()

            if self.if_classical_independent:
                # 1-σ uncertainty
                self.e_obs_params = data_stellar_params[e_observables].to_numpy()
            else:
                # covariance matrix
                self.c_obs_params = np.zeros((self.Nstar, self.Nobservable, self.Nobservable))
                self.cinv_obs_params = np.zeros((self.Nstar, self.Nobservable, self.Nobservable))

                for obs_i in range(0, self.Nobservable):
                    for obs_j in range(0, obs_i+1):
                        col_name_ij = 'c_'+observables[obs_i]+'_'+observables[obs_j]
                        col_name_ji = 'c_'+observables[obs_j]+'_'+observables[obs_i]
                        if (col_name_ij in data_stellar_params.columns):
                            self.c_obs_params[:, obs_i, obs_j] = data_stellar_params[col_name_ij].to_numpy()
                            self.c_obs_params[:, obs_j, obs_i] = data_stellar_params[col_name_ij].to_numpy()
                        elif (col_name_ji in data_stellar_params.columns):
                            self.c_obs_params[:, obs_i, obs_j] = data_stellar_params[col_name_ji].to_numpy()
                            self.c_obs_params[:, obs_j, obs_i] = data_stellar_params[col_name_ji].to_numpy()
                        else:
                            raise ValueError('{} not found in the stellar parameter table.'.format(col_name_ij))

                for istar in range(self.Nstar):
                    self.cinv_obs_params[istar, :, :] = np.linalg.inv(self.c_obs_params[istar, :, :])


        # read in stellar frequencies
        if self.if_seismic:
            self.Dnu = data_stellar_params[self.col_obs_Dnu].to_numpy()
            self.numax = data_stellar_params[self.col_obs_numax].to_numpy()

            if self.if_add_model_error & (self.add_model_error_method in [2, 3]):
                self.mod_e_freq = data_stellar_params[self.col_model_error].to_numpy()

            data_stellar_freqs = pd.read_csv(self.filepath_stellar_freqs)
            data_stellar_freqs[self.col_starIDs] = data_stellar_freqs[self.col_starIDs].astype('str')

            # group once instead of re-scanning the whole frequency table for every star
            freqs_by_star = {star_id: star_freq_rows for star_id, star_freq_rows in data_stellar_freqs.groupby(self.col_starIDs)}
            empty_freqs = data_stellar_freqs.iloc[0:0]

            self.obs_freq, self.obs_e_freq, self.obs_l = [np.empty(self.Nstar, dtype=object) for _ in range(3)]
            for istar in range(self.Nstar):
                star_freq_rows = freqs_by_star.get(self.starIDs[istar], empty_freqs)
                # `.matching.match_modes` returns the matched modes grouped by
                # ascending l, and `.process_star_results.compute_chi2_seismic`
                # indexes those matched arrays with masks built from `obs_l`.
                # The two line up only if the observations are held sorted by
                # l, so sort here rather than making it an undocumented
                # precondition on the input file: a frequency-sorted mode
                # table (the natural way to write one) would otherwise have
                # its degrees and uncertainties silently mismatched. Stable,
                # so the row order within a degree stays the file's.
                star_freq_rows = star_freq_rows.sort_values(self.col_obs_l, kind='stable')
                self.obs_freq[istar] = star_freq_rows[self.col_obs_freq].to_numpy()
                self.obs_e_freq[istar] = star_freq_rows[self.col_obs_e_freq].to_numpy()
                self.obs_l[istar] = star_freq_rows[self.col_obs_l].to_numpy()

            self.obs_l_uniq = np.array([np.unique(l) for l in self.obs_l], dtype=object)
            self.obs_N_l_uniq = np.array([len(np.unique(l)) for l in self.obs_l])

        # bundle each star's observational data for the (self-free) scan kernel in .scan_tracks.
        # additive only -- the parallel arrays above are untouched, since process_star_results
        # and output_results still read them directly.
        self.star_obs = []
        for istar in range(self.Nstar):
            self.star_obs.append(StarObs(
                obs_freq=self.obs_freq[istar] if self.if_seismic else None,
                obs_e_freq=self.obs_e_freq[istar] if self.if_seismic else None,
                obs_l=self.obs_l[istar] if self.if_seismic else None,
                obs_l_uniq=self.obs_l_uniq[istar] if self.if_seismic else None,
                obs_N_l_uniq=self.obs_N_l_uniq[istar] if self.if_seismic else None,
                Dnu=self.Dnu[istar] if self.if_seismic else None,
                numax=self.numax[istar] if self.if_seismic else None,
                obs_params=self.obs_params[istar] if self.if_classical else None,
                e_obs_params=self.e_obs_params[istar] if (self.if_classical and self.if_classical_independent) else None,
                cinv_obs_params=self.cinv_obs_params[istar] if (self.if_classical and not self.if_classical_independent) else None,
                mod_e_freq=(self.mod_e_freq[istar]
                            if (self.if_seismic and self.if_add_model_error and self.add_model_error_method in [2, 3])
                            else None),
            ))

        return


    def scan_tracks(self, thread_track_idx=None):
        '''
        Stage 1 of `run()`: score every model in every track against every
        star's observations, keeping only the models that pass loose cuts.

        For each track: load it via `self.read_models`, extract the per-model
        arrays once (`extract_track_arrays` -- these don't depend on which
        star is being scored, so hoisting them out of the star loop avoids
        redoing it once per star), then for each star call
        `compute_star_track_result` to get that track's classical chi2,
        matched seismic frequencies/Dnu (and surface correction, if
        configured), and a boolean mask of which models to keep. Only the
        kept models' columns are written into that star's `StarResults`, one
        track at a time -- tracks are not yet collapsed together (see
        `process_star_results()`).

        `self.star_results`, `self.keys`, and `self.star_obs`/`self.config`
        (from `read_data()`/`__init__`) must already exist -- in practice this
        means `run()` has already built `self.keys` before calling this.

        ----------
        Optional input:
        thread_track_idx: array-like[int] or None
            Indices into `self.tracks` to scan; `None` scans all of them.
            `run()` passes a contiguous chunk per worker when `Nthread > 1`
            and merges the per-worker results back together with `concat()`.

        ----------
        Output:
        list_of_star_results: list[StarResults], length self.Nstar
            One `StarResults` per star (same order as `self.starIDs`), each
            holding `len(thread_track_idx)` per-track ragged arrays -- collapse
            them (via `process_star_results()`) before reading a flat array.

        '''

        if (thread_track_idx is None): thread_track_idx = np.arange(0, self.Ntrack, dtype=int)
        tracks = self.tracks[thread_track_idx]
        Ntrack = len(tracks)

        list_of_star_results = [StarResults(Ntrack, self.keys[istar]) for istar in range(self.Nstar)]


        for itrack in range(Ntrack):

            # read in itrack
            track_table = self.read_models(tracks[itrack])
            if (track_table is None) : continue

            # per-model quantities that depend only on the track, not on istar,
            # extracted once per track instead of once per (istar, model_idx) pair.
            track_arrays = extract_track_arrays(track_table, self.config)

            # calculate posterior
            for istar in range(self.Nstar):
                result = compute_star_track_result(track_arrays, self.star_obs[istar], self.config)
                keep_mask = result['keep_mask']

                # save estimates
                for estimator_name in self.estimators:
                    list_of_star_results[istar][estimator_name, itrack] = np.array(track_table[estimator_name][keep_mask], dtype=float)

                if self.if_classical:
                    list_of_star_results[istar]['chi2_classical', itrack] = np.array(result['chi2_classical'][keep_mask], dtype=float)

                if self.if_seismic:
                    list_of_star_results[istar]['diff_freq', itrack] = np.array(result['diff_freq'][keep_mask], dtype=float)
                    list_of_star_results[istar]['mod_freq', itrack] = np.array(result['mod_freq'][keep_mask], dtype=float)
                    list_of_star_results[istar]['mod_n', itrack] = np.array(result['mod_n'][keep_mask], dtype=float)
                    list_of_star_results[istar]['mod_inertia', itrack] = np.array(result['mod_inertia'][keep_mask], dtype=float)
                    list_of_star_results[istar]['Dnu_freq', itrack] = np.array(result['Dnu_freq'][keep_mask], dtype=float)
                    list_of_star_results[istar]['eps', itrack] = np.array(result['eps'][keep_mask], dtype=float)

                    if self.if_correct_surface:
                        list_of_star_results[istar]['diff_freq_sc', itrack] = np.array(result['diff_freq_sc'][keep_mask], dtype=float)
                        list_of_star_results[istar]['mod_freq_sc', itrack] = np.array(result['mod_freq_sc'][keep_mask], dtype=float)
                        list_of_star_results[istar]['Dnu_freq_sc', itrack] = np.array(result['Dnu_freq_sc'][keep_mask], dtype=float)
                        list_of_star_results[istar]['eps_sc', itrack] = np.array(result['eps_sc'][keep_mask], dtype=float)
                        for surface_param_idx, surface_param_name in enumerate(self.surface_estimators):
                            list_of_star_results[istar][surface_param_name, itrack] = np.array(result['surface_parameters'][keep_mask, surface_param_idx], dtype=float)

        return list_of_star_results


    def process_star_results(self, thread_star_idx=None):
        '''
        Stage 2 of `run()`: collapse each star's per-track results from
        `scan_tracks()` into flat arrays and compute the final chi2.

        For each star: `collapse()` its `StarResults` in place (concatenates
        every track's surviving models together, removing the per-track
        dimension), then, if seismic, combine the per-mode squared frequency
        residuals into per-degree and total seismic chi2 via
        `compute_chi2_seismic` (adds model systematic uncertainty and
        low-order-mode regularization as configured) and write every returned
        key onto the star's `StarResults`. Finally combine classical and
        seismic chi2 with their configured weights (`compute_chi2_total`) into
        `'chi2'` -- the column `output_results()` ranks models by.

        Requires `self.star_results[istar]` to already hold the (uncollapsed)
        per-track results from `scan_tracks()`.

        ----------
        Optional input:
        thread_star_idx: array-like[int] or None
            Indices into `self.starIDs` to process; `None` processes all
            stars. `run()` passes a contiguous chunk per worker when
            `Nthread > 1`.

        ----------
        Output:
        list_of_star_results: list[StarResults], length len(thread_star_idx)
            The (now collapsed, chi2-populated) `StarResults` for the
            requested stars, in the same order as `thread_star_idx`.

        '''
        if (thread_star_idx is None): thread_star_idx = np.arange(0, self.Nstar, dtype=int)
        list_of_star_results = [[] for istar in thread_star_idx]

        for output_idx, istar in enumerate(thread_star_idx):
            self.star_results[istar].collapse(copy=False)

            if self.if_seismic:
                diff_freq = self.star_results[istar]['diff_freq_sc'] if self.if_correct_surface else self.star_results[istar]['diff_freq']
                chi2_seismic_result = compute_chi2_seismic(diff_freq, self.star_obs[istar], self.config)
                for result_key, result_value in chi2_seismic_result.items():
                    self.star_results[istar][result_key] = result_value

            chi2_classical = self.star_results[istar]['chi2_classical'] if self.if_classical else None
            chi2_seismic = self.star_results[istar]['chi2_seismic'] if self.if_seismic else None
            self.star_results[istar]['chi2'] = compute_chi2_total(chi2_classical, chi2_seismic, self.config)

            list_of_star_results[output_idx] = self.star_results[istar]
        return list_of_star_results


    def output_results(self, thread_star_idx=None):
        '''
        Stage 3 of `run()`: write each star's results to
        `self.filepath_output/{starID}/`, creating the directory if needed.

        For each star, summarizes its (already collapsed, chi2-populated)
        `StarResults` via `summarize_star_results` and, depending on that
        summary's status:
        - 'too_few_models' (fewer than 5 surviving models): writes only
          `log.txt` explaining the failure; nothing else for this star.
        - 'too_few_samples' (enough models, but too few to weight-quantile):
          same, with a different `log.txt` message, only if `self.if_plot`.
        - 'ok': if `self.if_plot`, writes corner plots (classical/seismic/
          combined probability), weighted-quantile summary tables, a
          best-models-by-each-chi2 table, and (if `self.if_seismic and
          self.if_plot_echelle`) an echelle diagram of the top-10 models by
          seismic chi2.

        Independently of the above (as long as there were enough models),
        if `self.if_data`, dumps every column in `self.keys[istar]` from the
        star's `StarResults` to `data.h5`.

        ----------
        Optional input:
        thread_star_idx: array-like[int] or None
            Indices into `self.starIDs` to write output for; `None` writes
            for all stars. `run()` passes a contiguous chunk per worker when
            `Nthread > 1`.

        ----------
        Output:
        None -- this method only has side effects (files written under
        `self.filepath_output`).

        '''

        if (thread_star_idx is None): thread_star_idx = np.arange(0, self.Nstar, dtype=int)

        for istar in thread_star_idx:
            star_output_dir = self.filepath_output + '{:s}'.format(self.starIDs[istar]) + '/'
            if not os.path.exists(star_output_dir):
                os.mkdir(star_output_dir)

            summary = summarize_star_results(self.star_results[istar], self.config)

            if summary['status'] == 'too_few_models':
                log_file = open(star_output_dir+'log.txt', 'w')
                log_file.write("Parameter estimation failed because fewer than 5 models have been selected.")
                log_file.close()
                continue

            if self.if_plot:
                if summary['status'] == 'too_few_samples':
                    log_file = open(star_output_dir+'log.txt', 'w')
                    log_file.write("Parameter estimation failed because samples.shape[0] <= samples.shape[1].")
                    log_file.close()
                else:
                    samples_to_plot = summary['samples_to_plot']

                    # plot prob distributions
                    if self.if_classical:
                        fig = plot_parameter_distributions(samples_to_plot, self.estimators_to_plot, summary['prob_classical'])
                        fig.savefig(star_output_dir+"corner_prob_classical.png")
                        plt.close()

                    if self.if_seismic:
                        fig = plot_parameter_distributions(samples_to_plot, self.estimators_to_plot, summary['prob_seismic'])
                        fig.savefig(star_output_dir+"corner_prob_seismic.png")
                        plt.close()

                    # output the prob (prior included)
                    fig = plot_parameter_distributions(samples_to_plot, self.estimators_to_plot, summary['prob'])
                    fig.savefig(star_output_dir+"corner_prob.png")
                    plt.close()

                    # write prob distribution summary file
                    if self.if_classical:
                        ascii.write(Table(summary['quantile_prob_classical'], names=self.estimators_to_summary), star_output_dir+"summary_prob_classical.txt",format="csv", overwrite=True)

                    if self.if_seismic:
                        ascii.write(Table(summary['quantile_prob_seismic'], names=self.estimators_to_summary), star_output_dir+"summary_prob_seismic.txt",format="csv", overwrite=True)

                    # output the prob (prior excluded)
                    ascii.write(Table(summary['quantile_prob'], names=self.estimators_to_summary), star_output_dir+"summary_prob.txt",format="csv", overwrite=True)

                    # output the best models
                    ascii.write(Table(summary['best_models_table'], names=np.concatenate([['best_model_by', 'chi2'], self.estimators_to_summary])), star_output_dir+"summary_best.txt",format="csv", overwrite=True)

                    # plot echelle diagrams
                    if self.if_seismic & self.if_plot_echelle:
                        chi2_seismic = summary['chi2_seismic']
                        top10_idx = np.argsort(chi2_seismic, axis=0)[:10]
                        mod_freq_sc = self.star_results[istar]['mod_freq_sc'][top10_idx] if self.if_correct_surface else None
                        fig = plot_seis_echelles(self.obs_freq[istar], self.obs_e_freq[istar], self.obs_l[istar],
                                self.star_results[istar]['mod_freq'][top10_idx], chi2_seismic[top10_idx], self.Dnu[istar], mod_freq_sc=mod_freq_sc)
                        fig.savefig(star_output_dir+"echelle_top10_prob_seismic.png")
                        plt.close()

            # write related parameters to file
            if self.if_data:
                with h5py.File(star_output_dir+'data.h5', 'w') as h5_file:
                    for star_results_key in self.keys[istar]:
                        h5_file.create_dataset(star_results_key, data=self.star_results[istar][star_results_key])

        return


    def run(self):
        """
        The main entry point -- call this after construction to actually
        estimate parameters. Runs the full pipeline and writes all output;
        see the class docstring for the three stages.

        First declares `self.keys[istar]`: the list of column names each
        star's `StarResults` will hold, which varies per star (it depends on
        that star's observed l-degrees) and per config (if_classical,
        if_seismic, if_correct_surface, if_add_model_error, if_regularize all
        add/remove columns). Every later stage relies on `self.keys` already
        being set.

        Then runs scan_tracks -> process_star_results -> output_results.
        If `self.Nthread == 1`, sequentially and in-process. Otherwise, all
        three stages share one `self.executor_class(max_workers=self.Nthread)`
        instance (default `concurrent.futures.ProcessPoolExecutor`, but
        swappable -- e.g. for `ThreadPoolExecutor`, or any class with the same
        interface): each stage's work is split into `self.Nthread` contiguous
        chunks (by track for scan_tracks, by star for the other two), mapped
        across the executor, and merged back with `concat()` between stages.
        Reusing one executor across all three stages avoids paying
        spawn/import/pickle-`self` overhead three times over.

        ----------
        Input: none (uses attributes set by `__init__`/`read_data`).

        ----------
        Output:
        None -- results end up on `self.star_results` (see the class
        docstring) and as files under `self.filepath_output` (see
        `output_results()`).

        """

        # # declaration
        self.keys = np.empty(self.Nstar, dtype=object)
        for istar in range(self.Nstar):
            star_results_keys = self.estimators + ['chi2']
            if self.if_classical: star_results_keys = star_results_keys + ['chi2_classical']
            if self.if_seismic:
                star_results_keys = star_results_keys + ['chi2_seismic']
                l_values = self.obs_l_uniq[istar]
                star_results_keys = star_results_keys + ['Dnu_freq', 'eps', 'diff_freq', 'mod_freq',
                                                         'mod_n', 'mod_inertia']
                star_results_keys = star_results_keys + ['chi2_seismic_l{:0.0f}'.format(l) for l in l_values]
                star_results_keys = star_results_keys + ['chi2_seismic_obs_l{:0.0f}'.format(l) for l in l_values]
                if self.if_add_model_error:
                    star_results_keys = star_results_keys + ['chi2_seismic_obs_mod_l{:0.0f}'.format(l) for l in l_values]
                    if self.if_regularize:
                        star_results_keys = star_results_keys + ['chi2_seismic_obs_mod_reg_l{:0.0f}'.format(l) for l in l_values]
                        star_results_keys = star_results_keys + ['chi2_seismic_obs_mod_nreg_l{:0.0f}'.format(l) for l in l_values]
                else:
                    if self.if_regularize:
                        star_results_keys = star_results_keys + ['chi2_seismic_obs_reg_l{:0.0f}'.format(l) for l in l_values]
                        star_results_keys = star_results_keys + ['chi2_seismic_obs_nreg_l{:0.0f}'.format(l) for l in l_values]
                if self.if_correct_surface:
                    star_results_keys = star_results_keys + ['diff_freq_sc', 'eps_sc', 'mod_freq_sc', 'Dnu_freq_sc'] + self.surface_estimators
            self.keys[istar] = star_results_keys


        if self.Nthread == 1:
            # # step 1, scan all tracks and collect star_results
            self.star_results = np.array(self.scan_tracks(), dtype=object)

            # # step 2, process star_results
            self.process_star_results()

            # # step 3: output results
            self.output_results()
        else:
            # one executor reused for all three stages, instead of spawning/importing/pickling
            # self three times over. self.executor_class defaults to
            # concurrent.futures.ProcessPoolExecutor but is swappable (e.g. for
            # ThreadPoolExecutor, or any class with the same interface).
            with self.executor_class(max_workers=self.Nthread) as executor:

                # # step 1, scan all tracks and collect star_results
                Ntrack_per_thread = int(self.Ntrack/self.Nthread)+1
                track_idx_chunks = [np.arange(0, self.Ntrack, dtype=int)[ithread*Ntrack_per_thread:(ithread+1)*Ntrack_per_thread] for ithread in range(self.Nthread)]

                thread_results = list(executor.map(self.scan_tracks, track_idx_chunks))

                # merge from different threads
                self.star_results = np.empty(self.Nstar, dtype=object)
                for istar in range(self.Nstar):
                    self.star_results[istar] = concat([thread_results[ithread][istar] for ithread in range(self.Nthread)])

                # # step 2, process star_results
                Nstar_per_thread = int(self.Nstar/self.Nthread)+1
                star_idx_chunks = [np.arange(0, self.Nstar, dtype=int)[ithread*Nstar_per_thread:(ithread+1)*Nstar_per_thread] for ithread in range(self.Nthread)]

                thread_results = list(executor.map(self.process_star_results, star_idx_chunks))

                # merge from different threads
                self.star_results = np.array([star_result for per_thread_results in thread_results for star_result in per_thread_results], dtype=object)

                # # step 3: output results
                Nstar_per_thread = int(self.Nstar/self.Nthread)+1
                star_idx_chunks = [np.arange(0, self.Nstar, dtype=int)[ithread*Nstar_per_thread:(ithread+1)*Nstar_per_thread] for ithread in range(self.Nthread)]

                list(executor.map(self.output_results, star_idx_chunks))

        return
