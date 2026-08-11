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

import multiprocessing
from functools import partial

# from scipy.spatial import distance
# from scipy.special import logsumexp

from .tools import return_2dmap_axes, quantile, plot_seis_echelles, plot_parameter_distributions
from .surface import get_surface_correction, surface_params_dict
from .Dnu import get_model_Dnu
from .container import stardata, concat
from .defaults import compulsory_global_params, default_global_params
from .matching import match_modes

__all__ = ['grid']


class grid:
    """
    
    Estimate stellar parameters from evolutionary grid models.

    """

    def __init__(self, read_models, tracks, params):

        """
        Initialize a parameter estimation class.

        ----------
        Input:
        read_models: function reference
            The function takes one argument 'atrack' which is the path of
            a track, and returns a structured array, or a table-like object 
            (like the one defined in 'astropy'). The track column of an 
            observable/estimate 'i' defined in self.setup should be able 
            to be called with 'atrack[i]'.
            Return a 'None' object to skip this track. This is useful for
            cases like the data is not properly laoding.
        tracks: array-like[Ntrack,]
            A list containing the paths of all tracks.
        params: a python dictionary containing various parameters

        """

        self.read_models = read_models
        self.tracks = np.array(tracks)

        # set up params and pass them into class attributes
        for param in compulsory_global_params :
            if not (param in list(params.keys()) ):
                raise ValueError('parameter {:s} not set.'.format(param)) 

        for param in default_global_params:
            if not (param in list(params.keys()) ):
                params[param] = default_global_params[param]

        for key in params.keys():
            setattr(self, key, params[key])

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

        return 


    def read_data(self):
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

                for i in range(0, self.Nobservable):
                    for j in range(0, i+1):
                        labelij = 'c_'+observables[i]+'_'+observables[j]
                        labelji = 'c_'+observables[j]+'_'+observables[i]
                        if (labelij in data_stellar_params.columns):
                            self.c_obs_params[:, i, j] = data_stellar_params[labelij].to_numpy()
                            self.c_obs_params[:, j, i] = data_stellar_params[labelij].to_numpy()
                        elif (labelji in data_stellar_params.columns):
                            self.c_obs_params[:, i, j] = data_stellar_params[labelji].to_numpy()
                            self.c_obs_params[:, j, i] = data_stellar_params[labelji].to_numpy()
                        else:
                            raise ValueError('{} not found in the stellar parameter table.'.format(labelij))

                for i in range(self.Nstar):
                    self.cinv_obs_params[i, :, :] = np.linalg.inv(self.c_obs_params[i, :, :])


        # read in stellar frequencies
        if self.if_seismic:
            self.Dnu = data_stellar_params[self.col_obs_Dnu].to_numpy()
            self.numax = data_stellar_params[self.col_obs_numax].to_numpy()

            if self.if_add_model_error & (self.add_model_error_method in [2, 3]): 
                self.mod_e_freq = data_stellar_params[self.col_model_error].to_numpy()
                
            data_stellar_freqs = pd.read_csv(self.filepath_stellar_freqs)
            data_stellar_freqs[self.col_starIDs] = data_stellar_freqs[self.col_starIDs].astype('str')

            self.obs_freq, self.obs_e_freq, self.obs_l = [np.empty(self.Nstar, dtype=object) for i in range(3)]
            for istar in range(self.Nstar):
                idx = data_stellar_freqs[self.col_starIDs] == self.starIDs[istar]
                self.obs_freq[istar] = data_stellar_freqs[idx][self.col_obs_freq].to_numpy()
                self.obs_e_freq[istar] = data_stellar_freqs[idx][self.col_obs_e_freq].to_numpy()
                self.obs_l[istar] = data_stellar_freqs[idx][self.col_obs_l].to_numpy()
            
            self.obs_l_uniq = np.array([np.unique(l) for l in self.obs_l], dtype=object)
            self.obs_N_l_uniq = np.array([len(np.unique(l)) for l in self.obs_l])

        return


    def scan_tracks(self, thread_track_idx=None):
        '''

        This is going to circulate all input tracks.
        Return a list of stardata object.
        
        '''

        if (thread_track_idx is None): thread_track_idx = np.arange(0, self.Ntrack, dtype=int)
        tracks = self.tracks[thread_track_idx]
        Ntrack = len(tracks)

        list_of_stardata = [stardata(Ntrack, self.keys[istar]) for istar in range(self.Nstar)]


        for itrack in range(Ntrack): 

            # read in itrack
            atrack = self.read_models(tracks[itrack])
            if (atrack is None) : continue
            Nmodel = len(atrack)

            # calculate posterior
            for istar in range(self.Nstar):
                # dt = (atrack[self.col_age_name][2:] - atrack[self.col_age_name][0:-2])/2.0
                # lifespan = atrack[self.col_age_name][-1] - atrack[self.col_age_name][0]
                # prior = dt/np.sum(lifespan)
                # prior = np.concatenate([prior[:1], prior, prior[-1:]])
                # lnprior = np.log(prior)
                # print(lnprior)

                if self.if_classical:
                    # classical observables
                    mod_params = np.array([atrack[col] for col in self.observables]).T.reshape(Nmodel,-1)
                    if self.if_classical_independent:
                        chi2_classical = np.sum((self.obs_params[istar]-mod_params)**2.0/(self.e_obs_params[istar]**2.0), axis=1)#/(Nobservable)
                    else:
                        # chi2_classical = np.sum(np.sum((self.obs_params[istar][None, None, :] - mod_params[:, :, None])**2.0/(self.c_obs_params[istar]), axis=2), axis=1)#/(Nobservable)
                        # chi2_classical = self.obs_params[istar][None,:] @ self.cinv_obs_params[istar, :, :] @ self.mod_params[:,:,None]
                        diff = mod_params[:,:] - self.obs_params[istar][None,:]
                        chi2_classical = np.einsum('ij,jk,ik->i', diff, self.cinv_obs_params[istar, :, :], diff)
                    idx_classical = chi2_classical < 16. # 4-sigma
                else:
                    idx_classical = True # all


                if self.if_seismic:
                    # seismic observables
                    obs_freq = self.obs_freq[istar]
                    obs_e_freq = self.obs_e_freq[istar]
                    obs_l = self.obs_l[istar]
                    obs_l_unique = self.obs_l_uniq[istar]
                    obs_N_l_uniq = self.obs_N_l_uniq[istar]
                    Nmode = len(obs_freq)

                    # initialize the variables to calculate & save
                    Dnu_freq = np.zeros(Nmodel, dtype=float) + np.nan
                    eps = np.zeros(Nmodel, dtype=float) + np.nan
                    diff_freq = np.zeros((Nmodel, Nmode), dtype=float) + np.nan
                    mod_freq = np.zeros((Nmodel, Nmode), dtype=float) + np.nan
                    if self.if_correct_surface:
                        surface_parameters = np.zeros((Nmodel,self.Nsurface), dtype=float) + np.nan
                        Dnu_freq_sc = np.zeros(Nmodel, dtype=float) + np.nan
                        eps_sc = np.zeros(Nmodel, dtype=float) + np.nan
                        diff_freq_sc = np.zeros((Nmodel, Nmode), dtype=float) + np.nan
                        mod_freq_sc = np.zeros((Nmodel, Nmode), dtype=float) + np.nan

                    
                    for imod in np.arange(0, Nmodel)[idx_classical]:
                        # get 1) Dnu from frquencies, 2) squared differences
                        # retrieve seismic model parameters
                        mode_freq = np.array(atrack[self.col_mode_freq][imod])
                        mode_l = np.array(atrack[self.col_mode_l][imod])
                        mode_n = np.array(atrack[self.col_mode_n][imod]) if ~(self.col_mode_n is None) else np.arange(len(mode_l))

                        if len(mode_freq) < len(obs_freq) : continue

                        obs_freq_matched, _, _, mode_freq_matched, mode_l_matched, mode_n_matched = match_modes(obs_freq, obs_e_freq, obs_l, mode_freq, mode_l, mode_n)

                        if len(mode_freq_matched) < len(obs_freq): continue

                        Dnu_freq[imod], eps[imod] = get_model_Dnu(mode_freq_matched, mode_l_matched, self.Dnu[istar], self.numax[istar], mode_n_matched)
                        diff_freq[imod, :] = (obs_freq_matched-mode_freq_matched)**2.0
                        mod_freq[imod, :] = mode_freq_matched

                        # get 1) Dnu, 2) squared differences, 
                        # but for the surface correction version, if there is any
                        if (self.if_correct_surface) & \
                            (np.abs((Dnu_freq[imod]-self.Dnu[istar])/self.Dnu[istar])<0.15 ) & \
                            (np.sum(np.isin(mode_l, 0))) :

                            # retrieve seismic model parameters
                            mode_inertia = np.array(atrack[self.col_mode_inertia][imod])
                            acoustic_cutoff = atrack[self.col_acoustic_cutoff][imod]

                            mode_freq_sc, surface_parameters[imod,:] = get_surface_correction(obs_freq, obs_l, \
                                                                            mode_freq, mode_l, \
                                                                            mode_inertia, acoustic_cutoff, \
                                                                            formula=self.surface_correction_formula, \
                                                                            ifFullOutput=True, \
                                                                            Dnu=self.Dnu[istar], numax=self.numax[istar])

                            obs_freq_matched, _, _, mode_freq_sc_matched, mode_l_sc_matched, mode_n_sc_matched = match_modes(obs_freq, obs_e_freq, obs_l, mode_freq_sc, mode_l, mode_n)
                            Dnu_freq_sc[imod], eps_sc[imod] = get_model_Dnu(mode_freq_sc_matched, mode_l_sc_matched, self.Dnu[istar], self.numax[istar], mode_n_sc_matched)
                            diff_freq_sc[imod, :] = (obs_freq_matched-mode_freq_sc_matched)**2.0
                            mod_freq_sc[imod, :] = mode_freq_sc_matched

                    idx_seismic = (np.abs((Dnu_freq-self.Dnu[istar])/self.Dnu[istar])<0.15 )
                    if self.if_correct_surface: idx_seismic = idx_seismic & np.isfinite(Dnu_freq_sc)
                    if self.if_correct_surface & self.require_negative_surface_correction: 
                        idx_seismic = idx_seismic & (np.sum(mod_freq_sc - mod_freq > 0, axis=1) == 0 )
                    if self.if_correct_surface & self.require_absolute_surface_correction_increase_with_nu & (0 in obs_l_unique):
                        idx_upper_nu = np.argmax(obs_freq[obs_l==0])
                        idx_lower_nu = np.argmin(obs_freq[obs_l==0])
                        diff_freq_l0 = mod_freq_sc[:, obs_l==0] - mod_freq[:, obs_l==0]
                        idx_seismic = idx_seismic & (np.abs(diff_freq_l0[:, idx_upper_nu]) - np.abs(diff_freq_l0[:, idx_lower_nu]) > 0)
                else:
                    idx_seismic = True


                # only save models with a large likelihood - otherwise not useful and quickly fill up memory
                idx = idx_classical & idx_seismic

                # save estimates
                for col in self.estimators:
                    list_of_stardata[istar][col, itrack] = np.array(atrack[col][idx], dtype=float)

                if self.if_classical:
                    list_of_stardata[istar]['chi2_classical', itrack] = np.array(chi2_classical[idx], dtype=float)

                if self.if_seismic:
                    list_of_stardata[istar]['diff_freq', itrack] = np.array(diff_freq[idx], dtype=float)
                    list_of_stardata[istar]['mod_freq', itrack] = np.array(mod_freq[idx], dtype=float)
                    list_of_stardata[istar]['Dnu_freq', itrack] = np.array(Dnu_freq[idx], dtype=float)
                    list_of_stardata[istar]['eps', itrack] = np.array(eps[idx], dtype=float)

                    if self.if_correct_surface:
                        list_of_stardata[istar]['diff_freq_sc', itrack] = np.array(diff_freq_sc[idx], dtype=float)
                        list_of_stardata[istar]['mod_freq_sc', itrack] = np.array(mod_freq_sc[idx], dtype=float)
                        list_of_stardata[istar]['Dnu_freq_sc', itrack] = np.array(Dnu_freq_sc[idx], dtype=float)
                        list_of_stardata[istar]['eps_sc', itrack] = np.array(eps_sc[idx], dtype=float)
                        for icol, col in enumerate(self.surface_estimators):
                            list_of_stardata[istar][col, itrack] = np.array(surface_parameters[idx, icol], dtype=float)

                # list_of_stardata[istar].append('lnprior', lnprior[fidx], dtype=float)

        return list_of_stardata

    
    def process_stardata(self, thread_star_idx=None):
        '''

        Calculate chi2s.
        
        '''
        if (thread_star_idx is None): thread_star_idx = np.arange(0, self.Nstar, dtype=int)
        list_of_stardata = [[] for istar in thread_star_idx]

        for ilist, istar in enumerate(thread_star_idx):
            self.stardata[istar].collapse(copy=False)

            if self.if_seismic: 
                # self.stardata[istar]['Teff', :] ... 
                obs_N_l_uniq = self.obs_N_l_uniq[istar]
                obs_l_uniq = self.obs_l_uniq[istar]
                obs_freq = self.obs_freq[istar]
                obs_e_freq = self.obs_e_freq[istar]
                obs_l = self.obs_l[istar]
                obs_l_idx = [obs_l==l for l in obs_l_uniq]

                diff_freq = self.stardata[istar]['diff_freq_sc'] if self.if_correct_surface else self.stardata[istar]['diff_freq']

                # determine model systematic uncertainty, mod_e_freq
                for il in range(obs_N_l_uniq):
                    l, idx = obs_l_uniq[il], obs_l_idx[il]
                    # obs_freq[idx], obs_e_freq[idx], diff_freq[idx]
                    Nnreg = np.sum(idx)-self.Nreg if self.if_reduce_seis_chi2 else 1
                    Nreg = self.Nreg if self.if_reduce_seis_reg_chi2 else 1
                    Nmode = np.sum(idx) if self.if_reduce_seis_chi2 else 1
                    if self.if_add_model_error:
                        if self.if_regularize:
                            for il, l in enumerate(obs_l_uniq):
                                if self.add_model_error_method == 1:
                                    mod_e_freq_nreg =  np.percentile(np.mean(diff_freq[:,obs_l_idx[il][self.Nreg:]], axis=1), self.rescale_percentile)
                                    mod_e_freq_reg =  np.percentile(np.mean(diff_freq[:,obs_l_idx[il][:self.Nreg]], axis=1), self.rescale_percentile)
                                elif self.add_model_error_method == 2:
                                    mod_e_freq_nreg, mod_e_freq_reg = self.mod_e_freq[istar], self.mod_e_freq[istar]
                                else: # self.add_model_error_method == 3:
                                    r = self.mod_e_freq[istar]/np.min(obs_e_freq[obs_l_idx[il][self.Nreg:]])
                                    r = r if r>1 else 1
                                    mod_e_freq_nreg = (r**2. - 1)**0.5 * obs_e_freq[obs_l_idx[il][self.Nreg:]]
                                    r = self.mod_e_freq[istar]/np.min(obs_e_freq[obs_l_idx[il][:self.Nreg]])
                                    r = r if r>1 else 1
                                    mod_e_freq_reg = (r**2. - 1)**0.5 * obs_e_freq[obs_l_idx[il][:self.Nreg]]
                                self.stardata[istar]['chi2_seismic_obs_mod_nreg_l{:0.0f}'.format(l)] = np.sum(diff_freq[:,obs_l_idx[il][self.Nreg:]]/(obs_e_freq[obs_l_idx[il][self.Nreg:]]**2.0 + mod_e_freq_nreg**2.0), axis=1) / Nnreg
                                self.stardata[istar]['chi2_seismic_obs_mod_reg_l{:0.0f}'.format(l)] = np.sum(diff_freq[:,obs_l_idx[il][:self.Nreg]]/(obs_e_freq[obs_l_idx[il][:self.Nreg]]**2.0 + mod_e_freq_reg**2.0), axis=1) / Nreg
                                self.stardata[istar]['chi2_seismic_obs_mod_l{:0.0f}'.format(l)] = np.sum(diff_freq[:,obs_l_idx[il]]/(obs_e_freq[obs_l_idx[il]]**2.0 + mod_e_freq**2.0), axis=1) / Nmode
                                self.stardata[istar]['chi2_seismic_obs_l{:0.0f}'.format(l)] = np.sum(diff_freq[:,obs_l_idx[il]]/obs_e_freq[obs_l_idx[il]]**2.0, axis=1) / Nmode
                                self.stardata[istar]['chi2_seismic_l{:0.0f}'.format(l)] = (1-self.weight_reg) * self.stardata[istar]['chi2_seismic_obs_mod_nreg_l{:0.0f}'.format(l)] + self.weight_reg * self.stardata[istar]['chi2_seismic_obs_mod_reg_l{:0.0f}'.format(l)]
                        else:
                            for il, l in enumerate(obs_l_uniq):
                                if self.add_model_error_method == 1:
                                    mod_e_freq =  np.percentile(np.mean(diff_freq[:,obs_l_idx[il]], axis=1), self.rescale_percentile)
                                elif self.add_model_error_method == 2:
                                    mod_e_freq =  self.mod_e_freq[istar]
                                else: # self.add_model_error_method == 3:
                                    r = self.mod_e_freq[istar]/np.min(obs_e_freq[obs_l_idx[il]])
                                    r = r if r>1 else 1
                                    mod_e_freq = (r**2. - 1)**0.5 * obs_e_freq[obs_l_idx[il]]
                                self.stardata[istar]['chi2_seismic_obs_mod_l{:0.0f}'.format(l)] = np.sum(diff_freq[:,obs_l_idx[il]]/(obs_e_freq[obs_l_idx[il]]**2.0 + mod_e_freq**2.0), axis=1) / Nmode
                                self.stardata[istar]['chi2_seismic_obs_l{:0.0f}'.format(l)] = np.sum(diff_freq[:,obs_l_idx[il]]/obs_e_freq[obs_l_idx[il]]**2.0, axis=1) / Nmode
                                self.stardata[istar]['chi2_seismic_l{:0.0f}'.format(l)] = self.stardata[istar]['chi2_seismic_obs_mod_l{:0.0f}'.format(l)]
                    else:
                        if self.if_regularize:
                            for il, l in enumerate(obs_l_uniq):
                                self.stardata[istar]['chi2_seismic_obs_nreg_l{:0.0f}'.format(l)] = np.sum(diff_freq[:,obs_l_idx[il][self.Nreg:]]/(obs_e_freq[obs_l_idx[il][self.Nreg:]]**2.0), axis=1) / Nnreg
                                self.stardata[istar]['chi2_seismic_obs_reg_l{:0.0f}'.format(l)] = np.sum(diff_freq[:,obs_l_idx[il][:self.Nreg]]/(obs_e_freq[obs_l_idx[il][:self.Nreg]]**2.0), axis=1) / Nreg
                                self.stardata[istar]['chi2_seismic_obs_l{:0.0f}'.format(l)] = np.sum(diff_freq[:,obs_l_idx[il]]/obs_e_freq[obs_l_idx[il]]**2.0, axis=1) / Nmode
                                self.stardata[istar]['chi2_seismic_l{:0.0f}'.format(l)] = (1-self.weight_reg) * self.stardata[istar]['chi2_seismic_obs_nreg_l{:0.0f}'.format(l)] + self.weight_reg * self.stardata[istar]['chi2_seismic_obs_reg_l{:0.0f}'.format(l)]
                        else:                            
                            for il, l in enumerate(obs_l_uniq):
                                self.stardata[istar]['chi2_seismic_obs_l{:0.0f}'.format(l)] = np.sum(diff_freq[:,obs_l_idx[il]]/obs_e_freq[obs_l_idx[il]]**2.0, axis=1) / Nmode
                                self.stardata[istar]['chi2_seismic_l{:0.0f}'.format(l)] = self.stardata[istar]['chi2_seismic_obs_l{:0.0f}'.format(l)]
                    # self.stardata[istar]['chi2_seismic_l{:0.0f}'.format(l), :] = ...
                self.stardata[istar]['chi2_seismic'] = np.sum([self.stardata[istar]['chi2_seismic_l{:0.0f}'.format(l)] for l in obs_l_uniq], axis=0)
            


            if (self.if_classical) & (~self.if_seismic):
                self.stardata[istar]['chi2'] = np.array(self.weight_classical * self.stardata[istar]['chi2_classical'], dtype=float)
            elif (~self.if_classical) & (self.if_seismic):
                self.stardata[istar]['chi2'] = np.array(self.weight_seismic * self.stardata[istar]['chi2_seismic'], dtype=float)
            else:
                self.stardata[istar]['chi2'] = np.array(self.weight_classical * self.stardata[istar]['chi2_classical'] + self.weight_seismic * self.stardata[istar]['chi2_seismic'], dtype=float)
                
            # print(self.stardata[istar]['chi2_classical'].shape)
            # print(self.stardata[istar]['chi2_seismic'].shape)
            # print(self.weight_classical, self.weight_seismic)
            # print(self.stardata[istar]['chi2'].shape)

            list_of_stardata[ilist] = self.stardata[istar]
        return list_of_stardata


    def output_results(self, thread_star_idx=None):
        
        if (thread_star_idx is None): thread_star_idx = np.arange(0, self.Nstar, dtype=int)

        for istar in thread_star_idx:
            # self.stardata[istar]

            # starIDs = self.starname[thread_star_idx]
            # if (self.if_seismic):
            #     obs_freq = self.obs_freq[thread_star_idx]
            #     obs_efreq = self.obs_efreq[thread_star_idx]
            #     obs_l = self.obs_l[thread_star_idx]
            #     Dnu = self.Dnu[thread_star_idx]
            #     obs_l_uniq = self.obs_l_uniq[thread_star_idx]
            # Nstar = len(starIDs)

            toutdir = self.filepath_output + '{:s}'.format(self.starIDs[istar]) + '/'
            if not os.path.exists(toutdir):
                os.mkdir(toutdir)
            
            # if type(list_of_stardata[istar]) is int: continue
            
            # collect samples to produce estimates/corner plots
            samples = []
            for para in self.estimators_to_summary:
                samples.append(self.stardata[istar][para])
            samples = np.transpose(np.array(samples))

            samples_to_plot = []
            for para in self.estimators_to_plot:
                samples_to_plot.append(self.stardata[istar][para])
            samples_to_plot = np.transpose(np.array(samples_to_plot))

            
            # locate best models
            # print(self.stardata[istar]['chi2'], self.stardata[istar]['chi2_seismic'])
            chi2 = self.stardata[istar]['chi2']
            if len(chi2) <= 5:
                f = open(toutdir+'log.txt', 'w')
                f.write("Parameter estimation failed because fewer than 5 models have been selected.")
                f.close()
                continue
            prob = np.exp(-chi2/2.)
            best_models_ranked_by = ['chi2']
            best_models_imod = [np.nanargmax(prob)]
            best_models_chi2s = [chi2[np.nanargmax(prob)]]

            if self.if_classical:
                chi2_classical = np.array(self.stardata[istar]['chi2_classical'], dtype=float)
                prob_classical = np.exp(-(chi2_classical)/2.)
                best_models_ranked_by.append('chi2_classical')
                best_models_imod.append(np.nanargmax(prob_classical))
                best_models_chi2s.append(chi2_classical[np.nanargmax(prob_classical)])

            if self.if_seismic:
                chi2_seismic = np.array(self.stardata[istar]['chi2_seismic'], dtype=float)
                prob_seismic = np.exp(-(chi2_seismic)/2.)
                best_models_ranked_by.append('chi2_seismic')
                best_models_imod.append(np.nanargmax(prob_seismic))
                best_models_chi2s.append(chi2_seismic[np.nanargmax(prob_seismic)])
                
            if self.if_plot:
                if samples.shape[0] <= samples.shape[1]:
                    f = open(toutdir+'log.txt', 'w')
                    f.write("Parameter estimation failed because samples.shape[0] <= samples.shape[1].")
                    f.close()
                else:
                    # plot prob distributions
                    if self.if_classical:
                        fig = plot_parameter_distributions(samples_to_plot, self.estimators_to_plot, prob_classical)
                        fig.savefig(toutdir+"corner_prob_classical.png")
                        plt.close()

                    if self.if_seismic:
                        fig = plot_parameter_distributions(samples_to_plot, self.estimators_to_plot, prob_seismic)
                        fig.savefig(toutdir+"corner_prob_seismic.png")
                        plt.close()

                    # output the prob (prior included)
                    fig = plot_parameter_distributions(samples_to_plot, self.estimators_to_plot, prob)
                    fig.savefig(toutdir+"corner_prob.png")
                    plt.close()

                    # write prob distribution summary file
                    if self.if_classical:
                        results = quantile(samples, (0.16, 0.5, 0.84), weights=prob_classical)
                        ascii.write(Table(results, names=self.estimators_to_summary), toutdir+"summary_prob_classical.txt",format="csv", overwrite=True)

                    if self.if_seismic:
                        results = quantile(samples, (0.16, 0.5, 0.84), weights=prob_seismic)
                        ascii.write(Table(results, names=self.estimators_to_summary), toutdir+"summary_prob_seismic.txt",format="csv", overwrite=True)

                    # output the prob (prior excluded)
                    results = quantile(samples, (0.16, 0.5, 0.84), weights=prob)
                    ascii.write(Table(results, names=self.estimators_to_summary), toutdir+"summary_prob.txt",format="csv", overwrite=True)

                    # output the best models
                    Nchi2 = len(best_models_chi2s)
                    results = np.concatenate([np.array(best_models_ranked_by).reshape(Nchi2,1), \
                                              np.array(best_models_chi2s).reshape(Nchi2,1), \
                                              samples[best_models_imod,:] ], axis=1)
                    ascii.write(Table(results, names=np.concatenate([['best_model_by', 'chi2'], self.estimators_to_summary])), toutdir+"summary_best.txt",format="csv", overwrite=True)

                    # # plot HR diagrams
                    # fig = self.plot_HR_diagrams(samples, self.estimates, zvals=logweights)
                    # if not (fig is None): 
                    #     fig.savefig(toutdir+"HR.png")
                    #     plt.close()

                    # plot echelle diagrams
                    if self.if_seismic & self.if_plot_echelle:
                        idx = np.argsort(chi2_seismic, axis=0)[:10]
                        mod_freq_sc = self.stardata[istar]['mod_freq_sc'][idx] if self.if_correct_surface else None
                        fig = plot_seis_echelles(self.obs_freq[istar], self.obs_e_freq[istar], self.obs_l[istar], 
                                self.stardata[istar]['mod_freq'][idx], chi2_seismic[idx], self.Dnu[istar], mod_freq_sc=mod_freq_sc)
                        fig.savefig(toutdir+"echelle_top10_prob_seismic.png")
                        plt.close()

            # write related parameters to file
            if self.if_data:
                with h5py.File(toutdir+'data.h5', 'w') as h5f:
                    for key in self.keys[istar]:
                        h5f.create_dataset(key, data=self.stardata[istar][key])

        return 


    def run(self):
        """
        
        Estimate parameters. Magic function!

        """

        # # declaration
        self.keys = np.empty(self.Nstar, dtype=object)
        for istar in range(self.Nstar):
            keys = self.estimators + ['chi2']
            if self.if_classical: keys = keys + ['chi2_classical']
            if self.if_seismic: 
                keys = keys + ['chi2_seismic']
                ls = self.obs_l_uniq[istar]
                keys = keys + ['Dnu_freq', 'eps', 'diff_freq', 'mod_freq']
                keys = keys + ['chi2_seismic_l{:0.0f}'.format(l) for l in ls]
                keys = keys + ['chi2_seismic_obs_l{:0.0f}'.format(l) for l in ls]
                if self.if_add_model_error:
                    keys = keys + ['chi2_seismic_obs_mod_l{:0.0f}'.format(l) for l in ls]
                    if self.if_regularize:
                        keys = keys + ['chi2_seismic_obs_mod_reg_l{:0.0f}'.format(l) for l in ls]
                        keys = keys + ['chi2_seismic_obs_mod_nreg_l{:0.0f}'.format(l) for l in ls]
                else:
                    if self.if_regularize:
                        keys = keys + ['chi2_seismic_obs_reg_l{:0.0f}'.format(l) for l in ls]
                        keys = keys + ['chi2_seismic_obs_nreg_l{:0.0f}'.format(l) for l in ls]
                if self.if_correct_surface:
                    keys = keys + ['diff_freq_sc', 'eps_sc', 'mod_freq_sc', 'Dnu_freq_sc'] + self.surface_estimators
            self.keys[istar] = keys


        # # step 1, scan all tracks and collect stardata
        if self.Nthread == 1:
            self.stardata = np.array(self.scan_tracks(), dtype=object)
        else:
            # multithreading
            Ntrack_per_thread = int(self.Ntrack/self.Nthread)+1
            arglist = [np.arange(0, self.Ntrack, dtype=int)[ithread*Ntrack_per_thread:(ithread+1)*Ntrack_per_thread] for ithread in range(self.Nthread)]

            pool = multiprocessing.Pool(processes=self.Nthread)
            result_list = pool.map(self.scan_tracks, arglist)
            pool.close()
            
            # merge from different threads
            self.stardata = np.empty(self.Nstar, dtype=object)
            for istar in range(self.Nstar):
                self.stardata[istar] = concat([result_list[ithread][istar] for ithread in range(self.Nthread)])

        
        

        # # step 2, process stardata
        if self.Nthread == 1:
            self.process_stardata()
        else:
            # multithreading
            Nstar_per_thread = int(self.Nstar/self.Nthread)+1
            arglist = [np.arange(0, self.Nstar, dtype=int)[ithread*Nstar_per_thread:(ithread+1)*Nstar_per_thread] for ithread in range(self.Nthread)]

            pool = multiprocessing.Pool(processes=self.Nthread)
            result_list = pool.map(self.process_stardata, arglist)
            pool.close()

            # merge from different threads
            self.stardata = np.array([item for threads in result_list for item in threads], dtype=object)

        # # step 3: output results
        if self.Nthread==1:
            self.output_results()
        else:
            Nstar_per_thread = int(self.Nstar/self.Nthread)+1
            arglist = [np.arange(0, self.Nstar, dtype=int)[ithread*Nstar_per_thread:(ithread+1)*Nstar_per_thread] for ithread in range(self.Nthread)]

            pool = multiprocessing.Pool(processes=self.Nthread)
            _ = pool.map(self.output_results, arglist)
            pool.close()

            # # merge from different threads
            # self.stardata = np.array([item for threads in result_list for item in threads], dtype=object)

        return

