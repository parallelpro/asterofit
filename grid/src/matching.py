#!/usr/bin/env/ python
# coding: utf-8

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ["match_modes"]

def match_modes(obs_freq, obs_efreq, obs_l, mod_freq, mod_l, *modargs):
    # assume all parameters are numpy arrays

    # assign n_p or n_g based on the closeness of the frequencies
    new_obs_freq, new_obs_efreq, new_obs_l, new_mod_freq, new_mod_l = [np.array([]) for i in range(5)]
    # new_mod_freq = np.zeros(len(obs_freq)) 
    new_mod_args = [np.array([]) for i in range(len(modargs))]

    for l in np.sort(np.unique(obs_l)):
        obs_freq_l = obs_freq[obs_l==l]
        obs_efreq_l = obs_efreq[obs_l==l]
        mod_freq_l = mod_freq[mod_l==l]

        mod_args = [[] for iarg in range(len(modargs))]
        for iarg in range(len(modargs)):
            mod_args[iarg] = modargs[iarg][mod_l==l]

        # because we don't know n_p or n_g from observation (if we do that will save tons of effort here)
        # we need to assign each obs mode with a model mode
        # this can be seen as a linear sum assignment problem, also known as minimun weight matching in bipartite graphs
        cost = np.abs(obs_freq_l.reshape(-1,1) - mod_freq_l)
        row_ind, col_ind = linear_sum_assignment(cost)
        # obs_freq_l = obs_freq_l[row_ind]
        # obs_efreq_l = obs_efreq_l[row_ind]

        # mod_freq_l = mod_freq_l[col_ind]
        # mod_inertia_l = mod_inertia_l[col_ind]

        new_obs_freq = np.append(new_obs_freq, obs_freq_l[row_ind])
        new_obs_efreq = np.append(new_obs_efreq, obs_efreq_l[row_ind])
        new_obs_l = np.append(new_obs_l, obs_l[obs_l==l][row_ind])

        new_mod_freq = np.append(new_mod_freq, mod_freq_l[col_ind])
        new_mod_l = np.append(new_mod_l, mod_l[mod_l==l][col_ind])

        for iarg in range(len(modargs)):
            new_mod_args[iarg] = np.append(new_mod_args[iarg], mod_args[iarg][col_ind])

    return (new_obs_freq, new_obs_efreq, new_obs_l, new_mod_freq, new_mod_l, *new_mod_args)
