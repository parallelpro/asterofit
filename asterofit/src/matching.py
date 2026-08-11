#!/usr/bin/env/ python
# coding: utf-8

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ["match_modes"]

def match_modes(obs_freq, obs_efreq, obs_l, mod_freq, mod_l, *modargs):
    """
    Pair each observed mode with the model mode of the same angular degree
    that best matches its frequency, since observations don't come labeled
    with a radial order n the way model modes do.

    Matching is done independently per angular degree l (only modes of the
    same l can be paired), and within each l as a linear sum assignment
    problem (`scipy.optimize.linear_sum_assignment`): the cost of pairing
    observed mode i with model mode j is `|obs_freq[i] - mod_freq[j]|`, and
    the assignment chosen is the one that minimizes the total cost summed
    over all pairs -- i.e. a global (not greedy) closest-frequency matching.
    If a degree has more model modes than observed modes (typical), the extra
    model modes are simply left unmatched.

    Any number of extra per-model arrays (e.g. mode_n, mode_inertia) can be
    passed positionally via `*modargs` and will be reordered/subset alongside
    `mod_freq`/`mod_l` using the same matched indices, so they stay aligned
    with the returned, matched frequencies.

    ----------
    Input:
    obs_freq, obs_efreq, obs_l: array-like[Nmode_obs]
        Observed mode frequency, frequency uncertainty, and angular degree.
    mod_freq, mod_l: array-like[Nmode_mod]
        Model mode frequency and angular degree. Nmode_mod is typically >=
        Nmode_obs (models usually have more computed modes than are observed).
    *modargs: array-like[Nmode_mod], any number
        Additional per-model arrays to carry through the same matching/subset
        as `mod_freq`/`mod_l` (e.g. mode_n, mode_inertia).

    ----------
    Output:
    (new_obs_freq, new_obs_efreq, new_obs_l, new_mod_freq, new_mod_l, *new_modargs)
        All array-like[Nmode_matched], where Nmode_matched <= Nmode_obs (equal
        unless some degree has more observed than model modes, in which case
        the unmatched observed modes for that degree are dropped). `new_obs_*`
        are `obs_*` reordered/subset to align 1:1 with `new_mod_*`, which are
        the best-matching model modes; `new_modargs` are `*modargs` subset the
        same way as `new_mod_freq`/`new_mod_l`.

    """
    # assign n_p or n_g based on the closeness of the frequencies
    new_obs_freq, new_obs_efreq, new_obs_l, new_mod_freq, new_mod_l = [[] for _ in range(5)]
    new_mod_args = [[] for _ in range(len(modargs))]

    for l in np.sort(np.unique(obs_l)):
        obs_l_mask = obs_l==l
        mod_l_mask = mod_l==l

        obs_freq_l = obs_freq[obs_l_mask]
        obs_efreq_l = obs_efreq[obs_l_mask]
        mod_freq_l = mod_freq[mod_l_mask]

        mod_args = [modargs[arg_idx][mod_l_mask] for arg_idx in range(len(modargs))]

        # because we don't know n_p or n_g from observation (if we do that will save tons of effort here)
        # we need to assign each obs mode with a model mode
        # this can be seen as a linear sum assignment problem, also known as minimun weight matching in bipartite graphs
        cost = np.abs(obs_freq_l.reshape(-1,1) - mod_freq_l)
        matched_obs_idx, matched_mod_idx = linear_sum_assignment(cost)

        new_obs_freq.append(obs_freq_l[matched_obs_idx])
        new_obs_efreq.append(obs_efreq_l[matched_obs_idx])
        new_obs_l.append(obs_l[obs_l_mask][matched_obs_idx])

        new_mod_freq.append(mod_freq_l[matched_mod_idx])
        new_mod_l.append(mod_l[mod_l_mask][matched_mod_idx])

        for arg_idx in range(len(modargs)):
            new_mod_args[arg_idx].append(mod_args[arg_idx][matched_mod_idx])

    new_obs_freq = np.concatenate(new_obs_freq) if new_obs_freq else np.array([])
    new_obs_efreq = np.concatenate(new_obs_efreq) if new_obs_efreq else np.array([])
    new_obs_l = np.concatenate(new_obs_l) if new_obs_l else np.array([])
    new_mod_freq = np.concatenate(new_mod_freq) if new_mod_freq else np.array([])
    new_mod_l = np.concatenate(new_mod_l) if new_mod_l else np.array([])
    new_mod_args = [np.concatenate(arg_values) if arg_values else np.array([]) for arg_values in new_mod_args]

    return (new_obs_freq, new_obs_efreq, new_obs_l, new_mod_freq, new_mod_l, *new_mod_args)
