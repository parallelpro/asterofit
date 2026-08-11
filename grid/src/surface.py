import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ['get_surface_correction', 'surface_params_dict']

surface_params_dict = {'cubic': ['surf_a3', 'surf_corr_at_numax'],
                       'combined': ['surf_a1', 'surf_a3', 'surf_corr_at_numax', 'surf_corr_at_1p1_numax'],
                       'kjeldsen': ['surf_a', 'surf_b', 'surf_corr_at_numax', 'surf_corr_at_1p1_numax']}


def get_surface_correction(obs_freq, obs_l, mod_freq, mod_l, mod_inertia, mod_acfreq, 
                            formula='cubic', ifFullOutput=False, Dnu=None, numax=None, ):
    
    # formula is one of 'cubic', 'BG14'
    if not (formula in ['cubic', 'combined', 'kjeldsen']):
        raise ValueError('formula must be one of ``cubic``, ``combined`` and ``kjeldsen''. ')

    new_mod_freq = np.array(mod_freq) 

    if (np.sum(np.isin(mod_l, 0))) :
        # if correction is needed, first we use l=0 modes to derive correction factors
        # if obs_l don't have a 0, well I am not expecting this! 
        obs_freq_l0 = obs_freq[obs_l==0]
        new_mod_freq_l0 = new_mod_freq[mod_l==0]
        mod_inertia_l0 = mod_inertia[mod_l==0]

        # because we don't know n_p or n_g from observation (if we do that will save tons of effort here)
        # we need to assign each obs mode with a model mode
        # this can be seen as a linear sum assignment problem, also known as minimun weight matching in bipartite graphs
        cost = np.abs(obs_freq_l0.reshape(-1,1) - new_mod_freq_l0)
        row_ind, col_ind = linear_sum_assignment(cost)
        obs_freq_l0 = obs_freq_l0[row_ind]
        new_mod_freq_l0 = new_mod_freq_l0[col_ind]
        mod_inertia_l0 = mod_inertia_l0[col_ind]

        # regression
        b = obs_freq_l0-new_mod_freq_l0
        # # avoid selecting reversed models
        # if (np.abs(np.median(np.diff(np.sort(obs_freq_l0)))) > np.abs(np.median(np.diff(np.sort(mod_freq_l0)))) ) :
        #     return None

        if formula == 'combined':
            A1 = (new_mod_freq_l0/mod_acfreq)**-1. / mod_inertia_l0
            A2 = (new_mod_freq_l0/mod_acfreq)**3. / mod_inertia_l0
            AT = np.array([A1, A2])
            A = AT.T
            b = b.reshape(-1,1)

            # apply corrections
            try:
                coeff = np.dot(np.dot(np.linalg.inv(np.dot(AT,A)), AT), b)
                coeff = coeff.reshape(-1)
                delta_freq = (coeff[0]*(new_mod_freq/mod_acfreq)**-1.  + coeff[1]*(new_mod_freq/mod_acfreq)**3. ) / mod_inertia
                new_mod_freq += delta_freq

                if ((numax != None) & (Dnu != None)):
                    surf_corr_at_numax = np.interp(numax, new_mod_freq, delta_freq)
                    surf_corr_at_1p1_numax = np.interp(1.1*numax, new_mod_freq, delta_freq)
                    surface_params = np.concatenate([coeff, [surf_corr_at_numax, surf_corr_at_1p1_numax]])
                else:
                    surface_params = np.concatenate([coeff, [np.nan, np.nan]])

            except:
                coeff = np.zeros(4)
                print('An exception occurred when correcting surface effect using combined form.')
                # pass

        if formula == 'cubic':
            A2 = (new_mod_freq_l0/mod_acfreq)**3. / mod_inertia_l0
            AT = np.array([A2])
            A = AT.T
            b = b.reshape(-1,1)

            # apply corrections
            try:
                coeff = np.dot(np.dot(np.linalg.inv(np.dot(AT,A)), AT), b)
                coeff = coeff.reshape(-1)
                delta_freq = ( coeff[0]*(new_mod_freq/mod_acfreq)**3. ) / mod_inertia
                new_mod_freq += delta_freq

                if ((numax != None) & (Dnu != None)):
                    surf_corr_at_numax = np.interp(numax, new_mod_freq, delta_freq)
                    surface_params = np.concatenate([coeff, [surf_corr_at_numax]])
                else:
                    surface_params = np.concatenate([coeff, [np.nan]])

            except:
                coeff = np.zeros(1)
                print('An exception occurred when correcting surface effect using cubic form.')
                # pass

        if formula == 'kjeldsen':
            if np.sum(b<0)>3:
                idx = b<0
                A1 = np.ones(len(new_mod_freq_l0[idx]))
                A2 = np.log(new_mod_freq_l0[idx]/mod_inertia_l0[idx])
                AT = np.array([A1, A2])
                A = np.swapaxes(AT, 0, 1)
                b = np.log(b[idx]).reshape(-1,1)

                # apply corrections
                try:
                    coeff = np.dot(np.dot(np.linalg.inv(np.dot(AT,A)), AT), b)
                    coeff = coeff.reshape(-1)
                    coeff[0] = np.exp(coeff[0])
                    delta_freq = ( coeff[0]*(new_mod_freq/mod_inertia)**coeff[1] )
                    new_mod_freq += delta_freq

                    if ((numax != None) & (Dnu != None)):
                        surf_corr_at_numax = np.interp(numax, new_mod_freq, delta_freq)
                        surf_corr_at_1p1_numax = np.interp(1.1*numax, new_mod_freq, delta_freq)
                        surface_params = np.concatenate([coeff, [surf_corr_at_numax, surf_corr_at_1p1_numax]])
                    else:
                        surface_params = np.concatenate([coeff, [np.nan, np.nan]])

                    
                except:
                    coeff = np.zeros(2)
                    print('An exception occurred when correcting surface effect using kjeldsen form.')
                    # pass
            else:
                coeff = np.zeros(2)
                print('Using kjeldsen form, not enough (at least 3) modes with negative difference.')
                # pass
    if ifFullOutput:
        return new_mod_freq, surface_params
    else:
        return new_mod_freq