'''
Stellar modelling with free values for surface correction. 
'''
import os
import numpy as np
import pandas as pd
import sys

from asterofit import grid, get_model_Dnu, user_setup_params
from astropy.table import Table


def read_models(filepath):
    atrack = Table(np.load(filepath, allow_pickle=True))
    # atrack['luminosity'] = 10.0**atrack['log_L']
    return atrack

if __name__ == '__main__':


    work_dir = './'
    # sys.path.append(rootpath)

    ### step 0: input parameters from the user
    ipart = 0 #690 stars, process 60 at a time
    sample = 'rg' # ms or rg or cl # ind
    Nstar = 10 #64
    offset = 0

    random_index = int(np.random.rand()*1e8)

    # the place to store results
    outdir = 'demo_results/'
    if not os.path.exists(outdir): os.makedirs(outdir, exist_ok=True)


    ### step 1: prepare samples
    if sample == 'ms':
        stars = pd.read_excel(work_dir+'test_samples/samples.xlsx')
    elif sample == 'rg':
        stars = pd.read_csv(work_dir+'test_samples/samples_rg.csv')
    else:
        stars = pd.read_excel(work_dir+'test_samples/samples_cl.xlsx')
    kics = np.loadtxt('exists.txt', dtype='int')
    idx = (np.isfinite(stars['Dnu'])) & (np.isfinite(stars['luminosity'])) & \
        (stars['[M/H]'] > -0.7) & (stars['[M/H]'] < 0.) & (stars['numax'] < 230.) & (~np.isin(stars['KIC'], kics)) 
    stars = stars.loc[idx,:].sort_values('[M/H]').reset_index(drop=True)
    idx = (stars.index>=(ipart*Nstar+offset)) & (stars.index<((ipart+1)*Nstar+offset))
    stars = stars.loc[idx,:].reset_index(drop=True)

    mh_llim, mh_ulim = np.min(stars['[M/H]'])-5*0.05, np.max(stars['[M/H]'])+5*0.05

    a, b = [0.1775, 0.4026]
    stars['mod_e_freq'] = a * (stars['numax']/3090)**b 

    cols = ['KIC', 'Teff', 'e_Teff', '[M/H]', 'e_[M/H]', 'luminosity', 'e_luminosity', 'Dnu', 'e_Dnu', 'numax', 'e_numax', 'mod_e_freq']
    stars.loc[:,cols].to_csv(work_dir+'inputs/samples_{:0.0f}.csv'.format(random_index), index=False)

    if sample == 'ms':
        modes = pd.read_excel(work_dir+'test_samples/modes.xlsx')
        idx = (np.isin(modes['KIC'], stars['KIC'])) & (modes['l']<=2) # & 
    else:
        modes = pd.read_csv(work_dir+'test_samples/modes_rg.csv')
        idx = (np.isin(modes['KIC'], stars['KIC'])) & (np.isin(modes['l'], [0,2])) # & 
    modes = modes.loc[idx,:].reset_index(drop=True)

    cols = ['KIC', 'l', 'fc', 'e_fc']
    modes.loc[:,cols].to_csv(work_dir+'inputs/modes_{:0.0f}.csv'.format(random_index), index=False)


    ### step 3: prepare a list of evolutionary tracks
    lists = []
    tracks = ['test_models/calibrated/'+f for f in os.listdir('test_models/calibrated/') if (f.endswith('.npy') )]
    # tracks = np.array(tracks)
    # tracks_indexes = np.array([int(track[-17:-11]) for track in tracks])

    # grid_params = pd.read_csv(rootpath+'hpc/coarse_v5/template/coarse_grid_input_params_v5.txt', sep=', ', engine='python')[:8192*2]

    # Zsun, Xsun = 0.0134, 0.7381 # 0.0134, 0.7381, a09 # 0.0169, 0.7345, gs98
    # grid_params['[M/H]'] = np.log10((grid_params['Zinit']/grid_params['Xinit'])) - np.log10(Zsun/Xsun)
    # idx = (grid_params['[M/H]']>=mh_llim) & (grid_params['[M/H]']<=mh_ulim) 
    # print('Number of tracks: {:.0f}'.format(np.sum(idx)))
    # idx = np.isin(tracks_indexes, np.array(grid_params.loc[idx, 'index']))
    # tracks = tracks[idx]

    ### step 4: setup params

    new_params = {
    "observables": ['luminosity', 'Teff', '[M/H]'], 
    "estimators": ['index', 'star_age', 'star_mass', 'radius', 'density',\
                'Teff','luminosity','FeH', 'numax_scaling',\
                'fov_core', 'fov_shell', 'amlt','Yinit','Zinit',\
                'profile_number'],

    "filepath_stellar_params": work_dir+'inputs/samples_{:0.0f}.csv'.format(random_index), 
    "col_starIDs": 'KIC',
    "filepath_stellar_freqs": work_dir+'inputs/modes_{:0.0f}.csv'.format(random_index), 

    # estimator names to appear in corner plots. suggests only keep essentials.
    "estimators_to_plot": ['star_age', 'star_mass', 'radius', 'density'],
    # if True, output top models to an h5 file
    "if_data": True, 
    "Nthread": 8, 

    "if_plot": True, 
    "if_plot_echelle": True,

    # filepath to output results 
    "filepath_output": outdir,

    "if_classical": True,
    "if_seismic": True,
    "if_reduce_seis_chi2": True, 
    "if_correct_surface": False, 
    # # added 
    # "require_negative_surface_correction": False,
    # "require_absolute_surface_correction_increase_with_nu": False,
    # # added
    # "surface_correction_formula": 'combined', 
    # "if_add_model_error": True,
    # "col_mode_n": "mode_n", 
    # 2 - set by the column in the filepath_stellar_params file
    # ``col_model_error'' must be set.
    "add_model_error_method": 2, 
    # used to evaluate the systematic uncertainties. the model rms frequency difference at below percentile 
    # "rescale_percentile": 10, 
    # # used to set the systematic uncertainties from the filepath_stellar_params file
    "col_model_error": "mod_e_freq",
    }


    for key in new_params.keys():
        user_setup_params[key] = new_params[key]



    # initialize a grid modelling class
    g = grid(read_models, tracks, user_setup_params)

    g.run()
