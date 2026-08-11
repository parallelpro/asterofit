__all__ = ['user_setup_params']

user_setup_params = {

### - 1 - pass data ###

# variables to be compared with observations
"observables": ['Teff', 'FeH', 'luminosity'] ,

# variables to be estimated
"estimators": ['star_age', 'star_mass', 'luminosity', 'radius', 'Teff', \
               'Zinit', 'numax_scaling', 'acoustic_cutoff'] ,

# csv file which contains stellar parameters set in the observables. 
# the column names should be exact as those set in the observables.
"filepath_stellar_params": 'test_samples/samples.csv', 
# will be the name of output sub-folders
"col_starIDs": 'KIC',
# must be set if if_seismic is True
"col_obs_Dnu": "Dnu", 
# must be set if if_seismic is True
"col_obs_numax": "numax", 


# a csv file which contains contain starIDs, l-degrees, frequencies and uncertainites. 
"filepath_stellar_freqs": 'test_samples/modes.csv', 
"col_obs_freq": "fc",
"col_obs_e_freq": "e_fc",
"col_obs_l": "l",


### - 2 - output settings ###

# if True, output corner plots. 
"if_plot": True, 
# if True, output the echelle diagram for top 10 models measured by chi2_seismic. 
"if_plot_echelle": True,
# estimator names to appear in corner plots. suggests only keep essentials.
"estimators_to_plot": ['star_mass', 'luminosity', 'radius', 'Teff', \
                       'Zinit', 'eps', 'Dnu_freq'], 
# if True, output top models to an h5 file
"if_data": True, 


# int, the number of threads to process in parallel
"Nthread": 6, 

# filepath to output results 
"filepath_output": 'test_output/',



### - 3 - classical settings ###

# grid modelling with classical variables (single-value variables, Teff, lum, numax, feh, etc.)
"if_classical": True, 
# weight factor multiplied with the classical chi2
"weight_classical": 1, 
# not implemented yet.
# star age column name, to add a prior. Consider deprecating this and allowing customizing prior/likelihood.
"col_age_name": "star_age", 

# if treat each observables as independent for constructing χ2.
# if True, then yes, and the code will search for std unceratinty in columns "e_observables", e.g. "e_Teff", "e_FeH", etc.
# if False, then no, and the code will search for covariance in columns "c_observablesi_observablesj", e.g. "c_Teff_Teff", "c_Teff_FeH", "c_FeH_luminosity", etc.
"if_classical_independent": True, 



### - 4 - seismic settings ###

# grid modelling with seismic individual frequencies 
"if_seismic": True, 
# weight factor multiplied with the seismic chi2
"weight_seismic": 1, 
# mode frequency column name in models
"col_mode_freq": "mode_freq",  
# mode l-degree column name in models
"col_mode_l": "mode_l", 
# mode inertia column name in models
"col_mode_inertia": "mode_inertia", 
# mode acoustic cutoff column name in models
"col_acoustic_cutoff": "acoustic_cutoff", 
# mode n (nodes) column name in models
"col_mode_n": "mode_n", 

# not implemented yet. 
# if True, use reduced chi2: sum((obs_i-mod_i)**2.0/sig**2.0/Nmode); 
# if False: use true chi2: sum((obs_i-mod_i)**2.0/sig**2.0)
"if_reduce_seis_chi2": True, 


# if True, correct model frequencies with supported formula, see next.
"if_correct_surface": True, 
# cubic/combined/kjeldsen, bg14/bg14/k+08
"surface_correction_formula": "cubic", 
# if True, discard models with the positive amount of surface correction.
"require_negative_surface_correction": True,
# if True, discard models with absolute amount of surface correction smaller at high frequencies.
# for radial modes only.
"require_absolute_surface_correction_increase_with_nu": True,

# if True, then add systematic uncertainties in the seismic chi2 
"if_add_model_error": True,
# 1, 2, or 3: 
# 1 - choose the rms frequency difference of the model at 10 percentile (sorted by chi2).
# ``rescale_percentile'' must be set.
# 2 - set by the column in the filepath_stellar_params file 
# the error set by the column will be added in quadrature
# ``col_model_error'' must be set.
# 3 - set by the column in the filepath_stellar_params file 
# the errors of all modes will be multiplied by a rescaling factor such that the smallest observational error
# equals to this value provided in the file.
# ``col_model_error'' must be set.
"add_model_error_method": 3, 
# used to evaluate the systematic uncertainties. the model rms frequency difference at below percentile 
"rescale_percentile": 10, 
# used to set the systematic uncertainties from the filepath_stellar_params file
"col_model_error": "mod_e_freq",


# not properly implemented yet.
# if True, then separate Nreg number of modes that do not apply surface corrections
"if_regularize": False, 
# weight factor multiplied with the regularized seismic chi2
"weight_reg": 1, 
# number of modes that do not apply surface corrections
"Nreg": 0,
# if True, use reduced chi2 for low-order modes: sum((obs_i-mod_i)**2.0/sig**2.0/Nmode);
"if_reduce_seis_reg_chi2": True, 

}

