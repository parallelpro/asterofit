#!/usr/bin/env/ python
# coding: utf-8


import numpy as np
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors
import corner

__all__ = ["echelle", "return_2dmap_axes", "quantile", \
           "plot_parameter_distributions", "plot_HR_diagrams", "plot_seis_echelles"]

def echelle(freq, power, period, fmin=None, fmax=None, echelletype="single", offset=0.0):
    '''
    Generate a z-map for echelle plotting.

    Input:

    freq: array-like[N,]
    power: array-like[N,]
    period: the large separation,
    fmin: the lower boundary
    fmax: the upper boundary
    echelletype: single/replicated
    offset: the horizontal shift

    Output:

    x, y:
        two 1-d arrays.
    z:
        a 2-d array.

    Exemplary call:

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6,8))
    ax1 = fig.add_subplot(111)
    echx, echy, echz = echelle(tfreq,tpowers_o,dnu,numax-9.0*dnu,numax+9.0*dnu,echelletype="single",offset=offset)
    levels = np.linspace(np.min(echz),np.max(echz),500)
    ax1.contourf(echx,echy,echz,cmap="gray_r",levels=levels)
    ax1.axis([np.min(echx),np.max(echx),np.min(echy),np.max(echy)])
    if offset > 0.0:
        ax1.set_xlabel("(Frequency - "+str("{0:.2f}").format(offset)+ ") mod "+str("{0:.2f}").format(dnu) + " ($\mu$Hz)")
    if offset < 0.0:
        ax1.set_xlabel("(Frequency + "+str("{0:.2f}").format(np.abs(offset))+ ") mod "+str("{0:.2f}").format(dnu) + " ($\mu$Hz)")
    if offset == 0.0:
        ax1.set_xlabel("Frequency mod "+str("{0:.2f}").format(dnu) + " ($\mu$Hz)")
    plt.savefig("echelle.png")

    '''

    if not echelletype in ["single", "replicated"]:
        raise ValueError("echelletype is on of 'single', 'replicated'.")

    if len(freq) != len(power):
        raise ValueError("freq and power must have equal size.")

    if fmin is None: fmin=0.
    if fmax is None: fmax=np.nanmax(freq)

    fmin = fmin - offset
    fmax = fmax - offset
    freq = freq - offset

    if fmin <= 0.0:
        fmin = 0.0
    else:
        fmin = fmin - (fmin % period)

    # first interpolate
    samplinginterval = np.median(freq[1:-1] - freq[0:-2]) * 0.1
    interp_freq = np.arange(fmin,fmax+period,samplinginterval)
    interp_power = np.interp(interp_freq, freq, power)

    n_stack = int((fmax-fmin)/period)
    n_element = int(period/samplinginterval)
    #print(n_stack,n_element,len())

    rows_per_stack = 2
    stack_offsets = np.arange(1,n_stack) * period # + period/2.0
    stack_offsets_2col = np.array([stack_offsets,stack_offsets])
    row_bounds = np.reshape(stack_offsets_2col,len(stack_offsets)*2,order="F")
    row_bounds = np.insert(row_bounds,0,0.0)
    row_bounds = np.append(row_bounds,n_stack*period) + fmin #+ offset

    if echelletype == "single":
        col_positions = np.arange(1,n_element+1)/n_element * period
        z_map = np.zeros([n_stack*rows_per_stack,n_element])
        for stack_idx in range(n_stack):
            for row_idx in range(stack_idx*rows_per_stack,(stack_idx+1)*rows_per_stack):
                z_map[row_idx,:] = interp_power[n_element*(stack_idx):n_element*(stack_idx+1)]
    if echelletype == "replicated":
        col_positions = np.arange(1,2*n_element+1)/n_element * period
        z_map = np.zeros([n_stack*rows_per_stack,2*n_element])
        for stack_idx in range(n_stack):
            for row_idx in range(stack_idx*rows_per_stack,(stack_idx+1)*rows_per_stack):
                z_map[row_idx,:] = np.concatenate([interp_power[n_element*(stack_idx):n_element*(stack_idx+1)],interp_power[n_element*(stack_idx+1):n_element*(stack_idx+2)]])

    return col_positions, row_bounds, z_map



def return_2dmap_axes(n_square_blocks):
    """
    Create a new figure with a roughly-square grid of subplots, sized and
    spaced the way `corner` lays out its panels (used here to keep ad hoc
    multi-panel plots, like `plot_HR_diagrams`, visually consistent with the
    corner plots from `plot_parameter_distributions`).

    ----------
    Input:
    n_square_blocks: int
        How many subplots are needed. The grid is `n_cols x n_rows` with
        `n_cols = ceil(sqrt(n_square_blocks))` and `n_rows` chosen to be as
        small as possible while still fitting all of them -- so `n_cols*n_rows`
        can exceed `n_square_blocks` by a few, leaving that many trailing
        panels in the last row unused (the caller decides what to do with
        those, if anything).

    ----------
    Output:
    fig: matplotlib.figure.Figure
    axes: array-like[n_cols*n_rows], flattened
        One Axes per grid cell, in row-major order (`axes.reshape(-1)`
        already applied) -- index directly rather than through `axes[i,j]`.

    """

    # Some magic numbers for pretty axis layout.
    # stole from corner
    n_cols = int(np.ceil(n_square_blocks**0.5))
    n_rows = n_cols if (n_cols**2-n_square_blocks) < n_cols else n_cols-1

    factor = 2.0           # size of one side of one panel
    lbdim = 0.4 * factor   # size of left/bottom margin, default=0.2
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.30         # w/hspace size
    plotdimx = factor * n_cols + factor * (n_cols - 1.) * whspace
    plotdimy = factor * n_rows + factor * (n_rows - 1.) * whspace
    dimx = lbdim + plotdimx + trdim
    dimy = lbdim + plotdimy + trdim

    # Create a new figure if one wasn't provided.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(dimx, dimy), squeeze=False)

    # Format the figure.
    left_frac = lbdim / dimx
    bottom_frac = lbdim / dimy
    top_frac = (lbdim + plotdimy) / dimy
    right_frac = (lbdim + plotdimx) / dimx
    fig.subplots_adjust(left=left_frac, bottom=bottom_frac, right=right_frac, top=top_frac,
                        wspace=whspace, hspace=whspace)
    axes = np.concatenate(axes)

    return fig, axes


def quantile(x, q, weights=None):

    """
    Compute sample quantiles with support for weighted samples.
    Modified based on 'corner'.

    ----------
    Input:
    x: array-like[nsamples, nfeatures]
        The samples.
    q: array-like[nquantiles,]
        The list of quantiles to compute. These should all be in the range
        '[0, 1]'.

    ----------
    Optional input:
    weights : array-like[nsamples,]
        Weight to each sample.

    ----------
    Output:
    quantiles: array-like[nquantiles,]
        The sample quantiles computed at 'q'.
    
    """

    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if weights is None:
        return np.percentile(x, 100.0 * q)
    else:
        weights = np.atleast_1d(weights)
        sort_idx = np.argsort(x, axis=0)

        quantiles_per_feature = []
        for feature_idx in range(x.shape[1]):
            sorted_weights = weights[sort_idx[:,feature_idx]]
            cdf = np.cumsum(sorted_weights)[:-1]
            cdf /= cdf[-1]
            cdf = np.append(0, cdf)
            quantiles_per_feature.append(np.interp(q, cdf, x[sort_idx[:,feature_idx],feature_idx]))
        return np.array(quantiles_per_feature).T

def statistics(x, weights=None, colnames=None):
    """
    Compute important sample statistics with supported for weighted samples. 
    The statistics include count, mean, std, min, (16%, 50%, 84%) quantiles, max,
    median (50% quantile), err ((84%-16%)quantiles/2.), 
    maxcprob (maximize the conditional distribution for each parameter), 
    maxjprob (maximize the joint distribution),
    and best (the sample with the largest weight, random if `weights` is None).
    Both `maxcprob` and `maxjprob` utilises a guassian_kde estimator. 
    
    ----------
    Input:
    x: array-like[nsamples, nfeatures]
        The samples.

    ----------
    Optional input:
    weights : array-like[nsamples,]
        Weight to each sample.

    colnames: array-like[nfeatures,]
        Column names for x (the second axis).

    ----------
    Output:
    stats: pandas DataFrame object [5,]
        statistics

    """

    x = np.atleast_1d(x)
    if (weights is None):
        weight = np.ones(x.shape[0])
    finite_mask = np.isfinite(weights)
    if np.sum(~finite_mask)>0:
        weight = weight[finite_mask]
        x = x[finite_mask,]

    nsamples, nfeatures = x.shape
    sample_count = np.sum(np.isfinite(x), axis=0)
    sample_mean = np.nanmean(x, axis=0)
    sample_min = np.nanmin(x, axis=0)
    sample_max = np.nanmax(x, axis=0)
    sample_quantile = quantile(x, (16, 50, 84), weights=weights)


    stats = pd.DataFrame([sample_count, sample_mean, sample_min, sample_max, sample_quantile, ],
        index=['count','mean','min','max','16%','50%','84%',])

    return stats


def plot_parameter_distributions(samples, estimates, probs):
    """
    Draw a `corner` plot (1D histogram per parameter on the diagonal,
    weighted 2D density for every pair off it) of the surviving models'
    parameter values, weighted by their probability -- this is what
    `output_results()` calls per star for the classical/seismic/combined
    corner plots.

    Columns that are entirely non-finite are dropped first (e.g. seismic
    columns when a star only has classical constraints), and any column whose
    finite values are all identical gets a +/-1 padded range instead of a
    zero-width one, since `corner` can't plot a zero-width axis.

    ----------
    Input:
    samples: array-like[nsamples, nfeatures]
        One row per surviving model, one column per parameter in `estimates`.
    estimates: array-like[nfeatures]
        Parameter names, used as axis labels (and to size the corner grid).
    probs: array-like[nsamples]
        Per-model weight (e.g. `exp(-chi2/2)`) used for the weighted
        histograms/quantile lines `corner` draws.

    ----------
    Output:
    fig: matplotlib.figure.Figure

    """
    # corner plot, overplotted with observation constraints

    Ndim = samples.shape[1]
    finite_col_mask = np.any(np.isfinite(samples), axis=0)
    estimates = np.array(estimates)
    samples, estimates = samples[:,finite_col_mask], estimates[finite_col_mask]
    ranges = [[np.nanmin(samples[:,dim_idx]), np.nanmax(samples[:,dim_idx])] for dim_idx in range(Ndim)]
    for range_idx, dim_range in enumerate(ranges):
        if np.sum(np.isfinite(dim_range))==2:
            if dim_range[0]==dim_range[1]: ranges[range_idx] = [dim_range[0]-1, dim_range[1]+1]
        else:
            ranges[range_idx] = [0, 1]

    fig = corner.corner(samples, range=ranges, labels=estimates, quantiles=(0.16, 0.5, 0.84), weights=probs)
    return fig

def plot_HR_diagrams(samples, estimates, zvals=None,
        Teff=['Teff', 'log_Teff'], lum=['luminosity', 'log_L', 'lum'],
        delta_nu=['delta_nu', 'delta_nu_scaling', 'delta_nu_freq', 'dnu'],
        nu_max=['nu_max', 'numax'], log_g=['log_g', 'logg']):
    """
    Scatter-plot a Teff-vs-Y diagram (HR diagram, or a seismic analog like
    Teff-vs-Dnu) for every Y quantity present in `estimates`, one subplot
    each, colored by `zvals` (typically a likelihood/probability). Not
    currently called anywhere in the pipeline -- `output_results()` has this
    commented out -- but kept available for ad hoc/manual use.

    Teff/lum/delta_nu/nu_max/log_g are each a list of acceptable column-name
    aliases for that physical quantity (e.g. either 'Teff' or 'log_Teff'),
    since different grids may name their columns differently; the first alias
    found in `estimates` is used. Axis limits are the 99.6% weighted-quantile
    range of the data (via `quantile`), flipped for the y-axis on
    delta_nu/nu_max/log_g panels (so e.g. higher log_g is at the bottom, as
    is conventional for HR-diagram-like plots).

    ----------
    Input:
    samples: array-like[nsamples, nfeatures]
        One row per surviving model, one column per parameter in `estimates`.
    estimates: array-like[nfeatures]
        Parameter names corresponding to `samples`' columns.

    ----------
    Optional input:
    zvals: array-like[nsamples] or None
        Per-model color value (e.g. log-likelihood); passed straight to
        `scatter`'s `c=`.
    Teff, lum, delta_nu, nu_max, log_g: list[str]
        Acceptable column-name aliases for each physical quantity.

    ----------
    Output:
    fig: matplotlib.figure.Figure, or None
        None if no Teff-like column is found in `estimates`, or if none of
        lum/delta_nu/nu_max/log_g are either (nothing to plot against Teff).

    """
    if np.sum(np.array([teff_alias in estimates for teff_alias in Teff], dtype=bool))==0:
        return None

    x_label = Teff[np.where(np.array([teff_alias in estimates for teff_alias in Teff], dtype=bool))[0][0]]

    n_plots = 0
    y_labels = []
    for candidate_name in lum+delta_nu+nu_max+log_g:
        if (candidate_name in estimates):
            n_plots += (candidate_name in estimates)
            y_labels.append(candidate_name)

    if len(y_labels) ==0:
        return None

    fig, axes = return_2dmap_axes(n_plots)

    for ax_idx, y_label in enumerate(y_labels):
        x_values = samples[:,np.where(np.array(estimates) == x_label)[0][0]]
        y_values = samples[:,np.where(np.array(estimates) == y_label)[0][0]]
        im = axes[ax_idx].scatter(x_values, y_values, marker='.', c=zvals, cmap='jet', s=1)
        axes[ax_idx].set_xlabel(x_label)
        axes[ax_idx].set_ylabel(y_label)
        axes[ax_idx].set_xlim(quantile(x_values, (0.998, 0.002)).tolist())
        if y_label in delta_nu+nu_max+log_g:
            axes[ax_idx].set_ylim(quantile(y_values, (0.998, 0.002)).tolist())
        else:
            axes[ax_idx].set_ylim(quantile(y_values, (0.002, 0.998)).tolist())

    # fig..colorbar(im, ax=axes, orientation='vertical').set_label('Log(likelihood)')
    # plt.tight_layout()

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, orientation='vertical').set_label('Log(likelihood)')

    return fig

def plot_seis_echelles(obs_freq, obs_e_freq, obs_l, mod_freq, model_chi2, Dnu,
                        mod_freq_sc=None):
    """
    Draw an echelle diagram (frequency mod Dnu on x, frequency on y --
    replicated once at `+Dnu` so ridges aren't cut off at the panel edge)
    overplotting several models' mode frequencies on the observed ones, with
    models shaded by chi2 (darker/more opaque = better fit; drawn back-to-
    front so the best models end up on top). This is what
    `output_results()` calls to plot the top-10 models by seismic chi2.

    If `mod_freq_sc` is given, draws a second panel with the surface-corrected
    frequencies instead of a single panel -- so the same observed points can
    be compared against both the raw and surface-corrected model tracks
    side by side.

    ----------
    Input:
    obs_freq, obs_l: array-like[Nmode_obs]
        Observed mode frequency and angular degree (only l=0..3 are drawn;
        each degree gets a fixed marker shape/color).
    mod_freq: array-like[n_models, Nmode_obs]
        Matched model mode frequencies (same modes/order as `obs_freq`) for
        each of the `n_models` models to overplot.
    model_chi2: array-like[n_models]
        Per-model chi2, used both for the plotting order (worst-to-best, so
        best is drawn last/on top) and the color scale.
    Dnu: float
        The large frequency separation used to fold the echelle diagram.

    ----------
    Optional input:
    obs_e_freq: array-like[Nmode_obs]
        Accepted for interface symmetry with the observed-frequency arrays
        elsewhere in the pipeline, but currently unused (no error bars drawn).
    mod_freq_sc: array-like[n_models, Nmode_obs] or None
        Surface-corrected counterpart to `mod_freq`. If given, adds the
        second ("After correction") panel described above.

    ----------
    Output:
    fig: matplotlib.figure.Figure

    """

    if_correct_surface =  not (mod_freq_sc is None)
    if if_correct_surface:
        fig, axes = plt.subplots(figsize=(12,5), nrows=1, ncols=2, squeeze=False)
    else:
        fig, axes = plt.subplots(figsize=(6,5), nrows=1, ncols=1, squeeze=False)
    # axes = axes.reshape(-1) # 0: uncorrected, 1: corrected

    markers = ['o', '^', 's', 'v']
    colors = ['blue', 'red', 'green', 'orange']     

    # plot observation frequencies
    for l in range(4):
        styles = {'marker':markers[l], 'color':colors[l], 'zorder':1}
        axes[0,0].scatter(obs_freq[obs_l==l] % Dnu, obs_freq[obs_l==l], **styles)
        axes[0,0].scatter(obs_freq[obs_l==l] % Dnu + Dnu, obs_freq[obs_l==l], **styles)
        if if_correct_surface:
            axes[0,1].scatter(obs_freq[obs_l==l] % Dnu, obs_freq[obs_l==l], **styles)
            axes[0,1].scatter(obs_freq[obs_l==l] % Dnu + Dnu, obs_freq[obs_l==l], **styles)

    norm = matplotlib.colors.Normalize(vmin=np.min(model_chi2), vmax=np.max(model_chi2))
    cmap = plt.cm.get_cmap('gray')
    for model_idx in np.argsort(model_chi2)[::-1]:
        for l in np.array(np.unique(obs_l), dtype=int):
            # axes[0] plot uncorrected frequencies
            color_values = np.zeros(np.sum(obs_l==l))+model_chi2[model_idx]
            scatterstyles = {'marker':markers[l], 'edgecolors':cmap(norm(color_values)), 'c':'None', 'zorder':2}
            axes[0,0].scatter(mod_freq[model_idx,:][obs_l==l] % Dnu, mod_freq[model_idx,:][obs_l==l], **scatterstyles)
            axes[0,0].scatter(mod_freq[model_idx,:][obs_l==l] % Dnu + Dnu, mod_freq[model_idx,:][obs_l==l], **scatterstyles)
            if if_correct_surface:
                # axes[1] plot surface corrected frequencies
                axes[0,1].scatter(mod_freq_sc[model_idx,:][obs_l==l] % Dnu, mod_freq_sc[model_idx,:][obs_l==l], **scatterstyles)
                axes[0,1].scatter(mod_freq_sc[model_idx,:][obs_l==l] % Dnu + Dnu, mod_freq_sc[model_idx,:][obs_l==l], **scatterstyles)

        # # label the radial orders n for l=0 modes
        # if (imod == np.argsort(model_chi2)[0]) & np.sum(mod_l_uncor==0):
        #     for idxn, n in enumerate(mod_n[mod_l_uncor==0]):
        #         nstr = '{:0.0f}'.format(n)
        #         # axes[0] plot uncorrected frequencies
        #         textstyles = {'fontsize':12, 'ha':'center', 'va':'center', 'zorder':100, 'color':'purple'}
        #         axes[0,0].text((mod_freq[imod,:][mod_l_uncor==0][idxn]+0.05*Dnu) % Dnu, mod_freq[imod,:][mod_l_uncor==0][idxn]+0.05*Dnu, nstr, **textstyles)
        #         axes[0,0].text((mod_freq[imod,:][mod_l_uncor==0][idxn]+0.05*Dnu) % Dnu + Dnu, mod_freq[imod,:][mod_l_uncor==0][idxn]+0.05*Dnu, nstr, **textstyles)
        #         if if_correct_surface:
        #             # axes[1] plot surface corrected frequencies
        #             axes[0,1].text((mod_freq_cor[mod_l_cor==0][idxn]+0.05*Dnu) % Dnu, mod_freq_cor[mod_l_cor==0][idxn]+0.05*Dnu, nstr, **textstyles)
        #             axes[0,1].text((mod_freq_cor[mod_l_cor==0][idxn]+0.05*Dnu) % Dnu + Dnu, mod_freq_cor[mod_l_cor==0][idxn]+0.05*Dnu, nstr, **textstyles)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap='gray'),cax=cbar_ax).set_label('chi2')

    for ax in axes.reshape(-1):
        ax.axis([0., Dnu*2, np.min(obs_freq)-Dnu*4, np.max(obs_freq)+Dnu*4])
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Frequency mod Dnu {:0.3f}'.format(Dnu))
    axes[0,0].set_title('Before correction')
    if if_correct_surface:
        axes[0,1].set_title('After correction')

    return fig