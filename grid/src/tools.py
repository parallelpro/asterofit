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

def echelle(x, y, period, fmin=None, fmax=None, echelletype="single", offset=0.0):
    '''
    Generate a z-map for echelle plotting.

    Input:

    x: array-like[N,]
    y: array-like[N,]
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

    if len(x) != len(y): 
        raise ValueError("x and y must have equal size.")    

    if fmin is None: fmin=0.
    if fmax is None: fmax=np.nanmax(x)

    fmin = fmin - offset
    fmax = fmax - offset
    x = x - offset

    if fmin <= 0.0:
        fmin = 0.0
    else:
        fmin = fmin - (fmin % period)

    # first interpolate
    samplinginterval = np.median(x[1:-1] - x[0:-2]) * 0.1
    xp = np.arange(fmin,fmax+period,samplinginterval)
    yp = np.interp(xp, x, y)

    n_stack = int((fmax-fmin)/period)
    n_element = int(period/samplinginterval)
    #print(n_stack,n_element,len())

    morerow = 2
    arr = np.arange(1,n_stack) * period # + period/2.0
    arr2 = np.array([arr,arr])
    yn = np.reshape(arr2,len(arr)*2,order="F")
    yn = np.insert(yn,0,0.0)
    yn = np.append(yn,n_stack*period) + fmin #+ offset

    if echelletype == "single":
        xn = np.arange(1,n_element+1)/n_element * period
        z = np.zeros([n_stack*morerow,n_element])
        for i in range(n_stack):
            for j in range(i*morerow,(i+1)*morerow):
                z[j,:] = yp[n_element*(i):n_element*(i+1)]
    if echelletype == "replicated":
        xn = np.arange(1,2*n_element+1)/n_element * period
        z = np.zeros([n_stack*morerow,2*n_element])
        for i in range(n_stack):
            for j in range(i*morerow,(i+1)*morerow):
                z[j,:] = np.concatenate([yp[n_element*(i):n_element*(i+1)],yp[n_element*(i+1):n_element*(i+2)]])

    return xn, yn, z



def return_2dmap_axes(numberOfSquareBlocks):

    # Some magic numbers for pretty axis layout.
    # stole from corner
    Kx = int(np.ceil(numberOfSquareBlocks**0.5))
    Ky = Kx if (Kx**2-numberOfSquareBlocks) < Kx else Kx-1

    factor = 2.0           # size of one side of one panel
    lbdim = 0.4 * factor   # size of left/bottom margin, default=0.2
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.30         # w/hspace size
    plotdimx = factor * Kx + factor * (Kx - 1.) * whspace
    plotdimy = factor * Ky + factor * (Ky - 1.) * whspace
    dimx = lbdim + plotdimx + trdim
    dimy = lbdim + plotdimy + trdim

    # Create a new figure if one wasn't provided.
    fig, axes = plt.subplots(Ky, Kx, figsize=(dimx, dimy), squeeze=False)

    # Format the figure.
    l = lbdim / dimx
    b = lbdim / dimy
    t = (lbdim + plotdimy) / dimy
    r = (lbdim + plotdimx) / dimx
    fig.subplots_adjust(left=l, bottom=b, right=r, top=t,
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
        idx = np.argsort(x, axis=0)

        res = []
        for i in range(x.shape[1]):
            sw = weights[idx[:,i]]
            cdf = np.cumsum(sw)[:-1]
            cdf /= cdf[-1]
            cdf = np.append(0, cdf)
            res.append(np.interp(q, cdf, x[idx[:,i],i]))
        return np.array(res).T

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
    idx = np.isfinite(weights)
    if np.sum(~idx)>0:
        weight = weight[idx]
        x = x[idx,]

    nsamples, nfeatures = x.shape
    scount = np.sum(np.isfinite(x), axis=0)
    smean = np.nanmean(x, axis=0)
    smin = np.nanmin(x, axis=0)
    smax = np.nanmax(x, axis=0)
    squantile = quantile(x, (16, 50, 84), weights=weights)


    stats = pd.DataFrame([scount, smean, smin, smax, squantile, ], 
        index=['count','mean','min','max','16%','50%','84%',])

    return stats


def plot_parameter_distributions(samples, estimates, probs):
    # corner plot, overplotted with observation constraints
    
    Ndim = samples.shape[1]
    idx = np.any(np.isfinite(samples), axis=0)
    estimates = np.array(estimates)
    samples, estimates = samples[:,idx], estimates[idx]
    ranges = [[np.nanmin(samples[:,idim]), np.nanmax(samples[:,idim])] for idim in range(Ndim)]
    for ir, r in enumerate(ranges):
        if np.sum(np.isfinite(r))==2:
            if r[0]==r[1]: ranges[ir] = [r[0]-1, r[1]+1]
        else:
            ranges[ir] = [0, 1]

    fig = corner.corner(samples, range=ranges, labels=estimates, quantiles=(0.16, 0.5, 0.84), weights=probs)
    return fig

def plot_HR_diagrams(samples, estimates, zvals=None,
        Teff=['Teff', 'log_Teff'], lum=['luminosity', 'log_L', 'lum'], 
        delta_nu=['delta_nu', 'delta_nu_scaling', 'delta_nu_freq', 'dnu'], 
        nu_max=['nu_max', 'numax'], log_g=['log_g', 'logg']):
    
    if np.sum(np.array([iTeff in estimates for iTeff in Teff], dtype=bool))==0:
        return None 
    
    xstr = Teff[np.where(np.array([iTeff in estimates for iTeff in Teff], dtype=bool))[0][0]]
    
    numberofPlots = 0
    ystrs = []
    for val in lum+delta_nu+nu_max+log_g:
        if (val in estimates):
            numberofPlots += (val in estimates)
            ystrs.append(val)
    
    if len(ystrs) ==0:
        return None
    
    fig, axes = return_2dmap_axes(numberofPlots)

    for iax, ystr in enumerate(ystrs):
        x = samples[:,np.where(np.array(estimates) == xstr)[0][0]]
        y = samples[:,np.where(np.array(estimates) == ystr)[0][0]]
        im = axes[iax].scatter(x, y, marker='.', c=zvals, cmap='jet', s=1)
        axes[iax].set_xlabel(xstr)
        axes[iax].set_ylabel(ystr)
        axes[iax].set_xlim(quantile(x, (0.998, 0.002)).tolist())
        if ystr in delta_nu+nu_max+log_g:
            axes[iax].set_ylim(quantile(y, (0.998, 0.002)).tolist())
        else:
            axes[iax].set_ylim(quantile(y, (0.002, 0.998)).tolist())

    # fig..colorbar(im, ax=axes, orientation='vertical').set_label('Log(likelihood)')     
    # plt.tight_layout()

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, orientation='vertical').set_label('Log(likelihood)')    

    return fig 

def plot_seis_echelles(obs_freq, obs_e_freq, obs_l, mod_freq, model_chi2, Dnu, 
                        mod_freq_sc=None):
    
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
    for imod in np.argsort(model_chi2)[::-1]:
        for l in np.array(np.unique(obs_l), dtype=int):
            # axes[0] plot uncorrected frequencies
            z = np.zeros(np.sum(obs_l==l))+model_chi2[imod]
            scatterstyles = {'marker':markers[l], 'edgecolors':cmap(norm(z)), 'c':'None', 'zorder':2}
            axes[0,0].scatter(mod_freq[imod,:][obs_l==l] % Dnu, mod_freq[imod,:][obs_l==l], **scatterstyles)
            axes[0,0].scatter(mod_freq[imod,:][obs_l==l] % Dnu + Dnu, mod_freq[imod,:][obs_l==l], **scatterstyles)
            if if_correct_surface:
                # axes[1] plot surface corrected frequencies
                axes[0,1].scatter(mod_freq_sc[imod,:][obs_l==l] % Dnu, mod_freq_sc[imod,:][obs_l==l], **scatterstyles)
                axes[0,1].scatter(mod_freq_sc[imod,:][obs_l==l] % Dnu + Dnu, mod_freq_sc[imod,:][obs_l==l], **scatterstyles)

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