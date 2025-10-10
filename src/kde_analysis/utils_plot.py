#Jam Sadiq
# A script for making plots for the paper

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import simpson
from matplotlib import rcParams
from matplotlib.colors import LogNorm, Normalize
from matplotlib import cm
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

rcParams["text.usetex"] = True
rcParams["font.serif"] = "Computer Modern"
rcParams["font.family"] = "Serif"
rcParams["xtick.labelsize"] = 18
rcParams["ytick.labelsize"] = 18
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"
rcParams["legend.fontsize"] = 18
rcParams["axes.labelsize"] = 18
rcParams["axes.grid"] = True
rcParams["grid.color"] = 'grey'
rcParams["grid.linewidth"] = 1.
#rcParams["grid.linestyle"] = ':'
rcParams["grid.alpha"] = 0.6


###########
## 1D mass rates
def oned_rate_mass(m1_src_grid, m2_src_grid, ratem1_arr, ratem2_arr, tag='', pathplot='./'):
    fig, ax = plt.subplots(figsize=(8, 5))
    color_m1 = 'royalblue'
    color_m2 = 'darkorange'

    # Plot m1 data
    median_m1 = np.median(ratem1_arr, axis=0)
    p5_m1 = np.percentile(ratem1_arr, 5., axis=0)
    p95_m1 = np.percentile(ratem1_arr, 95., axis=0)
    ax.plot(m1_src_grid, median_m1, color=color_m1, linewidth=2, label=r'$m_1$')
    ax.fill_between(m1_src_grid, p5_m1, p95_m1, color=color_m1, alpha=0.3)

    # Plot m2 data
    median_m2 = np.median(ratem2_arr, axis=0)
    p5_m2 = np.percentile(ratem2_arr, 5., axis=0)
    p95_m2 = np.percentile(ratem2_arr, 95., axis=0)
    ax.plot(m2_src_grid, median_m2, color=color_m2, linewidth=2, label=r'$m_2$')
    ax.fill_between(m2_src_grid, p5_m2, p95_m2, color=color_m2, alpha=0.3)

    ax.set_xlim(3., 110.)
    ax.set_ylim(ymin=3e-5, ymax=5)
    ax.legend()
    ax.grid(True, ls="--")
    ax.set_ylabel(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m\,[\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-1}]$', fontsize=18)
    ax.set_xlabel(r"$m$", fontsize=18)
    plt.semilogy()
    plt.tight_layout()
    plt.savefig(pathplot+"Rate_masses_1d_"+tag+".png")
    plt.loglog()
    plt.savefig(pathplot+"Rate_masses_log_1d_"+tag+".png")
    plt.close()

    #fig, ax = plt.subplots(figsize=(8, 5))
    #ax.plot(m2_src_grid, median_m2 * m2_src_grid, color=color_m2, linewidth=2, label=r'$m_2$')
    #ax.fill_between(m2_src_grid, p5_m2 * m2_src_grid, p95_m2 * m2_src_grid, color=color_m2, alpha=0.3)
    #ax.set_xlim(3., 110.)
    #ax.grid(True, ls="--")
    #ax.set_ylabel(r'$m_2\,\mathrm{d}\mathcal{R}/\mathrm{d}m_2\,[\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}]$', fontsize=18)
    #ax.set_xlabel(r"$m_2$", fontsize=18)
    #plt.loglog()
    #ax.set_ylim(1e-3, 1e2)
    #plt.tight_layout()
    #plt.savefig(pathplot+"Rate_m2_log_1d_"+tag+".png")
    return


def chieff_offset_plot(m_grid, chieff_grid, m_slice_values, rate_m_chieff, log=False, offset_increment=5, m_label='m_1', tag='', pathplot='./'):
    from scipy.integrate import simpson
    colormap = plt.cm.magma
    norm = Normalize(vmin=min(m_slice_values), vmax=max(m_slice_values) + 5)
    offset = 1 if log else 0
    plt.figure(figsize=(8, 8))
    for i, m_val in enumerate(m_slice_values):
        color = colormap(norm(m_slice_values[i]))
        ratechieff_slice_m = []
        idx = np.argmin(np.abs(m_grid - m_val))
        for rate in rate_m_chieff:
            slice_rate = rate[idx, :]
            normalize = simpson(y=slice_rate, x=chieff_grid)
            ratechieff_slice_m.append(slice_rate/normalize)
        median = np.percentile(ratechieff_slice_m, 50., axis=0)
        p05 = np.percentile(ratechieff_slice_m, 5., axis=0)
        p95 = np.percentile(ratechieff_slice_m, 95., axis=0)
        if log:
            plt.semilogy(chieff_grid, median * offset, color=color, linewidth=2, label=r'$'+m_label+'={0}$'.format(m_val))
            plt.fill_between(chieff_grid, p05*offset, p95*offset, color=color, alpha=0.3)
            plt.text(-0.67, offset*3, "$"+m_label+"={0:.1f}$".format(m_val), fontsize=14, color='k', verticalalignment='center')
            offset *= offset_increment
        else:
            plt.plot(chieff_grid, median+offset, color=color, linewidth=2, label=r'$'+m_label+'={0}$'.format(m_val))
            plt.fill_between(chieff_grid, p05+offset, p95+offset, color=color, alpha=0.3)
            plt.axhline(y=offset, color='grey', linestyle='-.', alpha=0.5)
            plt.text(-0.67, offset+0.7, "$"+m_label+"={0:.1f}$".format(m_val), fontsize=14, color='k', verticalalignment='center')
            offset += offset_increment
    plt.xlim(-0.8, 0.8)
    if log:
        plt.ylabel(r"$\log(p(\chi_\mathrm{eff}|"+m_label+")$ + offset", fontsize=20)
    else:
        plt.ylabel(r"$p(\chi_\mathrm{eff}|"+m_label+")$ + offset", fontsize=20)
    plt.xlabel(r"$\chi_\mathrm{eff}$", fontsize=20)
    plt.grid('False')
    plt.yticks([]) #remove y-ticks
    plt.tight_layout()
    logstring = 'log_' if log else ''
    plt.savefig(pathplot+logstring+'p_chieff_at'+m_label+'_slices_'+tag+'.png')
    plt.close()
    return


############# m1m2 contour plot integrated over chieff
def m1m2_contour(medianlist_m1, medianlist_m2, M1, M2, median_est, timesM=False, itertag='', pathplot='./', plot_name='KDE'):
    median_to_plot = median_est * (M1 * M2) if timesM else median_est
    prefix = r'm_1m_2\,' if timesM else ''  # multiply by m for display
    massunits = r'' if timesM else r'M_\odot^{-2}'
    if plot_name == 'Rate':
        colorbar_label = r'$'+prefix+r'd \mathcal{R}/dm_1 dm_2 [\mathrm{Gpc}^{-3} \mathrm{yr}^{-1}'+massunits+']$'
    else:
        colorbar_label = r'$p(m_1, m_2)$'
    max_density = np.max(median_to_plot)
    max_exp = np.floor(2. * np.log10(max_density))  # Find the highest power of 10 below max_density
    contourlevels = 10 ** (0.5 * (max_exp - np.arange(6))[::-1])

    plt.figure(figsize=(8, 6.4))
    norm1 = LogNorm(vmin=contourlevels[0], vmax=max_density)  # Apply log normalization
    pcm = plt.pcolormesh(M1, M2, median_to_plot, cmap='Purples', norm=norm1, shading='auto')
    contours = plt.contour(M1, M2, median_to_plot, levels=contourlevels, colors='black', linewidths=1, norm=LogNorm())
    cbar = plt.colorbar(pcm, label=colorbar_label)

    plt.fill_between(np.arange(3.01, 109.5), np.arange(3.01, 109.5), 109.5, color='white', alpha=1, zorder=50)
    plt.scatter(medianlist_m1, medianlist_m2, color='r', marker='+', s=20)
    plt.xlabel(r"$m_\mathrm{1} \,[M_\odot]$")
    plt.ylabel(r"$m_\mathrm{2} \,[M_\odot]$")
    plt.tick_params(axis='y', which='both', left=False, right=True, labelleft=False, labelright=True)
    plt.loglog()
    plt.xlim(3, 110)
    plt.ylim(3, 110)
    plt.tight_layout()
    plt.savefig(pathplot+'m1m2_'+plot_name+'_'+itertag+'.png')
    plt.close()
    return


def color_m1m2_plot(medianlist_m1, medianlist_m2, M1, M2, median_val, median_rate, timesM=False, itertag='', pathplot='./', plot_name='meanchi'):
    median_to_plot = median_rate * (M1 * M2) if timesM else median_rate
    prefix = r'm_1m_2\,' if timesM else ''  # multiply by m for display
    if plot_name == 'meanchi':
        cmap='coolwarm'
        vmin, vmax = -0.2, 0.2
        colorbar_label = r'$\langle\chi_\mathrm{eff}\rangle(m_1, m_2)$'
    elif plot_name == 'stdchi':
        cmap='Purples'
        vmin, vmax = 0.05, 0.4
        colorbar_label = r'$\sigma(\chi_\mathrm{eff})(m_1, m_2)$'
    max_density = np.max(median_to_plot)
    max_exp = np.floor(2. * np.log10(max_density))  # Find the highest power of 10 below max_density
    contourlevels = 10 ** (0.5 * (max_exp - np.arange(6))[::-1])

    plt.figure(figsize=(8, 6.4))
    #pcm = plt.pcolormesh(M1, M2, map_colors, shading='auto')
    pcm = plt.pcolormesh(M1, M2, median_val, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    contours = plt.contour(M1, M2, median_to_plot, levels=contourlevels, colors='black', linewidths=1)
    cbar = plt.colorbar(pcm, label=colorbar_label)

    plt.fill_between(np.arange(3.01, 109.5), np.arange(3.01, 109.5), 109.5, color='white', alpha=1, zorder=50)
    #plt.scatter(medianlist_m1, medianlist_m2, color='r', marker='+', s=20)
    plt.xlabel(r"$m_\mathrm{1} \,[M_\odot]$")
    plt.ylabel(r"$m_\mathrm{2} \,[M_\odot]$")
    plt.tick_params(axis='y', which='both', left=False, right=True, labelleft=False, labelright=True)
    plt.loglog()
    plt.xlim(3, 110)
    plt.ylim(3, 110)
    plt.tight_layout()
    plt.savefig(pathplot+'m1m2_'+plot_name+'_'+itertag+'.png')
    plt.close()
    return


def m_chieff_contour(median_m1, median_chieff, M, CF, medianKDE, timesM=False, itertag='', pathplot='./', plot_name='KDE', xlabel='m_1'):
    # Set colorbar label based on plot_name
    prefix = xlabel if timesM else ''  # multiply by m for display
    massunits = r'' if timesM else r'M_\odot^{-1}'
    if plot_name == 'Rate':
        colorbar_label = r'$'+prefix+r'd \mathcal{R}/d'+xlabel+'d \chi_\mathrm{eff}[\mathrm{Gpc}^{-3} \mathrm{yr}^{-1}'+massunits+']$'
    else:
        colorbar_label = r'$p(' + xlabel + ', \chi_\mathrm{eff})$'

    kde_to_plot = medianKDE * M if timesM else medianKDE
    max_density = np.nanmax(kde_to_plot)
    max_exp = np.floor(np.log10(max_density))  # Highest power of 10 below max_density
    contourlevels = 10 ** (max_exp - np.arange(4))[::-1]

    # Plot
    plt.figure(figsize=(8, 6))
    norm_val = LogNorm(vmin=contourlevels[0], vmax=max_density)
    pcm = plt.pcolormesh(M, CF, kde_to_plot, cmap='Purples', norm=norm_val, shading='auto')
    contours = plt.contour(M, CF, kde_to_plot, levels=contourlevels, colors='black', linewidths=1)
    cbar = plt.colorbar(pcm, label=colorbar_label)

    plt.scatter(median_m1, median_chieff, color='r', marker='+', s=20)

    plt.ylabel(r"$\chi_\mathrm{eff}$")
    plt.xlabel(r'$' + xlabel + r'\,[M_\odot]$')
    if timesM: plt.semilogx()  # log mass axis
    plt.xlim(4, 110)

    plt.tight_layout()
    times = 'times_m_' if timesM else 'lin_'
    plt.savefig(pathplot+xlabel+"_chieff_"+plot_name+"_"+times+itertag+".png")
    plt.close()
    return


def m1chieff_at_m2_slice(median_m1, median_chieff, m2_src_grid, m2_target, M1, CF, KDElist, VTinterp, ndet=69, iterN=1, pathplot='./', xlabel='m_1', plot_name='KDE'):
    # Same code for making m2-chieff at fixed m1 !
    xname = 'm1' if xlabel == 'm_1' else 'm2'
    m2_idx = np.argmin(np.abs(m2_src_grid - m2_target))
    new_2Dlists = []
    for kde in KDElist:
        new_2Dlists.append(kde[:, m2_idx, :])
        #KDEaverage = np.percentile(KDElist, 50, axis=0)
        #KDE_slice = KDEaverage[:, m2_idx, :]
    data_slice = np.percentile(new_2Dlists, 50, axis=0)
    if plot_name == 'Rate':
        data_slice = ndet * data_slice / VTinterp
        colorbar_label = r'$d \mathcal{R}/dm_1 dm_2 d\chi_\mathrm{eff}[\mathrm{Gpc}^{-3} \mathrm{yr}^{-1} M_\odot^{-2}] $'
    else:
        colorbar_label = r'$p('+ xlabel +r', \chi_\mathrm{eff})$'
    max_density = np.nanmax(data_slice)
    max_exp = np.floor(np.log10(max_density))  # Find the highest power of 10 below max_density
    contourlevels = 10 ** (max_exp - np.arange(4))[::-1]

    plt.figure(figsize=(8, 6))
    norm_val = LogNorm(vmin=contourlevels[0], vmax=max_density)
    pcm = plt.pcolormesh(M1, CF, data_slice, cmap='Purples', norm=norm_val, shading='auto')
    contours = plt.contour(M1, CF, data_slice, levels=contourlevels, colors='black', linewidths=1)
    #plt.clabel(contours, fmt="% .1e", colors='black', fontsize=14)

    cbar = plt.colorbar(pcm, label=colorbar_label)
    #cbar.set_ticks(contourlevels)
    plt.scatter(median_m1, median_cheff, color='r', marker='+', s=20)
    plt.ylabel(r"$\chi_\mathrm{eff}$")
    plt.xlabel(r"$" + xlabel + r"\,[M_\odot]$")
    plt.semilogx()
    #plt.title(r"$m_2 = {0}$".format(m2_target), fontsize=18)
    plt.tight_layout()
    plt.savefig(pathplot+plot_name+'_'+xname+f"chieff_at{m2_target}_Iter{iterN}.png")
    plt.close()
    return


def m1m2_at_chieff_slice(median_m1, median_m2, chieff_grid, chieff_target, M1, M2, KDElist, VTinterp, ndet=69, iterN=1, pathplot='./', plot_name='KDE'):
    chi_idx = np.argmin(np.abs(chieff_grid - chieff_target))
    new_2Dlists = []
    for kde in KDElist:
        new_2Dlists.append(kde[:, :, chi_idx])
        #KDEaverage = np.percentile(KDElist, 50, axis=0)
        #KDE_slice = KDEaverage[:, m2_idx, :]
    data_slice = np.percentile(new_2Dlists, 50, axis=0)
    if plot_name == 'Rate':
        data_slice = ndet * data_slice / VTinterp
        colorbar_label = r'$d \mathcal{R}/dm_1 dm_2 d\chi_\mathrm{eff}[\mathrm{Gpc}^{-3} \mathrm{yr}^{-1} M_\odot^{-2}] $'
    else:
        colorbar_label = r'$p(m_1, m_2)$'
    max_density = np.nanmax(data_slice)
    max_exp = np.floor(np.log10(max_density))  # Find the highest power of 10 below max_density
    contourlevels = 10 ** (max_exp - np.arange(4))[::-1]

    # Plot
    plt.figure(figsize=(8, 6))
    norm1 = LogNorm(vmin=contourlevels[0], vmax=max_density)
    #pcm = plt.pcolormesh(M1, M2, data_slice, cmap='viridis', norm=norm, shading='auto')
    pcm = plt.pcolormesh(M1, M2, data_slice, cmap='Purples', norm=norm1, shading='auto')
    contours = plt.contour(M1,  M2, data_slice, levels=contourlevels, colors='black', linewidths=1.5)
    #plt.clabel(contours, fmt="% .1e", colors='black', fontsize=14)

    # Colorbar
    cbar = plt.colorbar(pcm, label=colorbar_label)
    #cbar.set_ticks(contourlevels[:-1])
    plt.fill_between(np.arange(0, 105), np.arange(0, 105), 105, color='white',alpha=1,zorder=50)
    #plt.fill_between(np.arange(0, 50), np.arange(0, 50), 50 , color='white',alpha=1,zorder=100)
    plt.scatter(medianlist_m1, medianlist_m2, color='r', marker='+', s=20)
    plt.xlabel(r"$m_\mathrm{1} \,[M_\odot]$")
    plt.ylabel(r"$m_\mathrm{2} \,[M_\odot]$")
    plt.tick_params(axis='y', which='both', left=False, right=True, labelleft=False, labelright=True)
    plt.loglog()
    plt.xlim(3, 105)
    plt.ylim(3, 105)
    plt.title(r"$\chi_\mathrm{{eff}} = {0}$".format(xi_target), fontsize=18)
    plt.tight_layout()
    plt.savefig(pathplot+plot_name+f"m1m2_atchieff{chieff_target}_Iter{iterN}.png")
    plt.close()
    return


###################################
def plot_pdet_scatter(flat_samples1, flat_samples2, flat_pdet, xlabel=r'$m_{1, source} [M_\odot]$', ylabel=r'$d_L [Mpc]$', save_name="pdet_power_law_m2_m1_dL_scatter.png", pathplot='./'):
    flat_pdet = flat_pdet/1e9  # convert VT to Gpc^3 yr
    plt.figure(figsize=(8, 6))
    plt.scatter(flat_samples1, flat_samples2, c=flat_pdet, s=10, cmap='viridis', norm=LogNorm(vmin=min(flat_pdet), vmax=max(flat_pdet)))
    cbar = plt.colorbar() # label=r'$p_\mathrm{det}$')
    #cbar.set_label(r'$p_\mathrm{det}$', fontsize=20)
    cbar.set_label(r'$\mathrm{VT} [\mathrm{Gpc}^3\,\mathrm{yr}]$', fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.semilogx()
    #plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(pathplot+save_name)
    plt.close()

    return


def plot_pdet_3Dscatter(flat_samples1, flat_samples2, flat_samples3, flat_pdet, save_name="pdet_m1m2dL_3Dscatter.png", pathplot='./'):
    flat_pdet = flat_pdet/1e9  # convert VT to Gpc^3 yr
    from mpl_toolkits.mplot3d import Axes3D
    # 3D scatter plot
    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the data with logarithmic color scaling
    sc = ax.scatter(np.log10(flat_samples1), np.log10(flat_samples2), flat_samples3, c=flat_pdet, cmap='viridis', s=10, norm=LogNorm(vmin=flat_pdet.min(), vmax=flat_pdet.max()), alpha=0.6)
    plt.colorbar(sc, label=r'$VT(m_1, m_2, \chi_\mathrm{eff})$ [Mpc$^3$\,yr]')
    #ax.set_xscale('log')
    #ax.set_yscale('log')

    ax.set_xlabel(r'$\log_{10} m_{1,{\rm src}} [M_\odot]$', fontsize=18)
    ax.set_ylabel(r'$\log_{10} m_{2,{\rm src}} [M_\odot]$', fontsize=18)
    ax.set_zlabel(r'$\chi_\mathrm{eff}$', fontsize=18)

    plt.tight_layout()
    plt.savefig(pathplot+save_name)
    plt.close()

    return


def plot_pdetscatter_m1dL_redshiftYaxis(flat_samples1, flat_samples2, flat_pdetlist, flat_samples_z, xlabel=r'$m_{1, source} [M_\odot]$', ylabel=r'$d_L [Mpc]$', title=r'$p_\mathrm{det}$',  save_name="pdet_m1m2dL_3Dscatter.png", pathplot='./', show_plot=False):
    fig, ax1 = plt.subplots(figsize=(8, 6))
    # Scatter plot on the primary axis
    scatter = ax1.scatter(flat_samples1, flat_samples2, c=flat_pdetlist, s=10, cmap='viridis', norm=LogNorm(vmin=1e-5, vmax=1))
    cbar = plt.colorbar(scatter, ax=ax1, label=r'$p_\mathrm{det}$',  pad=0.1)
    cbar.set_label(r'$p_\mathrm{det}$', fontsize=20)
    # Primary axis labels and log scale
    ax1.set_xlabel(xlabel, fontsize=20)
    ax1.set_ylabel(ylabel, fontsize=20)
    # Secondary y-axis for flat_sample3
    ax2 = ax1.twinx()  # Create a twin y-axis
    ax2.semilogx()
    ax2.scatter(flat_samples1, flat_samples_z, c=flat_pdetlist, s=1, cmap='viridis', norm=LogNorm(vmin=1e-5, vmax=1), alpha=0)
    custom_ticks = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2]  # Custom tick positions
    ax2.set_yticks(custom_ticks)  # Specify custom ticks
    ax2.set_yticklabels([f'{val:.1f}' for val in custom_ticks])
    ax2.set_ylabel(r'$z$', fontsize=20)  # Add secondary axis label
    ax2.set_ylim(0.1, )
    ax2.grid(False)
    ax1.semilogx()
    plt.subplots_adjust(right=0.6)  # Shift the plot slightly left to make space for the secondary y-axis and color bar

    #plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(pathplot+save_name)
    if show_plot: plt.show()
    plt.close()

    return


def plot_pdet2D(XX, YY, pdet2D, Maxpdet=0.1, pathplot='./',  save_name='testpdet2Dpowerlaw_m2_on evalgrid.png' , show_plot=False):
    """
    Plots the 2D detection probability (pdet) with contours and saves the figure.

    Args:
        XX (ndarray): Meshgrid for the x-axis (e.g., source mass).
        YY (ndarray): Meshgrid for the y-axis (e.g., distance).
        pdet2D (ndarray): 2D array of detection probabilities.
        opts (object): Options object containing plot settings (e.g., Maxpdet and pathplot).
    """
    plt.figure(figsize=(10, 8))

    # Set contour levels and title based on Maxpdet
    if  Maxpdet == '0.03':
        levels = [0.03, 0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0]
        title = r'$p_\mathrm{det}, \,  q^{1.26}, \, \mathrm{with} \, max(0.03, p_\mathrm{det})$'
    else:
        levels = [0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0]
        title = r'$p_\mathrm{det}, \,  q^{1.26}, \, \mathrm{with} \, max(0.1, p_\mathrm{det})$'

    # Create filled contour plot
    plt.contourf(XX, YY, pdet2D, levels=levels, cmap='viridis', norm=Normalize(vmax=1))
    plt.title(title, fontsize=18)

    # Add colorbar
    plt.colorbar(label=r'$p_\mathrm{det}$')

    # Add contour lines
    plt.contour(XX, YY, pdet2D, colors='white', linestyles='dashed', levels=levels[1:])

    # Add labels and scale
    plt.xlabel(r'$m_{1, source} [M_\odot]$', fontsize=20)
    plt.ylabel(r'$d_L [Mpc]$', fontsize=20)
    plt.loglog()

    # Save and close the plot
    plt.savefig(pathplot + save_name)
    if show_plot: plt.show()
    plt.close()

    return


def average2D_m1dL_kde_plot(m1vals, dLvals, XX, YY, kdelists, pathplot='./', titlename=1, plot_label='Rate', x_label='m1', y_label='m2', plottag='Average', dLval=500):
    sample1, sample2 = m1vals, dLvals
    CI50 = np.percentile(kdelists, 50, axis=0)
    print("inside", CI50.shape)
    max_density = np.max(CI50)
    max_exp = np.floor(np.log10(max_density))  # Find the highest power of 10 below max_density
    contourlevels = 10 ** (max_exp - np.arange(3))[::-1]
    fig, axl = plt.subplots(1,1,figsize=(8,6))
    p = axl.pcolormesh(XX, YY, CI50, cmap=plt.cm.get_cmap('Purples'), norm=LogNorm(vmin=contourlevels[0], vmax=contourlevels[-1]))
    cbar = plt.colorbar(p, ax= axl)
    if plot_label =='KDE':
        cbar.set_label(r'$p(m_1, m_2)$',fontsize=18)
    else:
        cbar.set_label(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}m_2\mathrm{d}d_L [\mathrm{Mpc}^{-1}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-2}]$',fontsize=18)
    CS = axl.contour(XX, YY, CI50, colors='black', levels=contourlevels ,linestyles='dashed', linewidths=2, norm=LogNorm(vmin=contourlevels[0], vmax=contourlevels[-1]))
    axl.scatter(sample1, sample2,  marker="+", color="r", s=20)
    axl.scatter(sample2, sample1,  marker="+", color="r", s=20)
    axl.fill_between(np.arange(0, 100), np.arange(0, 100),100 , color='white',alpha=1,zorder=50)
    axl.fill_between(np.arange(0, 50), np.arange(0, 50), 50 , color='white',alpha=1,zorder=100)
    axl.set_ylim(3, 101)
    axl.loglog()
    axl.set_aspect('equal')
    axl.set_title('dL={0:.1f}[Mpc]'.format(dLval))
    fig.tight_layout()
    plt.savefig(pathplot+plottag+'m1_'+y_label+'_2D'+plot_label+'Iter{0}dL{1:.3f}.png'.format(titlename, dLval), bbox_inches='tight')
    plt.close()

    return CI50

# combined average rate plot with correct units and contour line of pdet
def special_plot_rate(meanxi1, meanxi2, XX, YY, pdet2Dnofilter, CI50, save_name="Special_pdetcontourlines_on_combined_average_Rate1000Iteration.png", pathplot='./'):
    fig, axl = plt.subplots(1, 1, figsize=(8, 6))
    levels_pdet = [0.01, 0.03, 0.1]
    # Plot PDET contours
    pdet_contour = axl.contour(
        XX, YY, pdet2Dnofilter,
        colors=['orange'] * len(levels_pdet),
        levels=levels_pdet,
        linewidths=2,
        linestyles=['--'] * len(levels_pdet)
    )

    # Add labels to contours
    contour_label_positions = []
    for collection in pdet_contour.collections:
        paths = collection.get_paths()
        for path in paths:
            vertices = path.vertices
            midpoint_index = len(vertices) // 2
            contour_label_positions.append(vertices[midpoint_index])

    axl.clabel(
        pdet_contour,
        inline=True,
        inline_spacing=8,
        use_clabeltext=True,
        fontsize=16,
        fmt="%.2f",
        manual=contour_label_positions
    )

    # Determine contour levels for CI50
    max_density = np.max(CI50)
    max_exp = np.floor(np.log10(max_density))
    contourlevels = 10 ** (max_exp - np.arange(3))[::-1]
    # CI50 colormap
    p = axl.pcolormesh(
        XX, YY, CI50,
        cmap=plt.cm.Purples,
        norm=LogNorm(vmin=contourlevels[0], vmax=contourlevels[-1]),
        shading='auto'
    )
    cbar = plt.colorbar(p, ax=axl)
    cbar.set_label(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}V_c [\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-1}]$', fontsize=20)

    # CI50 contour lines
    axl.contour(
        XX, YY, CI50,
        colors='black',
        levels=contourlevels,
        linestyles='dashed',
        linewidths=1.5,
        norm=LogNorm(vmin=contourlevels[0], vmax=contourlevels[-1])
    )

    # Add scatter points
    axl.scatter(meanxi1, meanxi2, marker="+", color="r", s=20)
    # Set axis labels and limits
    axl.set_ylabel(r'$d_L\,[\mathrm{Mpc}]$', fontsize=20)
    axl.set_xlabel(r'$m_\mathrm{1, source} \,[M_\odot]$', fontsize=20)
    axl.set_ylim(ymin=200, ymax=7000)
    axl.semilogx()

    # Save the plot
    fig.tight_layout()
    plt.savefig(pathplot + save_name)
    plt.close()



def new2DKDE(XX, YY,  ZZ, Mv, z, saveplot=False, plot_label='KDE', title='median', show_plot=False, pathplot='./'):
    plt.figure(figsize=(8, 6))
    max_density = np.max(ZZ)
    max_exp = np.floor(np.log10(max_density))  # Find the highest power of 10 below max_density
    contourlevels = 10 ** (max_exp - np.arange(4))[::-1]
    #contourlevels = np.logspace(-5, 0, 11)[:]

    # Plotting pcolormesh and contour
    p = plt.pcolormesh(XX, YY, ZZ, cmap=plt.cm.get_cmap('Purples'),  norm=LogNorm(vmin=contourlevels[0], vmax=contourlevels[-1]))#vmin=1e-5))
    CS = plt.contour(XX, YY, ZZ, colors='black', linestyles='dashed', linewidths=2, norm=LogNorm(), levels=contourlevels)
    plt.plot(Mv, z, 'r+')
    # Colorbar
    cbar = plt.colorbar(p)
    if plot_label =='Rate':
        cbar.set_label(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}d_L [\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-1}]$',fontsize=18)
    else:
        cbar.set_label(r'$p(m_{1,source}, d_L)$',fontsize=18)
    cbar.ax.tick_params(labelsize=20)

    plt.tick_params(labelsize=15)
    #plt.xlabel(r'$m_\mathrm{1, detector} \, [M_\odot]$', fontsize=20)
    plt.xlabel(r'$m_\mathrm{1, source} \, [M_\odot]$', fontsize=20)

    # Y-axis formatting
    scale_y = 1  # Adjust if needed, e.g., 1e3 for scaling
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / scale_y))
    plt.gca().yaxis.set_major_formatter(ticks_y)
    plt.ylabel(r"$d_L\, [Mpc]$", fontsize=20)
    plt.loglog()
    plt.tight_layout()
    if saveplot:
        plt.savefig(pathplot+plot_label+title+'.png')
    else:
        print("notsaving")
    if show_plot: plt.show()
    plt.close()
    return


def histogram_bw(bws, dataname='bw',  pathplot='./', tag=1):
    """
    """
    plt.figure(figsize=(8, 6))
    plt.xlabel(dataname, fontsize=15)
    bws = np.array(bws)
    if 'bw' in dataname:  # use log spaced bins
        plt.hist(bws, bins=np.logspace(np.log10(bws.min()), np.log10(bws.max()), 10), color='red', histtype='step', density=True)
        plt.semilogx()
        plt.xlim(0.7 * bws.min(), 1.4 * bws.max())
    elif dataname == 'alpha':
        plt.hist(bws, bins=np.linspace(0, 1, 15), color='red', histtype='step', density=True)
        plt.xlim(0, 1)
    else:
        plt.hist(bws, bins=10, color='red', histtype='step', density=True)
        plt.xlim(bws.min() - 0.1, bws.max() + 0.1)

    plt.savefig(pathplot+dataname+f"_hist_iter{tag}.png", bbox_inches='tight')
    plt.close()
    return


def average2Dkde_m1m2_plot(m1vals, m2vals, XX, YY, kdelists, pathplot='./', titlename=1, plot_label='Rate', x_label='m1', y_label='m2', plottag='Average', dLval=500, correct_units=False):
    if correct_units:
        volume_factor = get_dVdz_factor(dLval) #one value
    else:
        volume_factor = 1.0
    sample1, sample2 = m1vals, m2vals
    CI50 = np.percentile(kdelists, 50, axis=0)#/volume_factor
    max_density = np.max(CI50)
    max_exp = np.floor(np.log10(max_density))  # Find the highest power of 10 below max_density
    contourlevels = 10 ** (max_exp - np.arange(4))[::-1]
    fig, axl = plt.subplots(1,1,figsize=(8,6))
    p = axl.pcolormesh(XX, YY, CI50, cmap=plt.cm.get_cmap('Purples'), norm=LogNorm(vmin=contourlevels[0], vmax=contourlevels[-1]))
    cbar = plt.colorbar(p, ax= axl)
    if plot_label == 'Rate':
        if correct_units:
            cbar.set_label(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}m_2\mathrm{d}dV_c [\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-2}]$',fontsize=18)
        else:
            #cbar.set_label(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}m_2\mathrm{d}dV_c [\mathrm{Mpc}^{-1}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-2}]$',fontsize=18)
            cbar.set_label(r'$\mathrm{d}^2\mathcal{R}/\mathrm{d}m_1\mathrm{d}m_2[\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-2}]$',fontsize=18)
    else:
        cbar.set_label(r'$p(m_{1, source}, d_L)$',fontsize=18)
    CS = axl.contour(XX, YY, CI50, colors='black', levels=contourlevels ,linestyles='dashed', linewidths=2, norm=LogNorm(vmin=contourlevels[0], vmax=max_density))
    axl.scatter(sample1, sample2,  marker="+", color="r", s=20)
    axl.scatter(sample2, sample1,  marker="+", color="r", s=20)
    axl.fill_between(np.arange(0, 100), np.arange(0, 100),100 , color='white',alpha=1,zorder=50)
    axl.fill_between(np.arange(0, 50), np.arange(0, 50), 50 , color='white',alpha=1,zorder=100)
    axl.set_ylim(4, 101)
    axl.tick_params(axis="y",direction="in")
    axl.yaxis.tick_right()
    axl.yaxis.set_ticks_position('both')
    axl.set_ylabel(r'$m_2\,[M_\odot]$', fontsize=18)
    cbar.ax.tick_params(labelsize=20)
    axl.tick_params(labelsize=18)
    axl.set_xlabel(r'$m_1\,[M_\odot]$', fontsize=18)
    scale_y = 1#e3
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
    axl.yaxis.set_major_formatter(ticks_y)
    axl.set_xlim(4, 100.1)
    axl.loglog()
    axl.set_aspect('equal')
    #axl.set_title(r'$d_L=${0}[Mpc]'.format(dLval), fontsize=18)
    fig.tight_layout()
    plt.show()
    #plt.savefig(pathplot+plottag+'m1_'+y_label+'_2D'+plot_label+'Iter{0}dL{1:.3f}.png'.format(titlename, dLval), bbox_inches='tight')
    plt.close()

    return CI50


def average2DlineardLrate_plot(m1vals, m2vals, XX, YY, kdelists, pathplot='./', titlename=1, plot_label='Rate', x_label='m1', y_label='dL', show_plot=False):
    sample1, sample2 = m1vals, m2vals
    CI50 = np.percentile(kdelists, 50, axis=0)
    fig, axl = plt.subplots(1,1,figsize=(8,6))
    max_density = np.max(CI50)
    max_exp = np.floor(np.log10(max_density))  # Find the highest power of 10 below max_density
    contourlevels = 10 ** (max_exp - np.arange(3))[::-1]
    p = axl.pcolormesh(XX, YY, CI50, cmap=plt.cm.get_cmap('Purples'), norm=LogNorm(vmin=contourlevels[0], vmax =contourlevels[-1]),  label=r'$p(m_1, d_L)$')
    cbar = plt.colorbar(p, ax= axl)
    if plot_label =='Rate':
        cbar.set_label(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}d_L [\mathrm{Mpc}^{-1}\, (\mathrm{m}/{M}_\odot)^{-1}  \mathrm{yr}^{-1}]$',fontsize=18)
    else:
         cbar.set_label(r'$p(m_{1, source}, d_L)$',fontsize=18)
    CS = axl.contour(XX, YY, CI50, colors='black', levels=contourlevels ,linestyles='dashed', linewidths=2, norm=LogNorm())
    axl.scatter(sample1, sample2,  marker="+", color="r", s=20)
    axl.set_ylabel(r'$d_L\,[Mpc]$', fontsize=18)
    axl.set_xlabel(r'$m_\mathrm{1, source} \,[M_\odot]$', fontsize=18)
    #axl.legend()
    axl.set_ylim(ymax=7000)
    axl.semilogx()
    #axl.set_aspect('equal')
    fig.tight_layout()
    plt.savefig(pathplot+"average_"+plot_label+"{0}.png".format(titlename))
    if show_plot: plt.show()
    else: plt.close()

    return CI50


def bw_correlation(bwlist, n_corr=100, param='bw', pathplot='./', log=True):
    """
    Plot BW time series and correlation coefficient
    """
    plt.figure()
    plt.plot(bwlist, '+')
    if min(bwlist) > 0:
        plt.semilogy()
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel(param)
    plt.savefig(pathplot+param+'_iters.png', bbox_inches='tight')
    plt.close()

    Cxy = []
    iternumber = []
    for i in range(int(len(bwlist)/2)):
        iternumber.append(i+1)
        if log == False:
            M = np.corrcoef(bwlist[i+1:], bwlist[:-i-1])
        else:
            M = np.corrcoef(np.log(bwlist[i+1:]), np.log(bwlist[:-i-1]))
        Cxy.append(M[0][1])
    plt.figure(figsize=(8,4))
    plt.plot(iternumber[n_corr:], Cxy[n_corr:],'+-')
    plt.xlabel('Number of iterations')
    plt.ylabel(r'$C_{01}$', fontsize=14)
    plt.semilogx()
    plt.grid(True)
    plt.savefig(pathplot+param+f"C01_iter_after_{n_corr}.png", bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(8, 4))
    plt.plot(iternumber[:n_corr], Cxy[:n_corr],'+-')
    plt.xlabel('Number of iterations')
    plt.ylabel(r'$C_{01}$')
    #plt.semilogx()
    plt.grid(True)
    plt.savefig(pathplot+param+f"C01_iter_before_{n_corr}.png", bbox_inches='tight')
    plt.close()

    #corrcoefficient
    Cxy = []
    for i in range(int(len(bwlist)/2)):
        if log:
            M = np.corrcoef(np.log(bwlist[i+1:]), np.log(bwlist[:-i-1]))
        else:
            M = np.corrcoef(bwlist[i+1:], bwlist[:-i-1])
        Cxy.append(M[0][1])
    plt.figure(figsize=(8, 4))
    plt.plot(Cxy,'+-')
    plt.xlabel('Number of iterations')
    plt.xlim(xmin=1)
    plt.ylabel(r'$C_{xy}$')
    plt.semilogx()
    plt.grid(True)
    plt.savefig(pathplot+param+"CxyiterN.png", bbox_inches='tight')
    plt.close()
    return

