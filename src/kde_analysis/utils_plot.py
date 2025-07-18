#Jam Sadiq
# A script for making plots for the paper

#import libraries
import numpy as np
import matplotlib.pyplot as plt
<<<<<<< HEAD
=======
import scipy
from scipy.integrate import simpson
>>>>>>> 629fe549e028ed7a2df81328618ce8d857c9db88
from matplotlib import rcParams
from matplotlib.colors import LogNorm, Normalize
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import matplotlib.patches
from matplotlib.patches import Rectangle
import glob
import deepdish as dd

rcParams["text.usetex"] = True
rcParams["font.serif"] = "Computer Modern"
rcParams["font.family"] = "Serif"
rcParams["xtick.labelsize"]=18
rcParams["ytick.labelsize"]=18
rcParams["xtick.direction"]="in"
rcParams["ytick.direction"]="in"
rcParams["legend.fontsize"]=18
rcParams["axes.labelsize"]=18
rcParams["axes.grid"] = True
rcParams["grid.color"] = 'grey'
rcParams["grid.linewidth"] = 1.
#rcParams["grid.linestyle"] = ':'
rcParams["grid.alpha"] = 0.6



dict_p = {'m1':'m_1', 'm2':'m_2', 'Xieff':'\chi_{eff}', 'chieff': '\chi_{eff}', 'DL':'D_L', 'logm1':'ln m_1', 'logm2': 'ln m_2', 'alpha':'\alpha'}
###########
<<<<<<< HEAD
############# m1-Xieff plot 2D slice plot
def get_m1Xieff_at_m2_slice_plot(medianlist_m1, medianlist_xieff, m2_src_grid, m2_target, M1, XIEFF, KDElist, VTinterp,  iterN=1, pathplot='./', plot_name='KDE'):
=======
##OneD mass Rates
def Rate_masses(m1_src_grid, m2_src_grid, ratem1_arr, ratem2_arr, pathplot='./'):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Define colors
    color_m1 = 'royalblue'
    color_m2 = 'darkorange'

    # Plot m1 data
    median_m1 = np.median(ratem1_arr, axis=0)
    p5_m1 = np.percentile(ratem1_arr, 5., axis=0)
    p95_m1 = np.percentile(ratem1_arr, 95., axis=0)
    ax.plot(m1_src_grid, median_m1, color=color_m1, linewidth=2, label='primary-mass')
    ax.fill_between(m1_src_grid, p5_m1, p95_m1, color=color_m1, alpha=0.3)

    # Plot m2 data
    median_m2 = np.median(ratem2_arr, axis=0)
    p5_m2 = np.percentile(ratem2_arr, 5., axis=0)
    p95_m2 = np.percentile(ratem2_arr, 95., axis=0)
    ax.plot(m2_src_grid, median_m2, color=color_m2, linewidth=2, label='secondary-mass')
    ax.fill_between(m2_src_grid, p5_m2, p95_m2, color=color_m2, alpha=0.3)

    ax.set_xlim(4., 105.)
    ax.legend()
    ax.grid(True, ls="--")
    ax.set_ylabel(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m\mathrm{d}\chi_\mathrm{eff}[\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-1}]$', fontsize=14)
    ax.set_xlabel(r"$m$", fontsize=20)
    plt.semilogy()
    plt.tight_layout()
    plt.savefig(pathplot+"Rate_masses_Marginalized.png")
    plt.close()
    return 0

########offset plot
def Xieff_offset_plot(m_grid, Xieff_grid, m_slice_values, rate2D_m_Xieff_list, offset_increment=5, m_label='m_1', pathplot='./'):
    colormap = plt.cm.magma
    norm = Normalize(vmin=min(m_slice_values), vmax=max(m_slice_values)+5)
    offset = 0
    plt.figure(figsize=(8, 8))
    for i, m_val in enumerate(m_slice_values):
        color = colormap(norm(m_slice_values[i]))
        rateXieff_slice_m = []
        idx = np.argmin(np.abs(m_grid -  m_val))
        for rate in rate2D_m_Xieff_list:
            slice_rate = rate[idx, :]
            normalize = simpson(y=slice_rate, x=Xieff_grid) #+ 1e-10 to avoid zero 
            prob = slice_rate/normalize
            rateXieff_slice_m.append(prob)
        median = np.percentile(rateXieff_slice_m, 50., axis=0)
        p05 = np.percentile(rateXieff_slice_m, 5., axis=0)
        p95 = np.percentile(rateXieff_slice_m, 95., axis=0)
        plt.plot(Xieff_grid, median+offset, color=color, linewidth=2,label=r'$'+m_label+'={0}$'.format(m_val))
        plt.fill_between(Xieff_grid, p05+offset, p95+offset,color=color, alpha=0.3)
        plt.axhline(y=offset, color='grey', linestyle='-.', alpha=0.5)
        plt.text(-0.67, offset+0.7, "$"+m_label+"={0}$".format(m_val), fontsize=14, color='k', verticalalignment='center')
        offset += offset_increment
    plt.xlim(-0.72, 0.72)
    plt.ylabel(r'$p(\chi_\mathrm{eff}| '+m_label+')$ + offset', fontsize=20)
    plt.xlabel(r"$\chi_\mathrm{eff}$", fontsize=20)
    plt.grid(False)
    plt.yticks([]) #remove y-ticks
    plt.tight_layout()
    plt.savefig(f"{pathplot}offset_plot_Xieff_at_{m_label}_slice.png")
    plt.close()
    return 0
############# m1-Xieff plot 2D slice plot
def get_averagem1m2_plot(medianlist_m1, medianlist_m2, M1, M2, KDElist, iterN=1, pathplot='./', plot_name='KDE'):
    data_slice = np.percentile(KDElist, 50, axis=0)
    if plot_name=='Rate':
        colorbar_label = r'$d \mathcal{R}/dm_1 dm_2 [\mathrm{Gpc}^{-3} \mathrm{yr}^{-1} M_\odot^{-2}] $'
    else:
        colorbar_label = r'$p(m_1, m_2)$'
    max_density = np.max(data_slice)
    max_exp = np.floor(np.log10(max_density))  # Find the highest power of 10 below max_density
    contourlevels = 10 ** (max_exp - np.arange(4))[::-1]
    contourlevels[-1] = max_density
    plt.figure(figsize=(8, 6))
    norm1 = LogNorm(vmin=contourlevels[0], vmax=max_density)  # Apply log normalization
    pcm = plt.pcolormesh(M1, M2, data_slice, cmap='Purples', norm=norm1, shading='auto')
    contours = plt.contour(M1,  M2, data_slice, levels=np.logspace(np.log10(contourlevels[0]),np.log10(max_density), 5), colors='black', linewidths=1.5, norm=LogNorm()) 
    cbar = plt.colorbar(pcm, label=colorbar_label)

    plt.fill_between(np.arange(0, 105), np.arange(0, 105), 105 , color='white',alpha=1,zorder=50)
    plt.scatter(medianlist_m1, medianlist_m2, color='r', marker='+', s=20)
    plt.xlabel(r"$m_\mathrm{1} \,[M_\odot]$")
    plt.ylabel(r"$m_\mathrm{2} \,[M_\odot]$")
    plt.tick_params(axis='y', which='both', left=False, right=True, labelleft=False, labelright=True)
    plt.semilogx()
    plt.semilogy()
    plt.xlim(4, 102)
    plt.ylim(4, 102)
    plt.tight_layout()
    plt.savefig('Average_m1_m2'+plot_name+'int_wrt_Xieff_Iter'+iterN+'.png')
    plt.close()
    return 0

def get_m_Xieff_plot(medianlist_m1, medianlist_xieff, M1, XIEFF, KDElist, iterN=1, pathplot='./', plot_name='KDE', xlabel='m-1'):
    data_slice = np.percentile(KDElist, 50, axis=0)
    # Set colorbar label based on plot_name
    if plot_name == 'Rate':
        colorbar_label = r'$d \mathcal{R}/d'+ xlabel+'d \chi_\mathrm{eff}[\mathrm{Gpc}^{-3} \mathrm{yr}^{-1} M_\odot^{-1}] $'
    else:
        colorbar_label = r'$p(' + xlabel + ', \chi_\mathrm{eff})$'

    max_density = np.nanmax(data_slice)
    max_exp = np.floor(np.log10(max_density))  # Highest power of 10 below max_density

    contourlevels = 10 ** (max_exp - np.arange(4))[::-1]  
    # Plot
    plt.figure(figsize=(8, 6))
    
    # Use LogNorm for both pcolormesh and contour
    norm_val = LogNorm(vmin=contourlevels[0], vmax=max_density)
    contourlevels = np.logspace(np.log10(contourlevels[0]), np.log10(max_density), 5)
    pcm = plt.pcolormesh(M1, XIEFF, data_slice, cmap='Purples', norm=norm_val, shading='auto')

    contours = plt.contour(M1, XIEFF, data_slice, levels=contourlevels, colors='black', linewidths=1.5)

    cbar = plt.colorbar(pcm, label=colorbar_label)

    plt.scatter(medianlist_m1, medianlist_xieff, color='r', marker='+', s=20)

    plt.ylabel(r"$\chi_\mathrm{eff}$")
    plt.xlabel(r'$' + xlabel + r'\,[M_\odot]$')
    plt.semilogx()  # Use logarithmic x-axis
    plt.tight_layout()
    plt.savefig("Average_2D_"+plot_name+"_Xieff_"+xlabel+"Iter_"+iterN+"_.png")
    plt.close()
    return 0 


def get_m1Xieff_at_m2_slice_plot(medianlist_m1, medianlist_xieff, m2_src_grid, m2_target, M1, XIEFF, KDElist, VTinterp,  iterN=1, pathplot='./', plot_name='KDE'):
    m2_idx = np.argmin(np.abs(m2_src_grid - m2_target))
>>>>>>> 629fe549e028ed7a2df81328618ce8d857c9db88
    new_2Dlists = []
    for kde in KDElist:
        new_2Dlists.append(kde[:, m2_idx, :])
        #KDEaverage = np.percentile(KDElist, 50, axis=0)
        #KDE_slice = KDEaverage[:, m2_idx, :] 
    data_slice = np.percentile(new_2Dlists, 50, axis=0)
    if plot_name=='Rate':
<<<<<<< HEAD
        data_slice = 69*KDE_slice/VTinterp #69 is numbe of observed BBH signals         
        colorbar_label = r'$\mathcal{R}(m_1, \chi_\mathrm{eff})$"'
    else:
        colorbar_label = r'KDE-value'
    max_density = np.nanmax(data_slice)
    max_exp = np.floor(np.log10(max_density))  # Find the highest power of 10 below max_density
    contourlevels = 10 ** (max_exp - np.arange(4))[::-1]
    vmin, vmax = contourlevels[0] , max_density # np.nanmax(KDE_slice)  # Min and max values for KDE
=======
        data_slice = 69*data_slice/VTinterp #69 is numbe of observed BBH signals         
        colorbar_label = r'$d \mathcal{R}/dm_1 dm_2 d\chi_\mathrm{eff}[\mathrm{Gpc}^{-3} \mathrm{yr}^{-1} M_\odot^{-2}] $'
    else:
        colorbar_label = r'$p(m_1, \chi_\mathrm{eff})$'
    max_density = np.nanmax(data_slice)
    max_exp = np.floor(np.log10(max_density))  # Find the highest power of 10 below max_density
    contourlevels = 10 ** (max_exp - np.arange(4))[::-1]
    #contourlevels[-1] = max_density
    vmin, vmax = contourlevels[0],  max_density  # Min and max values for KDE
>>>>>>> 629fe549e028ed7a2df81328618ce8d857c9db88
    print("vmin, vmax is =", vmin, vmax)
    # Plot
    plt.figure(figsize=(8, 6))
    norm_val = LogNorm(vmin=vmin, vmax=vmax)  # Apply log normalization
<<<<<<< HEAD
    pcm = plt.pcolormesh(M1, XIEFF, data_slice, cmap='Purples', norm=norm_val, shading='auto')
    contours = plt.contour(M1,  XIEFF, data_slice, levels=contourlevels, colors='black', linewidths=0.5)

    # Colorbar
    cbar = plt.colorbar(pcm, label=colorbar_label)
    plt.scatter(medianlist_m1, medianlist_xieff, color='blue', marker='+', s=20)
    plt.ylabel(r"$\chi_\mathrm{effective}$")
    plt.xlabel(r"$m_\mathrm{1,source} \,[M_\odot]$")
    plt.semilogx()
    plt.title(f"Slice at m2 = {m2_target}")
    plt.tight_layout()
    plt.savefig(pathplot+"Average"+plot_name+"m1Xieffatm2_{1}_Iter{0}.png".format(iterN, m2_target))
    plt.close()
    return 0

=======
    #pcm = plt.pcolormesh(M1, XIEFF, data_slice, cmap='viridis', norm=norm, shading='auto')
    pcm = plt.pcolormesh(M1, XIEFF, data_slice, cmap='Purples', norm=norm_val, shading='auto')
    contours = plt.contour(M1,  XIEFF, data_slice, levels=contourlevels, colors='black', linewidths=1.5)
    #plt.clabel(contours, fmt="% .1e", colors='black', fontsize=14)

    # Colorbar
    cbar = plt.colorbar(pcm, label=colorbar_label)
    #cbar.set_ticks(contourlevels)
    plt.scatter(medianlist_m1, medianlist_xieff, color='r', marker='+', s=20)
    plt.ylabel(r"$\chi_\mathrm{eff}$")
    plt.xlabel(r"$m_\mathrm{1,source} \,[M_\odot]$")
    plt.semilogx()
    plt.title(r"$m_2 = {0}$".format(m2_target), fontsize=18)
    plt.tight_layout()
    #plt.savefig(pathplot+"Average"+plot_name+"m1Xieffatm2_{1}_Iter{0}.png".format(iterN, m2_target))
    #plt.close()
    plt.show()
    return 0

def get_m2Xieff_at_m1_slice_plot(medianlist_m2, medianlist_xieff, m1_src_grid, m1_target, M2, XIEFF, KDElist, VTinterp,  iterN=1, pathplot='./', plot_name='KDE'):
    m1_idx = 0 #np.argmin(np.abs(m1_src_grid - m1_target))
    new_2Dlists = []
    for kde in KDElist:
        new_2Dlists.append(kde[m1_idx, : , :])
        #KDEaverage = np.percentile(KDElist, 50, axis=0)
        #KDE_slice = KDEaverage[:, m2_idx, :] 
    data_slice = np.percentile(new_2Dlists, 50, axis=0)
    if plot_name=='Rate':
        data_slice = 69*data_slice/VTinterp #69 is numbe of observed BBH signals         
        colorbar_label = r'$d \mathcal{R}/dm_1 dm_2 d \chi_\mathrm{eff}[\mathrm{Gpc}^{-3} \mathrm{yr}^{-1} M_\odot^{-2}] $'
    else:
        colorbar_label = r'$p(m_2, \chi_\mathrm{eff})$'
    max_density = np.nanmax(data_slice)
    max_exp = np.floor(np.log10(max_density))  # Find the highest power of 10 below max_density
    contourlevels = 10 **(max_exp - np.arange(4))[::-1]
    vmin, vmax = contourlevels[0] , contourlevels[-1]# np.nanmax(KDE_slice)  # Min and max values for KDE
    print("vmin, vmax is =", vmin, vmax)
    # Plot
    plt.figure(figsize=(8, 6))
    norm_val = LogNorm(vmin=vmin, vmax=max_density)  # Apply log normalization
    #pcm = plt.pcolormesh(M2, XIEFF, data_slice, cmap='viridis', norm=norm, shading='auto')
    pcm = plt.pcolormesh(M2, XIEFF, data_slice, cmap='Purples', norm=LogNorm(vmin=vmin, vmax= max_density), shading='auto')
    contours = plt.contour(M2,  XIEFF, data_slice, levels=contourlevels, colors='black', linewidths=1.5)
    #plt.clabel(contours, fmt="% .1e", colors='black', fontsize=14)

    # Colorbar
    cbar = plt.colorbar(pcm, label=colorbar_label)
    plt.scatter(medianlist_m2, medianlist_xieff, color='r', marker='+', s=20)
    plt.ylabel(r"$\chi_\mathrm{eff}$")
    plt.xlabel(r"$m_\mathrm{2} \,[M_\odot]$")
    plt.semilogx()
    plt.title(r"$m_1 = {0}$".format(m1_target), fontsize=18)
    plt.tight_layout()
    #plt.savefig(pathplot+"Average"+plot_name+"m2Xieffatm1_{1}_Iter{0}.png".format(iterN, m1_target))
    #plt.close()
    plt.show()
    return 0

def get_m1m2_at_xieff_slice_plot(medianlist_m1, medianlist_m2, xi_src_grid, xi_target, M1, M2, KDElist, VTinterp,  iterN=1, pathplot='./', plot_name='KDE'):
    xi_idx = np.argmin(np.abs(xi_src_grid - xi_target))
    new_2Dlists = []
    for kde in KDElist:
        new_2Dlists.append(kde[:, : , xi_idx])
        #KDEaverage = np.percentile(KDElist, 50, axis=0)
        #KDE_slice = KDEaverage[:, m2_idx, :] 
    data_slice = np.percentile(new_2Dlists, 50, axis=0)
    if plot_name=='Rate':
        data_slice = 69*data_slice/VTinterp #69 is numbe of observed BBH signals         
        colorbar_label = r'$d \mathcal{R}/dm_1 dm_2 d\chi_\mathrm{eff}[\mathrm{Gpc}^{-3} \mathrm{yr}^{-1} M_\odot^{-2}] $'
    else:
        colorbar_label = r'$p(m_1, m_2)$'
    max_density = np.nanmax(data_slice)
    max_exp = np.floor(np.log10(max_density))  # Find the highest power of 10 below max_density
    contourlevels = 10 ** (max_exp - np.arange(4))[::-1]
    contourlevels[-1] = max_density
    vmin, vmax = contourlevels[0] , contourlevels[-1]# np.nanmax(KDE_slice)  # Min and max values for KDE
    print("vmin, vmax is =", vmin, vmax)
    # Plot
    plt.figure(figsize=(8, 6))
    norm1 = LogNorm(vmin=vmin, vmax=vmax)  # Apply log normalization
    #pcm = plt.pcolormesh(M1, M2, data_slice, cmap='viridis', norm=norm, shading='auto')
    pcm = plt.pcolormesh(M1, M2, data_slice, cmap='Purples', norm=norm1, shading='auto')
    contours = plt.contour(M1,  M2, data_slice, levels=contourlevels, colors='black', linewidths=1.5)
    #plt.clabel(contours, fmt="% .1e", colors='black', fontsize=14)

    # Colorbar
    cbar = plt.colorbar(pcm, label=colorbar_label)
    #cbar.set_ticks(contourlevels[:-1])
    plt.fill_between(np.arange(0, 105), np.arange(0, 105), 105 , color='white',alpha=1,zorder=50)
    #plt.fill_between(np.arange(0, 50), np.arange(0, 50), 50 , color='white',alpha=1,zorder=100)
    plt.scatter(medianlist_m1, medianlist_m2, color='r', marker='+', s=20)
    plt.xlabel(r"$m_\mathrm{1} \,[M_\odot]$")
    plt.ylabel(r"$m_\mathrm{2} \,[M_\odot]$")
    plt.tick_params(axis='y', which='both', left=False, right=True, labelleft=False, labelright=True)
    plt.semilogx()
    plt.semilogy()
    plt.xlim(4, 102)
    plt.ylim(4, 102)
    plt.title(r"$\chi_\mathrm{{eff}} = {0}$".format(xi_target), fontsize=18)
    plt.tight_layout()
    #plt.savefig(pathplot+"Average"+plot_name+"m1m2atXieff_{1}_Iter{0}.png".format(iterN, xi_target))
    #plt.close()
    plt.show()
    return 0


>>>>>>> 629fe549e028ed7a2df81328618ce8d857c9db88


###################################
def plot_pdetscatter(flat_samples1, flat_samples2, flat_pdetlist, xlabel=r'$m_{1, source} [M_\odot]$', ylabel=r'$d_L [Mpc]$', title=r'$p_\mathrm{det}\, \,\, q^{1.26}$',save_name="pdet_power_law_m2_correct_mass_frame_m1_dL_scatter.png", pathplot='./', show_plot=False):
    flat_pdetlist = flat_pdetlist/1e9
    plt.figure(figsize=(8,6))
    plt.scatter(flat_samples1, flat_samples2, c=flat_pdetlist, s=10 ,cmap='viridis', norm=LogNorm(vmin=min(flat_pdetlist), vmax=max(flat_pdetlist)))
    cbar = plt.colorbar(label=r'$p_\mathrm{det}$')
    #cbar.set_label(r'$p_\mathrm{det}$', fontsize=20)
<<<<<<< HEAD
    cbar.set_label(r'$\mathrm{VT}/\mathrm{Gpc}^3 yr$', fontsize=20)
=======
    cbar.set_label(r'$\mathrm{VT} [\mathrm{Gpc}^3-\mathrm{yrs}]$', fontsize=20)
>>>>>>> 629fe549e028ed7a2df81328618ce8d857c9db88
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.semilogx()
    #plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(pathplot+save_name)
    if show_plot ==True:
        plt.show()
    plt.close()
    return 0


def plotpdet_3Dm1m2dLscatter(flat_samples1, flat_samples2, flat_samples3, flat_pdetlist, save_name="pdet_m1m2dL_3Dscatter.png", pathplot='./', show_plot=False): 
    from mpl_toolkits.mplot3d import Axes3D
    # 3D scatter plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the data with logarithmic color scaling
    sc = ax.scatter(flat_samples1, flat_samples2, flat_samples3, c=flat_pdetlist, cmap='viridis', s=10, norm=LogNorm(vmin=1e-5, vmax=1))
    plt.colorbar(sc, label=r'$p_\mathrm{det}(m_1, m_2, d_L)$')

    # Set axis labels and limits
    ax.set_xlabel(r'$m_{1, source} [M_\odot]$', fontsize=20)
    ax.set_ylabel(r'$m_{2, source} [M_\odot]$', fontsize=20)
    ax.set_zlabel(r'$d_L [Mpc]$', fontsize=20)
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(pathplot+save_name)
    if show_plot ==True:
        plt.show()

    plt.close()
    return 0

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
    if show_plot ==True:
        plt.show()
    plt.close()
    return 0


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
    if show_plot==True:
        plt.show()
    plt.close()
    return 0



############################# Resultsplot #####################
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
    p = plt.pcolormesh(XX, YY, ZZ, cmap=plt.cm.get_cmap('Purples'),  norm=LogNorm(vmin=contourlevels[0], vmax= contourlevels[-1]))#vmin=1e-5))
    CS = plt.contour(XX, YY, ZZ, colors='black', linestyles='dashed', linewidths=2, norm=LogNorm(), levels= contourlevels)
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
    if saveplot==True:
        plt.savefig(pathplot+plot_label+title+'.png')
    else:
        print("notsaving")
    if show_plot== True:
        plt.show()
    else:
        plt.close()
    return 0



def histogram_bwdata(datalist, dataname='bw',  pathplot='./', Iternumber=1):
    plt.figure(figsize=(8,6))
    plt.xlabel(dataname, fontsize=15)
    plt.hist(datalist, bins=20, color='red', fc='gray', histtype='step', alpha=0.8, density=True, label=dataname)
    plt.xlim(0.01, 1)
    plt.legend()
    plt.title("Iter{0}".format(Iternumber))
    plt.savefig(pathplot+dataname+"_histogramIter{0}.png".format(Iternumber), bbox_inches='tight')
    plt.close()
    return 0

def histogram_datalist(datalist, dataname='bw',  pathplot='./', Iternumber=1):
    """
    inputs: originalmean, shiftedmean, Iternumber, pathplot
    histogram to see shifted data alaonf with original data
    """
    plt.figure(figsize=(8,6))
    #plt.hist(datalist, bins=np.logspace(-2, 0, 25), color='red', fc='gray', histtype='step', alpha=0.8, density=True, label=dataname)
    plt.xlabel(dataname, fontsize=15)
    if dataname =='bw':
        plt.hist(datalist, bins=np.logspace(-1.5, -0.1, 15), color='red', fc='gray', histtype='step', alpha=0.8, density=True, label=dataname)
        minX, maxX = min(np.logspace(-2, 0, 25)) , max(np.logspace(-2, 0, 25))
        plt.xlim(0.01, 1.0)
        plt.semilogx()
    elif dataname =='alpha':
        plt.hist(datalist, bins=np.linspace(0, 1, 11), color='red', fc='gray', histtype='step', alpha=0.8, density=True, label=dataname)
        plt.xlim(0, 1)
    else:
        plt.hist(datalist, bins=np.logspace(np.log10(min(datalist)), 0, 15), color='red', fc='gray', histtype='step', alpha=0.8, density=True, label=dataname)
        plt.xlim(min(datalist)-0.1, max(datalist)+0.1)

    plt.legend()
    plt.title("Iter{0}".format(Iternumber))
    plt.savefig(pathplot+dataname+"_histogramIter{0}.png".format(Iternumber), bbox_inches='tight')
    plt.close()
    return 0


def average2Dkde_m1m2_plot(m1vals, m2vals, XX, YY, kdelists, pathplot='./', titlename=1, plot_label='Rate', x_label='m1', y_label='m2', plottag='Average', dLval=500, correct_units=False):
    if correct_units==True:
        volume_factor = get_dVdz_factor(dLval) #one value
    else:
        volume_factor = 1.0
    sample1, sample2 = m1vals, m2vals
<<<<<<< HEAD
    CI50 = np.percentile(kdelists, 50, axis=0)/volume_factor
    max_density = np.max(CI50)
    max_exp = np.floor(np.log10(max_density))  # Find the highest power of 10 below max_density
    contourlevels = 10 ** (max_exp - np.arange(3))[::-1]
=======
    CI50 = np.percentile(kdelists, 50, axis=0)#/volume_factor
    max_density = np.max(CI50)
    max_exp = np.floor(np.log10(max_density))  # Find the highest power of 10 below max_density
    contourlevels = 10 ** (max_exp - np.arange(4))[::-1]
>>>>>>> 629fe549e028ed7a2df81328618ce8d857c9db88
    fig, axl = plt.subplots(1,1,figsize=(8,6))
    p = axl.pcolormesh(XX, YY, CI50, cmap=plt.cm.get_cmap('Purples'), norm=LogNorm(vmin=contourlevels[0], vmax=contourlevels[-1]))
    cbar = plt.colorbar(p, ax= axl)
    if plot_label =='Rate':
        if correct_units==True:
            cbar.set_label(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}m_2\mathrm{d}dV_c [\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-2}]$',fontsize=18)
        else:
<<<<<<< HEAD
            cbar.set_label(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}m_2\mathrm{d}dV_c [\mathrm{Mpc}^{-1}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-2}]$',fontsize=18)
    else:
        cbar.set_label(r'$p(m_{1, source}, d_L)$',fontsize=18)
    CS = axl.contour(XX, YY, CI50, colors='black', levels=contourlevels ,linestyles='dashed', linewidths=2, norm=LogNorm(vmin=contourlevels[0], vmax=contourlevels[-1]))
=======
            #cbar.set_label(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}m_2\mathrm{d}dV_c [\mathrm{Mpc}^{-1}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-2}]$',fontsize=18)
            cbar.set_label(r'$\mathrm{d}^2\mathcal{R}/\mathrm{d}m_1\mathrm{d}m_2[\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-2}]$',fontsize=18)
    else:
        cbar.set_label(r'$p(m_{1, source}, d_L)$',fontsize=18)
    CS = axl.contour(XX, YY, CI50, colors='black', levels=contourlevels ,linestyles='dashed', linewidths=2, norm=LogNorm(vmin=contourlevels[0], vmax=max_density))
>>>>>>> 629fe549e028ed7a2df81328618ce8d857c9db88
    axl.scatter(sample1, sample2,  marker="+", color="r", s=20)
    axl.scatter(sample2, sample1,  marker="+", color="r", s=20)
    axl.fill_between(np.arange(0, 100), np.arange(0, 100),100 , color='white',alpha=1,zorder=50)
    axl.fill_between(np.arange(0, 50), np.arange(0, 50), 50 , color='white',alpha=1,zorder=100)
<<<<<<< HEAD
    axl.set_ylim(5, 101)
=======
    axl.set_ylim(4, 101)
>>>>>>> 629fe549e028ed7a2df81328618ce8d857c9db88
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
<<<<<<< HEAD
    axl.set_xlim(5, 100.1)
    axl.loglog()
    axl.set_aspect('equal')
    axl.set_title(r'$d_L=${0}[Mpc]'.format(dLval), fontsize=18)
    fig.tight_layout()
    plt.savefig(pathplot+plottag+'m1_'+y_label+'_2D'+plot_label+'Iter{0}dL{1:.3f}.png'.format(titlename, dLval), bbox_inches='tight')
=======
    axl.set_xlim(4, 100.1)
    axl.loglog()
    axl.set_aspect('equal')
    #axl.set_title(r'$d_L=${0}[Mpc]'.format(dLval), fontsize=18)
    fig.tight_layout()
    plt.show()
    #plt.savefig(pathplot+plottag+'m1_'+y_label+'_2D'+plot_label+'Iter{0}dL{1:.3f}.png'.format(titlename, dLval), bbox_inches='tight')
>>>>>>> 629fe549e028ed7a2df81328618ce8d857c9db88
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
    if show_plot== True:
        plt.show()
    else:
        plt.close()
    return CI50


def bandwidth_correlation(bwlist,  number_corr=100, error=0.02, param='bw', pathplot='./', log=True):
    """
    change the number for when we want before and after 
    Make a scatter plot with some random 
    error added to bwd 
    scatter point bw[i]  vs bw[i+1]
    and also plot correlation coefficient 
    """
#    listN = np.array(bwlist[:-1]) #from 0 to N-1 values of array
#    listNp1 = np.array(bwlist[1:]) #1 to Nth (last value of array)
#
#    listN = np.random.lognormal(np.log(listN), error)
#    listNp1 = np.random.lognormal(np.log(listNp1), error)
#    plt.figure(figsize=(8,5))
#    plt.scatter(listN , listNp1, marker= '+')
#    plt.xlabel(param+" N")
#    plt.ylabel(param+" N+1")
#    plt.semilogy()
#    plt.semilogx()
#    if error > 0.0:
#        plt.savefig(pathplot+param+"_correlation_scatter.png")
#    else:
#        plt.savefig(pathplot+param+"_correlation.png")
#    plt.close()
#
    #new correlation
    Cxy  =  []
    iternumber = []
    for i in range(int(len(bwlist)/2)):
        iternumber.append(i+1)
        if log==False:
            M = np.corrcoef(bwlist[i+1: ], bwlist[ :-i-1])
        else:
            M = np.corrcoef( np.log(bwlist[i+1: ]), np.log(bwlist[ :-i-1]))
        Cxy.append(M[0][1])
    print([Cxy[i] for i in range(10)])
    plt.figure(figsize=(8,4))
    plt.plot(iternumber[number_corr:], Cxy[number_corr:],'+-')
    plt.xlabel('Number of iterations')
    plt.ylabel(r'$C_{01}$', fontsize=14)
    plt.semilogx()
    plt.grid(True)
    plt.savefig(pathplot+param+"C01_iter_after_100.png", bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(8, 4))
    plt.plot(iternumber[:number_corr], Cxy[:number_corr],'+-')
    plt.xlabel('Number of iterations')
    plt.ylabel(r'$C_{01}$')
    #plt.semilogx()
    plt.grid(True)
    plt.savefig(pathplot+param+"C01_iter_before_100.png", bbox_inches='tight')
    plt.close()


    #corrcoefficient
    Cxy  =  []#[1.0]
    #for i in range(number_corr, int(len(bwlist)/2)):
    for i in range(int(len(bwlist)/2)):
        if log == True:
            M = np.corrcoef( np.log(bwlist[i+1: ]), np.log(bwlist[ :-i-1]))
        else:
            M = np.corrcoef(bwlist[i+1: ], bwlist[ :-i-1])
        Cxy.append(M[0][1])
    #print([Cxy[i] for i in range(10)])
    plt.figure(figsize=(8, 4))
    plt.plot(Cxy,'+-')
    plt.xlabel('Number of iterations')
    plt.xlim(xmin=1)
    plt.ylabel(r'$C_{xy}$')
    plt.semilogx()
    plt.grid(True)
    plt.savefig(pathplot+param+"CxyiterN.png", bbox_inches='tight')
    plt.close()
    return 0





def corre_tom(series, before_use_buf=5, bufsize=100, quantity='bandwidths', log=True, pathplot='./'):
    """
    series is array of values of bw of alpha
    chnage quantity with alpha for alpha correlation
    """
    #if quantity != 'bandwidths':
    #    log = False
    series_before = series[:before_use_buf]
    series_after = series[before_use_buf:]
    plt.plot(series_before, '+')
    #plt.plot(np.convolve(series_before, np.ones(5)/5., mode='valid'), '-')
    plt.plot(np.convolve(series_before, np.ones(10)/10., mode='valid'), '-.')
    if log:
        plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel(quantity.rstrip('s'))
    plt.savefig(pathplot+'before_'+quantity+'_series.png', bbox_inches='tight')
    plt.close()

    # autocorrelation
    if log:
        series_before = np.log(series_before)
        series_after = np.log(series_after)
    
    # pre-buffer
    acorr_before = []
    #iternumber = []
    for i in range(int(2. * len(series_before) / 3.)):  # lag between samples
        #iternumber.append(i+1)
        # correlate the series with i samples removed from the beginning and end resp.
        M = np.corrcoef(series_before[:-i-1], series_before[i+1:])
        acorr_before.append(M[0][1])
        #print([Cxy[i] for i in range(10)])
    plt.figure()
    plt.plot(np.arange(len(acorr_before))[1:], acorr_before[1:], '+-')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation of '+quantity)
    plt.semilogx()
    plt.grid()
    plt.savefig(pathplot+'before_'+quantity+'_autocorr.png', bbox_inches='tight')
    plt.close()
    
    # using buffer
    acorr_after = []
    #iternumber = []
    for i in range(int(2. * len(series_after) / 3.)):  # lag between samples
        #iternumber.append(i+1)
        # correlate the series with i samples removed from the beginning and end resp.
        M = np.corrcoef(series_after[:-i-1], series_after[i+1:])
        acorr_after.append(M[0][1])
        #print([Cxy[i] for i in range(10)])
    plt.figure()
    plt.plot(np.arange(len(acorr_after))[1:], acorr_after[1:], '+')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation of '+quantity)
    #plt.semilogx()
    plt.grid()
    plt.savefig(pathplot+'after_'+quantity+'_autocorr.png', bbox_inches='tight')
    plt.close()
    
