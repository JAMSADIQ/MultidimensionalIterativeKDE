#Jam Sadiq
# A script for making plots for the paper

#import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LogNorm
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

def normdata(dataV):
    normalized_data = (dataV - np.min(dataV)) / (np.max(dataV) - np.min(dataV))
    return normalized_data


dict_p = {'m1':'m_1', 'm2':'m_2', 'Xieff':'\chi_{eff}', 'chieff': '\chi_{eff}', 'DL':'D_L', 'logm1':'ln m_1', 'logm2': 'ln m_2', 'alpha':'\alpha'}
###########
def TwocontourKDE(XX, YY,  ZZ, LogMzvals, zvals, pdetvals, title='KDE', iterN=0, saveplot=False):
    contourlevels = np.logspace(-7, 0, 10)
    plt.figure(figsize=(8, 6))
    contourlevels = np.logspace(-7, 3, 10)[3:]
    # Plotting pcolormesh and contour
    #p = plt.pcolormesh(XX, YY, ZZ, cmap=plt.cm.get_cmap('Purples'),  norm=LogNorm(vmin=1e-5))
    CS = plt.contour(XX, YY, ZZ, colors='black', linestyles='dashed', linewidths=2, norm=LogNorm(), levels= np.logspace(np.log10(np.min(ZZ)), np.log10(np.max(ZZ)), 10))
    p = plt.scatter(LogMzvals, zvals, c=pdetvals, s=10, alpha=1.0, cmap=plt.cm.get_cmap('viridis'), norm=LogNorm())#vmin=1e-3))
    # Colorbar
    cbar = plt.colorbar(p)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(title, fontsize=18)  # Adjust the label text as neede
    # Axes and labels
    plt.tick_params(labelsize=15)
    plt.xlabel(r'$\log_{10}[Mz]$', fontsize=20)

    # Y-axis formatting
    scale_y = 1  # Adjust if needed, e.g., 1e3 for scaling
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / scale_y))
    plt.gca().yaxis.set_major_formatter(ticks_y)
    plt.ylabel(r"$z$", fontsize=20)

    # Title and layout
    plt.tight_layout()
    if saveplot ==True:
        plt.savefig("KDEIter{0}.png".format(iterN))
        plt.close()
    else:
        print("dont save")
        plt.show()
    return 0


def compare_twodimensionalKDEPlot(XX, YY, ZZ, ZZ2, title1='instrinsic KDE', title2='PE KDE', plot_name='compareKDE'):
    contourlevels = np.logspace(-4, -1, 7)
    #fig, ax = plt.subplots()
    fig, (ax, ax1) = plt.subplots(2, 1, figsize=(7, 10), sharex=True)
    p = ax.pcolormesh(XX, YY, ZZ, cmap=plt.cm.get_cmap('Purples'), norm=LogNorm(vmin=1e-4))
    CS = ax.contour(XX, YY, ZZ,levels = contourlevels , colors='black', linestyles='dashed', linewidths=1, norm=LogNorm())
    #axl.scatter(datam1, dataD marker=".", color="k", s=40)
    cbar = plt.colorbar(p, ax=ax)
    cbar.ax.tick_params(labelsize=20)
    ax.tick_params(labelsize=15)
    ax.set_xlabel(r'$Log10[Mz]$')
    scale_y = 1#e3
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
    ax.yaxis.set_major_formatter(ticks_y)
    ax.set_ylabel(r"$z$", fontsize=20)
    ax.set_ylim(ymax=20)
    ax.set_title("intrinsic-KDE", fontsize=16)
    p1 = ax1.pcolormesh(XX, YY, ZZ2, cmap=plt.cm.get_cmap('Purples'), norm=LogNorm(vmin=1e-4))
    ax1.contour(XX, YY, ZZ2,levels = contourlevels , colors='black', linestyles='dashed', linewidths=1, norm=LogNorm())
    cbar2 = plt.colorbar(p1, ax=ax1)
    #cbar2.ax1.tick_params(labelsize=20)
    ax1.tick_params(labelsize=15)

    ax1.set_xlabel(r'$Log10[Mz]$')
    ax1.set_ylim(ymax=20)
    #fig.suptitle("Compare KDEs", fontsize=16)
    ax1.set_title("detected-KDE", fontsize=16)
    fig.tight_layout()
    plt.savefig(pathplot+plot_name+'.png')
    plt.show()
    return 0



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

def average2Dkde_plot(m1vals, m2vals, XX, YY, kdelists, pathplot='./', titlename=1, plot_label='Rate', x_label='m1', y_label='dL', show_plot=False):
    sample1, sample2 = m1vals, m2vals
    CI50 = np.percentile(kdelists, 50, axis=0) 
    fig, axl = plt.subplots(1,1,figsize=(8,6))
    max_density = np.max(CI50)
    max_exp = np.floor(np.log10(max_density))  # Find the highest power of 10 below max_density
    contourlevels = 10 ** (max_exp - np.arange(4))[::-1]
    #if plot_label=='Rate':
    #    contourlevels = np.logspace(-3, 3, 7)#np.array([1e-4, 3e-4,1e-3, 3e-3, 1e-2, 3e-2, 6e-2, 1e-1, 3e-1, 6e-1, 1, 10, 18])
    #else:
    #    contourlevels =np.logspace(-5, 0, 6) 
    p = axl.pcolormesh(XX, YY, CI50, cmap=plt.cm.get_cmap('Purples'), norm=LogNorm(vmin=contourlevels[0], vmax =contourlevels[-1]),  label=r'$p(m_1, d_L)$')
    cbar = plt.colorbar(p, ax= axl)
    if plot_label =='Rate':
        cbar.set_label(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}d_L [\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-1}]$',fontsize=18)
    else:
        cbar.set_label(r'$p(m_{1, source}, d_L)$',fontsize=18)
    CS = axl.contour(XX, YY, CI50, colors='black', levels=contourlevels ,linestyles='dashed', linewidths=2, norm=LogNorm())
    axl.scatter(sample1, sample2,  marker="+", color="r", s=20, label='medianval')
    axl.set_ylabel(r'$d_L\,[Mpc]$', fontsize=18)
    axl.set_xlabel(r'$m_\mathrm{1, source} \,[M_\odot]$', fontsize=18)
    axl.legend()
    axl.loglog()
    #axl.set_aspect('equal')
    fig.tight_layout()
    plt.savefig(pathplot+"average_"+plot_label+"{0}.png".format(titlename))
    if show_plot== True:
        plt.show()
    else:
        plt.close()
    return CI50

def average2DlineardLrate_plot(m1vals, m2vals, XX, YY, kdelists, pathplot='./', titlename=1, plot_label='Rate', x_label='m1', y_label='dL', show_plot=False):
    sample1, sample2 = m1vals, m2vals
    CI50 = np.percentile(kdelists, 50, axis=0) 
    fig, axl = plt.subplots(1,1,figsize=(8,6))
    max_density = np.max(CI50)
    max_exp = np.floor(np.log10(max_density))  # Find the highest power of 10 below max_density
    contourlevels = 10 ** (max_exp - np.arange(4))[::-1]
    #if plot_label=='Rate':
    #    contourlevels = np.logspace(-3, 3, 7)#np.array([1e-4, 3e-4,1e-3, 3e-3, 1e-2, 3e-2, 6e-2, 1e-1, 3e-1, 6e-1, 1, 10, 18])
    #else:
    #    contourlevels =np.logspace(-5, 0, 6) 
    p = axl.pcolormesh(XX, YY, CI50, cmap=plt.cm.get_cmap('Purples'), norm=LogNorm(vmin=contourlevels[0], vmax =contourlevels[-1]),  label=r'$p(m_1, d_L)$')
    cbar = plt.colorbar(p, ax= axl)
    if plot_label =='Rate':
        #cbar.set_label(r'$\mathrm{d}\mathcal{R}/\mathrm{dln}m_1\mathrm{d}d_L [\mathrm{Mpc}^{-1}\, (\mathrm{ln m}/{M}_\odot)^{-1}  \mathrm{yr}^{-1}]$',fontsize=18)
        cbar.set_label(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}d_L [\mathrm{Mpc}^{-1}\, (\mathrm{m}/{M}_\odot)^{-1}  \mathrm{yr}^{-1}]$',fontsize=18)
    else:
        #cbar.set_label(r'$p(\mathrm{ln}m_{1, source}, d_L)$',fontsize=18)
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


#def average_allkde_plot(x_grid, kdelists, pathplot='./', plot_analytic=False):
def average_allkde_plot(x_grid, analyticfuncvalue, kdelists, pathplot='./', plot_analytic=False):
    meankde = np.zeros_like(x_grid)
    for kde in kdelists:
        meankde += kde/len(kdelists)
    #5th, 50th and 95th percentile kde
    CI5 = np.percentile(kdelists, 5, axis=0)
    CI50 = np.percentile(kdelists, 50, axis=0)
    CI95 = np.percentile(kdelists, 95, axis=0)
    #np.savetxt('Finall100to1000_iteration_average_rate_xgrid_median_5th_50th_95th_.txt', np.c_[x_grid, meankde, CI5, CI50, CI95])
    plt.figure(figsize=(8, 5))
    if plot_analytic==True:
        plt.plot(x_grid, analyticfuncvalue, ls='solid', color='cyan', label='analytic')
    plt.plot(x_grid, meankde, 'r', linewidth = 2. , label='median')
    plt.plot(x_grid, CI5, 'r--', linewidth = 2., label='5th')
    plt.plot(x_grid, CI95, 'r--', linewidth = 2.,  label='95th')
    plt.ylim(ymin = 3e-4)
    plt.semilogy()
    plt.xlabel('$m_1$ [$\mathrm{M}_\odot$]', fontsize=14)
    plt.ylabel('$\mathrm{d}\mathcal{R}/\mathrm{d}m_1[\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-1}]$', fontsize=14)
    plt.grid(ls=':',color='grey', alpha = 0.6)
    #plt.legend()
    #plt.title('combine_KDE')
    plt.savefig(pathplot+"combined_rate1D.png", bbox_inches='tight')
    #plt.savefig(pathplot+"combinedkde.png")
    plt.close()


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




import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def combine_twoD1Dplot(XX, YY, ZZ, sample1, sample2, x_gridvalues, analyticfuncvalue1, analyticfuncvalue2, originalkdevalues1, originalkdevalues2, pathplot='./'):
    # Plot the conditional distributions
    fig = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])
    # gs.update(wspace=0., hspace=0.)

    # Plot surface on top left
    ax1 = plt.subplot(gs[0])
#   Plot 2D
    contourlevels = np.logspace(-4, 1.2, 8)#np.array([1e-4, 3e-4,1e-3, 3e-3, 1e-2, 3e-2, 6e-2, 1e-1, 3e-1, 6e-1, 1, 10, 15, 20])
    p = ax1.pcolormesh(XX, YY, ZZ, cmap=plt.cm.get_cmap('Purples'), vmin=1e-5, norm=LogNorm())

    CS = ax1.contour(XX, YY, ZZ, colors='black',vmin=1e-5,linestyles='dashed', linewidths=2, norm=LogNorm())
    ax1.scatter(sample1, sample2,  marker="+", color="r", s=20)
    ax1.scatter(sample2, sample1,  marker="+", color="r", s=20)
    ax1.yaxis.set_label_position('right')
    sx = np.arange(3, 100, 1)
    ax1.fill_between(sx, sx, 100, color='white', alpha=1, zorder=50)
    ax1.set_ylim(2.3, 101)
    ax1.set_ylabel(r'$m_2\,[M_\odot]$', fontsize=15)
    ax1.set_xlabel(r'$m_1\,[M_\odot]$', fontsize=15)
    scale_y = 1#e3
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
    ax1.yaxis.set_major_formatter(ticks_y)
    ax1.set_xlim(3, 100.1)
    #axl.set_ylabel(r'$'+dict_p[y_label]+'$')
    fig.tight_layout()


    # Plot y
    ax2 = plt.subplot(gs[1])
    ax2.plot(analyticfuncvalue1, x_gridvalues,lw=2, ls='solid', color='cyan', label='True')

    ax2.plot(originalkdevalues1, x_gridvalues,lw=2, ls='dashed', color='b', label=r'$p(m_1)$')
    #ax2.plot(x_gridvalues, originalkdevalues1, lw=2, ls='dashed', color='b', label=r'$p(m_1)$')
    lgd = ax2.legend(ncol=2, fancybox=False)
    ax2.set_xlim(xmin=1e-4)
    ax2.semilogx()

    # Plot x
    ax3 = plt.subplot(gs[2])
    ax3.plot(x_gridvalues, analyticfuncvalue2, lw=2, ls='solid', color='cyan', label='True')

    ax3.plot(x_gridvalues, originalkdevalues2, lw=2, ls='dashed', color='b', label=r'$p(m_2)$')
    lgd = ax3.legend(ncol=2, fancybox=False)
    ax3.set_ylim(ymin=1e-4)
    ax3.semilogy()
    # Clear axis 4 and plot colarbar in its place
    ax4 = plt.subplot(gs[3])
    ax4.set_visible(False)
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('left', size='20%', pad=0.05)
    cbar = fig.colorbar(p, cax=cax)
    cbar.ax.set_ylabel('density: $p(m_1, m_2)$', fontsize=13)

    cbar.ax.tick_params(labelsize=20)
    ax4.tick_params(labelsize=15)
    plt.savefig(pathplot+'testplot.png', bbox_inches='tight')
    plt.close()


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
    
