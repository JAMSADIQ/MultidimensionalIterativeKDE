import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import h5py as h5
import scipy
from scipy.interpolate import RegularGridInterpolator
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



from matplotlib import rcParams
from matplotlib.colors import LogNorm, Normalize
import matplotlib.colors as colors
pathplot = './'
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
rcParams["grid.linestyle"] = ':'
rcParams["grid.alpha"] = 0.6




##############convert to dR(z)/dVc  in 1/(Gpc^3 yr M^2)
from scipy.integrate import quad
def hubble_parameter(z, H0, Omega_m):
    """
    Calculate the Hubble parameter H(z) for a flat Lambda-CDM cosmology.
    """
    Omega_Lambda = 1.0 - Omega_m
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)

def comoving_distance(z, H0, Omega_m):
    """
    Compute the comoving distance D_c(z) in Mpc.
    """
    integrand = lambda z_prime: c / hubble_parameter(z_prime, H0, Omega_m)
    D_c, _ = quad(integrand, 0, z)
    return D_c

def comoving_distance_derivative(z, H0, Omega_m):
    """
    Compute the derivative of comoving distance dD_c/dz in Mpc.
    """
    return c / hubble_parameter(z, H0, Omega_m)
#cosmology parameter we are using are
H0 = 67.9 #km/sMpc
c = 3e5 #km/s
omega_m = 0.3065 #matter density
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value, Planck15
cosmo =  FlatLambdaCDM(H0, omega_m)

def get_ddL_bydz_factor(dL_Mpc):
    z_at_dL = z_at_value(cosmo.luminosity_distance, dL_Mpc*u.Mpc).value
    D_c = comoving_distance(z_at_dL, H0, omega_m)
    dD_c_dz = comoving_distance_derivative(z_at_dL, H0, omega_m)
    ddL_dz = D_c + (1 + z_at_dL) * dD_c_dz
    return  z_at_dL, ddL_dz

def get_dVdz_factor(dL_Mpc):
    z_at_dL, ddL_dz = get_ddL_bydz_factor(dL_Mpc)
    dV_dMpc_cube = 4.0 * np.pi * cosmo.differential_comoving_volume(z_at_dL)/ ddL_dz
    dV_dzGpc3 = dV_dMpc_cube.to(u.Gpc**3 / u.sr).value
    return dV_dzGpc3

def average2Dkde_plot(m1vals, m2vals, XX, YY, kdelists, pathplot='./', titlename=1, plot_label='Rate', x_label='m1', y_label='m2', plottag='Average', dLval=500):
    volume_factor = get_dVdz_factor(dLval) #one value
    z_val = z_at_value(cosmo.luminosity_distance, dLval*u.Mpc).value
    sample1, sample2 = m1vals, m2vals
    CI50 = np.percentile(kdelists, 50, axis=0)/volume_factor
    max_density = np.max(CI50)
    max_exp = np.floor(np.log10(max_density))  # Find the highest power of 10 below max_density
    contourlevels = 10 ** (max_exp - np.arange(4))[::-1]
    fig, axl = plt.subplots(1,1,figsize=(8,6))
    p = axl.pcolormesh(XX, YY, CI50, cmap=plt.cm.get_cmap('Purples'), norm=LogNorm(vmin=contourlevels[0], vmax=contourlevels[-1]))
    cbar = plt.colorbar(p, ax= axl)
    cbar.set_label(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}m_2\mathrm{d}dV_c [\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-2}]$',fontsize=18)
    CS = axl.contour(XX, YY, CI50, colors='black', levels=contourlevels ,linestyles='dashed', linewidths=2, norm=LogNorm(vmin=contourlevels[0], vmax=contourlevels[-1]))
    axl.scatter(sample1, sample2,  marker="+", color="r", s=20)
    axl.scatter(sample2, sample1,  marker="+", color="r", s=20)
    axl.fill_between(np.arange(0, 100), np.arange(0, 100),100 , color='white',alpha=1,zorder=50)
    axl.fill_between(np.arange(0, 50), np.arange(0, 50), 50 , color='white',alpha=1,zorder=100)
    axl.set_ylim(3, 101)
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
    axl.set_xlim(5, 105)
    axl.set_ylim(5, 100)
    axl.loglog()
    axl.set_aspect('equal')
    #axl.set_title(r'$d_L={0:.1f}\,$ [Mpc] ,$\,\,z={1:.1f}'.format(dLval, z_val))
    axl.set_title(r'$d_L={0:.1f}\,\mathrm{{Mpc}},\,\,z={1:.1f}$'.format(dLval, z_val), fontsize=20)
    fig.tight_layout()
    #plt.show()

    plt.savefig('m1_m2_dL3Danalysis_Xieffbased_max_bw_03_combined_all_m1_m2_2DKDEIter1001dL{0}.png'.format(dLval), bbox_inches='tight')
    plt.show()

    return CI50
    

#######################################################

from scipy import integrate
def get_rate_m1(maskm1m2, rate_m1_m2, m2grid, volume_factor):
    rate_m1_m2_masked = np.where(maskm1m2, rate_m1_m2, 0)
    rate_m1 = integrate.simpson(rate_m1_m2_masked, x=m2grid, axis=1)
    return rate_m1 /volume_factor

def IntegrateRm1m2_wrt_m2(m1val, m2val, Ratem1m2):
    ratem1 = np.zeros(len(m1val))
    ratem2 = np.zeros(len(m2val))
    xval = 1.0 *m1val
    yval = 1.0 *m2val
    kde = Ratem1m2
    for xid, m1 in enumerate(m1val):
        y_valid = yval <= xval[xid]  # Only accept points with y <= x
        y_q1 = np.argmin(abs(xval[xid] - yval))  # closest y point to y=x
        rate_vals = kde[y_valid, xid]
        ratem1[xid] = integrate.simpson(rate_vals, x=yval[y_valid])
    for yid, m2 in enumerate(m2val):
        x_valid = xval >= yval[yid]  # Only accept points with y <= x
        rate_vals = kde[x_valid, yid]
        ratem2[yid] = integrate.simpson(rate_vals,x= xval[x_valid])
    return ratem1


#######
fz = h5.File('../Final_noncosmo_GWTC3_redshift_datafile.h5', 'r')
dz = fz['randdata']
f1 = h5.File('../Final_noncosmo_GWTC3_m1srcdatafile.h5', 'r')#m1
d1 = f1['randdata']
f2 = h5.File('../Final_noncosmo_GWTC3_m2srcdatafile.h5', 'r')#m2
d2 = f2['randdata']
f3 = h5.File('../Final_noncosmo_GWTC3_dL_datafile.h5', 'r')#dL
d3 = f3['randdata']
#print(d1.keys())
sampleslists1 = []
medianlist1 = []
eventlist = []
sampleslists2 = []
medianlist2 = []
sampleslists3 = []
medianlist3 = []
pdetlists = []
for k in d1.keys():
    eventlist.append(k)
    m1_values = d1[k][...]
    m2_values = d2[k][...]
    d_Lvalues = d3[k][...]
    sampleslists1.append(m1_values)
    sampleslists2.append(m2_values)
    sampleslists3.append(d_Lvalues)
    medianlist1.append(np.percentile(m1_values, 50))
    medianlist2.append(np.percentile(m2_values, 50))
    medianlist3.append(np.percentile(d_Lvalues, 50))

f1.close()
f2.close()
f3.close()
fz.close()
meanxi1= np.array(medianlist1)
meanxi2 = np.array(medianlist2)
meanxi3 = np.array(medianlist3)




###########
for dLval in [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000]:#, 3500, 4000, 4500, 5000]:
    print("dL = ", dLval)
    p3grid = np.array([dLval])
    savehf  =  h5.File('Correctedpdet_KDEsdata_fromOptimizeData_dL{0}.hdf5'.format(dLval), 'r')
    PDET_slice  = savehf['PDET2D'][...]
    M1_dLslice = savehf['xx2d'][...]
    M2_dLslice = savehf['yy2d'][...]
    PDETfilter = savehf['PDET2Dfiltered01'][...]
    m1m2mask =  M2_dLslice > M1_dLslice #use zero in all indices of rates for which m2 > m1
    kde_list = []
    rate_list = []
    for i in range(1100):#fix this as this can change
        KDE_slice = savehf["kde_iter{0}".format(i)][...]
        current_rateval = 69*KDE_slice/PDET_slice
        kde_list.append(KDE_slice)
        rate_list.append(current_rateval)
    savehf.close()
    kde_array = np.array(kde_list)  #Shape:(num_iter, num_eval_pts)
    rate_array = np.array(rate_list)
    #average2Dkde_plot(meanxi1, meanxi2, M1_dLslice, M2_dLslice, kde_list[100:], pathplot=pathplot, titlename=1001, plot_label='KDE', x_label='m1', y_label='m2', plottag='Combined_all_', dLval=dLval)
    average2Dkde_plot(meanxi1, meanxi2, M1_dLslice, M2_dLslice, rate_list[100:], pathplot=pathplot, titlename=1001, plot_label='Rate', x_label='m1', y_label='m2', plottag='Combined_all_', dLval=dLval)

quit()
