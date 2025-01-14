import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import h5py as h5
import scipy
from scipy.signal import find_peaks, peak_prominences
from scipy.interpolate import RegularGridInterpolator
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
#import density_estimate as d
#import adaptive_kde as ad

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

H0 = 67.9 #km/sMpc
c = 299792.458#km/s
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
    dV_dMpc_cube = 4.0 * np.pi * cosmo.differential_comoving_volume(z_at_dL)/ (1 + z_at_dL) /ddL_dz
    dV_dzGpc3 = dV_dMpc_cube.to(u.Gpc**3 / u.sr).value
    return dV_dzGpc3
 

#######################################################
xmin, xmax, ymin, ymax, zmin, zmax, Npoints = 3, 100, 3, 100, 10, 5000, 200
p1grid = np.logspace(np.log10(xmin), np.log10(xmax), Npoints)
p2grid = np.logspace(np.log10(ymin), np.log10(ymax), Npoints)
from scipy import integrate

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
    return ratem1#, ratem2


def get_rate_m1(maskm1m2, rate_m1_m2, m2grid, volume_factor):
    rate_m1_m2_masked = np.where(maskm1m2, rate_m1_m2, 0)
    rate_m1 = integrate.simpson(rate_m1_m2_masked, x=m2grid, axis=1)
    return rate_m1 * volume_factor



fig, ax = plt.subplots(figsize=(10, 8))
colormap = plt.cm.magma
#colormap = plt.cm.hsv
norm = Normalize(vmin=300, vmax=5000)
percentile_offset = 100

def fixtoosmall(array):
    max_value = np.max(array)
    x = np.arange(len(array))
    threshold = max_value * 5e-3
    mask = array >= threshold
    filtered_x = x[mask]
    return filtered_x 

def get_massed_indetector_frame(dLMpc, mass):
    zcosmo = z_at_value(cosmo.luminosity_distance, dLMpc*u.Mpc).value
    mdet = mass*(1.0 + zcosmo)
    return mdet

colormap = plt.cm.magma#rainbow
dLarray = np.array([300, 600 ,900, 1200, 1500, 1800, 2100, 2400, 2700, 3000])#, 3000, 3500, 4000, 4500, 5000])
zarray = z_at_value(cosmo.luminosity_distance, dLarray*u.Mpc).value
norm = Normalize(vmin=zarray.min(), vmax=zarray.max())
y_offset = 0  # Starting offset for ridgelines
for ik, dLval in  enumerate(dLarray):#[300, 500, 800, 1200, 1500, 1800, 2100, 2500, 3000, 3500, 4000, 4500, 5000]):
    p3grid = np.array([dLval])
    savehf  =  h5.File('Correctedpdet_KDEsdata_fromOptimizeData_dL{0}.hdf5'.format(dLval), 'r')
    color = colormap(norm(zarray[ik]))
    volume_factor = get_dVdz_factor(dLval)
    M1_dLslice = savehf['xx2d'][:]
    p1grid = M1_dLslice[:, 0]
    p2grid = M1_dLslice[0:, 0]
    #print(p1grid.shape, min(p2grid), max(p2grid))
    M2_dLslice = savehf['yy2d'][:]
    PDET_slice = savehf['PDET2D'][:]
    #PDET_slice = savehf['PDET2Dfiltered01'][:]
    PDET_slice = np.maximum(PDET_slice, 0.1)
    print(np.min(PDET_slice))
    m1m2mask =  M2_dLslice < M1_dLslice
    kde_list = []
    rate_list = []
    rate1Dmedlist = []
    for i in range(100, 1100):#fix this as this can change
        KDE_slice =  savehf["kde_iter{0}".format(i)][:]
        current_rateval = 69*KDE_slice/PDET_slice #*len(meanxi1)
        rate_list.append(current_rateval)
        rateOneD = IntegrateRm1m2_wrt_m2(p1grid, p2grid,current_rateval)
        #print(len(rateOneD), np.array(rateOneD).shape)
        rate1Dmedlist.append(rateOneD)
    savehf.close()
    rate_m1 = np.median(rate1Dmedlist, axis=0)
    rate_m15 = np.percentile(rate1Dmedlist, 5.0, axis=0)
    rate_m195 = np.percentile(rate1Dmedlist, 95.0, axis=0)
    rate50 =  rate_m1/volume_factor
    rate05=  rate_m15/volume_factor
    rate95= rate_m195/volume_factor
    m1_values = p1grid.copy()
    ##########MASKING IF IT WORK
    mask = (rate05 >= 1e-3 * rate50) & (rate95 <= 5e3 * rate50)
    # Use masked arrays to filter the invalid regions
    m1_masked = np.ma.masked_where(~mask, m1_values)  # Masked x values
    p50_masked = np.ma.masked_where(~mask, rate50)
    p5_masked = np.ma.masked_where(~mask, rate05)
    p95_masked = np.ma.masked_where(~mask, rate95)

    ax.plot(m1_masked, np.log10(p50_masked)+y_offset, color=color,  lw=1.5, label=f'z={zarray[ik]:.1f}')
    ax.fill_between(m1_masked, np.log10(p5_masked) + y_offset, np.log10(p95_masked) + y_offset, color=color, alpha=0.3)
    y_offset += 2


sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label='$z$')
ax.set_xlabel(r"$m_1$")
ax.set_ylim(-5, 17)
ax.set_ylabel(r'$\mathrm{log}10(\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}V_c) [\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-1}]$ + offset',fontsize=18)
ax.set_xlim(5, 80)
#ax.set_ylim(ymin=-5)
plt.tight_layout()
plt.savefig('offset_rate_m1_redshiftcolor_m1m2dLanalysis_pdetcap_01_Xieffmaxbw_dL_03.png'.format(dLval), bbox_inches='tight')
plt.semilogx()
plt.tight_layout()
plt.savefig('offset_rate_m1_redshiftcolor_m1m2dLanalysis_pdetcap_01_Xieffmaxbw_dL_03LogX_axis.png', bbox_inches='tight')
plt.show()
