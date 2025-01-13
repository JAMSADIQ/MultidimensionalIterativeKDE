import matplotlib
import numpy as np
import matplotlib.pyplot as plt
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


#######
fz = h5.File('/home/jsadiq/Research/J_Ana/cbc_pdet/cbc_pdet/Final_noncosmo_GWTC3_redshift_datafile.h5', 'r')
dz = fz['randdata']
f1 = h5.File('/home/jsadiq/Research/J_Ana/cbc_pdet/cbc_pdet/Final_noncosmo_GWTC3_m1srcdatafile.h5', 'r')#m1
d1 = f1['randdata']
f2 = h5.File('/home/jsadiq/Research/J_Ana/cbc_pdet/cbc_pdet/Final_noncosmo_GWTC3_m2srcdatafile.h5', 'r')#m2
d2 = f2['randdata']
f3 = h5.File('/home/jsadiq/Research/J_Ana/cbc_pdet/cbc_pdet/Final_noncosmo_GWTC3_dL_datafile.h5', 'r')#dL
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


##############convert to dR(z)/dVc  in 1/(Gpc^3 yr M^2)
import numpy as np
from scipy.integrate import quad
def hubble_parameter(z, H0, Omega_m):
    """
    Calculate the Hubble parameter H(z) for a flat Lambda-CDM cosmology.
    """
    Omega_Lambda = 1.0 - Omega_m
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)

hubble_parameter_vec = np.vectorize(hubble_parameter)
def comoving_distance(z, H0, Omega_m):
    """
    Compute the comoving distance D_c(z) in Mpc.
    """
    integrand = lambda z_prime: c / hubble_parameter(z_prime, H0, Omega_m)
    D_c, _ = quad(integrand, 0, z)
    return D_c
comoving_distance_vec = np.vectorize(comoving_distance)

def comoving_distance_derivative(z, H0, Omega_m):
    """
    Compute the derivative of comoving distance dD_c/dz in Mpc.
    """
    return c / hubble_parameter(z, H0, Omega_m)
comoving_distance_derivative_vec = np.vectorize(comoving_distance_derivative)

#cosmology parameter we are using are
H0 = 67.9 #km/sMpc
c = 3e5 #km/s
omega_m = 0.3065 #matter density
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value, Planck15
cosmo =  FlatLambdaCDM(H0, omega_m)

def get_massed_indetector_frame(dLMpc, mass):
    zcosmo = z_at_value(cosmo.luminosity_distance, dLMpc*u.Mpc).value
    mdet = mass*(1.0 + zcosmo)
    return mdet

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


def average2DlineardLrate_plot(m1vals, m2vals, XX, YY, kdelists, pdet2Dnofilter,pathplot='./', titlename=1, plot_label='Rate', x_label='m1', y_label='dL', show_plot=False):
    sample1, sample2 = m1vals, m2vals
    CI50 = np.percentile(kdelists, 50, axis=0)
    fig, axl = plt.subplots(1,1,figsize=(8,6))

    levels_pdet = [0.01, 0.03,0.1]  # Two levels for PDET
    pdet_contour = axl.contour(XX, YY, pdet2Dnofilter, colors=['orange', 'orange', 'orange'], linestyles=['--', '--', '--'], levels=levels_pdet, linewidths=2)
    contour_label_positions = []
    for collection in pdet_contour.collections:
        paths = collection.get_paths()
        for path in paths:
        # Get the middle of each path segment
            vertices = path.vertices
            midpoint_index = len(vertices) // 4
            contour_label_positions.append(vertices[-midpoint_index])
    axl.clabel(pdet_contour, inline=True, fontsize=15, fmt="%.2f", manual=contour_label_positions) 
    #rest of the plot
    max_density = np.max(CI50)
    max_exp = np.floor(np.log10(max_density))  # Find the highest power of 10 below max_density
    contourlevels = 10 ** (max_exp - np.arange(5))[::-1]
    p = axl.pcolormesh(XX, YY, CI50, cmap=plt.cm.get_cmap('Purples'), norm=LogNorm(vmin=contourlevels[0], vmax =contourlevels[-1]),  label=r'$p(m_1, d_L)$')
    cbar = plt.colorbar(p, ax= axl)
    if plot_label =='Rate':
        cbar.set_label(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}m_2\mathrm{d}dV_c [\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-2}]$',fontsize=18)
        #cbar.set_label(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}d_L [\mathrm{Mpc}^{-1}\, (\mathrm{m}/{M}_\odot)^{-1}  \mathrm{yr}^{-1}]$',fontsize=18)
    else:
        #cbar.set_label(r'$p(\mathrm{ln}m_{1, source}, d_L)$',fontsize=18)
        cbar.set_label(r'$p(m_{1, source}, d_L)$',fontsize=18)
    CS = axl.contour(XX, YY, CI50, colors='black', levels=contourlevels ,linestyles='dashed', linewidths=2, norm=LogNorm())
    axl.scatter(sample1, sample2,  marker="+", color="r", s=20)
    axl.set_ylabel(r'$d_L\,[Mpc]$', fontsize=18)
    axl.set_xlabel(r'$m_\mathrm{1, source} \,[M_\odot]$', fontsize=20)
    #axl.legend()
    axl.set_ylim(ymin=200, ymax=7000)
    axl.semilogx()
    #axl.set_aspect('equal')
    axl.set_title(r'$m_2={0}\, [M_\odot]$'.format(m2val), fontsize=20)
    fig.tight_layout()
    #plt.savefig(pathplot+"Comoving_Volume_Units_average_"+plot_label+"{0}.png".format(titlename))
    #if show_plot== True:
    plt.show()
    #else:
    #    plt.close()
    return CI50


#########################################################
xmin, xmax = 5.,  105.#  m1  
ymin, ymax = 200, 8000 #dL np.min(flat_samples2) , np.max(flat_samples2)
Npoints = 150 #opts.Npoints

m1_grid = np.logspace(np.log10(xmin), np.log10(xmax),  200)
#dLgrid
dL_grid = np.linspace(ymin, ymax, Npoints)
#######Volume_factor ##############
volume_factor1D = np.zeros(len(dL_grid))
for i, dLval in enumerate(dL_grid):
    volume_factor1D[i] = get_dVdz_factor(dLval)
plt.figure(figsize=(8,6))
plt.plot(dL_grid, volume_factor1D, 'r')
#plt.semilogy()
plt.xlabel(r"$d_L\,$[Mpc]", fontsize=18)
plt.ylabel(r"$\frac{dV_c}{dz}$", fontsize=18)
plt.show()
# we need volume2D shape shape as KDE
pathplot = './comoving_units_'
for m2val  in [10, 35]:#
    m2_grid = np.array([m2val])
    savehf  =  h5.File('Correctedpdet_KDEsdata_fromOptimizeData_m2{0}.hdf5'.format(m2val), 'r')
    XX, YY, ZZ = np.meshgrid(m1_grid, m2_grid, dL_grid, indexing='ij')
    xxd, yyd, volume3D = np.meshgrid(m1_grid, m2_grid, volume_factor1D, indexing='ij')
    
    #PDET = np.zeros((Npoints, 1, Npoints))
    #now PDET is correctly obtained at source frame masses
    PDET_slice = savehf['PDET2D'][...]
    PDETfilter = np.maximum(PDET_slice, 0.1)
    print("min pdet = ", np.min(PDETfilter))
    #M1_slice = XX[:, 0, :]
    #dL_slice = ZZ[:, 0, :]
    M1_slice = savehf['xx2d'][...]
    dL_slice = savehf['yy2d'][...]
    volume2D = volume3D[:, 0, :]
    mm, vd2 = np.meshgrid(m1_grid, volume_factor1D, indexing='ij')
    cPDETfilter = savehf['PDET2Dfiltered01'][...]
    #plot to check
    plt.figure(figsize=(10, 8))
    # Meshgrid plot with log normalization
    mesh = plt.pcolormesh(M1_slice, dL_slice, volume2D, norm=LogNorm())#vmin=levels[0], vmax=levels[-1]), cmap='viridis', shading='auto')
    plt.colorbar(mesh, label=r'$\frac{dV_c}{dz}$')
    # Contour lines
    contours = plt.contour(M1_slice, dL_slice, volume2D,  norm=LogNorm(), colors='k', linewidths=1.5)
    plt.clabel(contours, inline=True, fontsize=20, fmt="%.1e")
    # Labels and title
    plt.xlabel(r'$m_1\, [M_\odot]$', fontsize=18)
    plt.ylabel(r'$d_L\,$[Mpc]', fontsize=18)
    plt.semilogx()
    plt.title('comoving volume factor')
    plt.show()

    kde_list = []
    kdev_list = []
    rate_list = []
    ratev_list = []
    rate1Dlist = []
    for i in range(1100):#fix this as this can change
        KDE_slice = savehf["kde_iter{0}".format(i)][...]
        current_rateval = 69*KDE_slice/PDETfilter
        current_rateval_v = current_rateval/volume2D
        kde_list.append(KDE_slice)
        rate_list.append(current_rateval)
        ratev_list.append(current_rateval_v)
    savehf.close()
    kde_array = np.array(kde_list)  #Shape:(num_iter, num_eval_pts)
    rate_array = np.array(rate_list)
    #here we need m1dL plots but also volume factor
    #average2DlineardLrate_plot(meanxi1, meanxi3, M1_slice, dL_slice, kde_list[100:], PDET_slice,pathplot='./', titlename=1, plot_label='KDE', x_label='m1', y_label='dL', show_plot=True)
    #average2DlineardLrate_plot(meanxi1, meanxi3, M1_slice, dL_slice, rate_list[100:], PDET_slice,pathplot='./', titlename=1, plot_label='Rate', x_label='m1', y_label='dL', show_plot=True)
    average2DlineardLrate_plot(meanxi1, meanxi3, M1_slice, dL_slice, ratev_list, PDET_slice, pathplot='./', titlename=1, plot_label='Rate', x_label='m1', y_label='dL', show_plot=True)

