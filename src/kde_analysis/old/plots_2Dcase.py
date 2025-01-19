import sys
sys.path.append('pop-de/popde/')
import density_estimate as d
import adaptive_kde as ad
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import h5py as h5
import scipy
from scipy.interpolate import RegularGridInterpolator
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


import utils_plot as u_plot
import o123_class_found_inj_general as u_pdet

#careful parsers 
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--datafilename1', help='h5 file containing N samples for m1for all gw bbh event')
parser.add_argument('--datafilename2', help='h5  file containing N sample of parameter2 for each event, ')
parser.add_argument('--datafilename3', help='h5  file containing N sample of redshift for each event, ')
parser.add_argument('--parameter1', help='name of parameter which we use for x-axis for KDE', default='m1')
parser.add_argument('--parameter2', help='name of parameter which we use for y-axis for KDE [can be m2, Xieff, DL]', default='m2')
### For KDE in log parameter we need to add --logkde 
parser.add_argument('--logkde', action='store_true',help='if True make KDE in log params but results will be in onlog')
# limits on KDE evulation: 
parser.add_argument('--m1-min', help='minimum value for primary mass m1', type=float)
parser.add_argument('--m1-max', help='maximum value for primary mass m1', type=float)
parser.add_argument('--Npoints', default=200, type=int, help='Total points on which to evaluate KDE')
#m2-min must be <= m1-min
parser.add_argument('--param2-min', default=2.95, type=float, help='minimum value of m2 ,chieff =-1, DL= 1Mpc, used, must be below m1-min')
parser.add_argument('--param2-max', default=100.0, type=float, help='max value of m2 used, could be m1-max for chieff +1, for DL 10000Mpc')

parser.add_argument('--fpopchoice', default='kde', help='choice of fpop to be rate or kde', type=str)
parser.add_argument('--type-data', choices=['gw_pe_samples', 'mock_data'], help='mock data for some power law with gaussian peak or gwtc  pe samples data. h5 files for two containing data for median and sigma for m1')
#EMalgorithm reweighting 
parser.add_argument('--reweightmethod', default='bufferkdevals', help='Only for gaussian sample shift method: we can reweight samples via buffered kdevals(bufferkdevals) or buffered kdeobjects (bufferkdeobject)', type=str)
parser.add_argument('--reweight-sample-option', default='reweight', help='choose either "noreweight" or "reweight" if reweight use fpop prob to get reweight sample (one sample for no bootstrap or no or multiple samples for poisson)', type=str)
parser.add_argument('--bootstrap-option', default='poisson', help='choose either "poisson" or "nopoisson" if None it will reweight based on fpop prob with single reweight sample for eaxh event', type=str)

parser.add_argument('--buffer-start', default=500, type=int, help='start of buffer in reweighting.')
parser.add_argument('--buffer-interval', default=100, type=int, help='interval of buffer that choose how many previous iteration resulkts we use in next iteration for reweighting.')
parser.add_argument('--NIterations', default=1000, type=int, help='Total Iterations in reweighting')
parser.add_argument('--Maxpdet', default=0.1, type=float, help='capping for small pdet to regularization')

parser.add_argument('--pathplot', default='ThisComputer_2Dm1_dLpdet01casePlots/', help='public_html path for plots', type=str)
parser.add_argument('--pathfile', default='AnalysisCodes/', help='public_html path for plots', type=str)
parser.add_argument('--pathtag', default='re-weight-bootstrap_', help='public_html path for plots', type=str)
parser.add_argument('--output-filename', default='MassRedshift_with_reweight_output_data_', help='write a proper name of output hdf files based on analysis', type=str)
opts = parser.parse_args()



#######################################################################
##### If we want to use log param we need proper grid spacing in log scale
#p1grid = np.logspace(np.log10(xmin), np.log10(xmax), Npoints) 
#p2grid = np.linspace(ymin, ymax, Npoints) 
#XX, YY = np.meshgrid(p1grid, p2grid)
#xy_grid_pts = np.array(list(map(np.ravel, [XX, YY]))).T
#sample = sample = np.vstack((meanxi1, meanxi2)).T
injection_file = "/home/jxs1805/Research/CITm1dL/endo3_bbhpop-LIGO-T2100113-v12.hdf5"
with h5.File(injection_file, 'r') as f:
    T_obs = f.attrs['analysis_time_s']/(365.25*24*3600) # years
    N_draw = f.attrs['total_generated']

    injection_m1 = f['injections/mass1_source'][:]
    m2 = f['injections/mass2_source'][:]
    s1x = f['injections/spin1x'][:]
    s1y = f['injections/spin1y'][:]
    s1z = f['injections/spin1z'][:]
    s2x = f['injections/spin2x'][:]
    s2y = f['injections/spin2y'][:]
    s2z = f['injections/spin2z'][:]
    z = f['injections/redshift'][:]
    injection_dL = f["injections/distance"][:]
    injection_m1_det = injection_m1*(1.0 +  z)
    p_draw = f['injections/sampling_pdf'][:]
    pastro_pycbc_bbh = f['injections/pastro_pycbc_bbh'][:]
#here we need pdet
run_fit = 'o3'
run_dataset = 'o3'
dmid_fun = 'Dmid_mchirp_fdmid_fspin' #'Dmid_mchirp_fdmid'
emax_fun = 'emax_exp'
alpha_vary = None
g = u_pdet.Found_injections(dmid_fun = 'Dmid_mchirp_fdmid_fspin', emax_fun='emax_exp', alpha_vary = None, ini_files = None, thr_far = 1, thr_snr = 10)
H0 = 67.9 #km/sMpc
c = 299792.458 #3e5 #km/s
omega_m = 0.3065
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value, Planck15
cosmo =  FlatLambdaCDM(H0, omega_m)
def get_massed_indetector_frame(dLMpc, mass):
    zcosmo = z_at_value(cosmo.luminosity_distance, dLMpc*u.Mpc).value
    mdet = mass*(1.0 + zcosmo)
    return mdet



fz = h5.File('/home/jxs1805/Research/CITm1dL/PEfiles/Final_noncosmo_GWTC3_redshift_datafile.h5', 'r')
dz1 = fz['randdata']
f1 = h5.File('/home/jxs1805/Research/CITm1dL/PEfiles/Final_noncosmo_GWTC3_m1srcdatafile.h5', 'r')
d1 = f1['randdata']
f2 = h5.File('/home/jxs1805/Research/CITm1dL/PEfiles/Final_noncosmo_GWTC3_dL_datafile.h5', 'r')
d2 = f2['randdata']
sampleslists1 = []
medianlist1 = []
eventlist = []
sampleslists2 = []
medianlist2 = []
pdetlists = []
for k in d1.keys():
    eventlist.append(k)
    if (k  == 'GW190719_215514_mixed-nocosmo' or k == 'GW190805_211137_mixed-nocosmo'):
        print(k)
        d_Lvalues = d2[k][...]
        m_values = d1[k][...]
        mdet_values = d1[k][...]*(1.0 + dz1[k][...])
        #m_values, d_Lvalues, correct_indices =  preprocess_data(injection_m1, injection_dL, m_values, d_Lvalues, num_bins=10)
        #mdet_values = mdet_values[correct_indices]
        #pdet_values =  np.zeros(len(d_Lvalues))
        #for i in range(len(d_Lvalues)):
        #    pdet_values[i] = u_pdet.pdet_of_m1_dL_powerlawm2(mdet_values[i], 5.0, d_Lvalues[i], beta=1.26, classcall=g)

    else:
        m_values = d1[k][...]
        mdet_values = d1[k][...]*(1.0 + dz1[k][...])
        d_Lvalues = d2[k][...]
        #pdet_values =  np.zeros(len(d_Lvalues))
        #for i in range(len(d_Lvalues)):
        #    pdet_values[i] = u_pdet.pdet_of_m1_dL_powerlawm2(mdet_values[i], 5.0, d_Lvalues[i], beta=1.26, classcall=g)
    #pdetlists.append(pdet_values)
    sampleslists1.append(m_values)
    sampleslists2.append(d_Lvalues)
    medianlist1.append(np.percentile(m_values, 50))
    medianlist2.append(np.percentile(d_Lvalues, 50))

f1.close()
f2.close()
fz.close()
meanxi1 = np.array(medianlist1)
meanxi2 = np.array(medianlist2)
flat_samples1 = np.concatenate(sampleslists1).flatten()
flat_samples2 = np.concatenate(sampleslists2).flatten()
#flat_pdetlist = np.concatenate(pdetlists).flatten()
#print(len(flat_pdetlist))




############### save data in HDF file:
#frateh5 = h5.File('/home/jxs1805/Research/CITm1dL/Final_Fixed_pdet_m1det_MeanKDEbuffer_with_input_transf_PdetCap0_1KDEpycode1000iterationsm1source_rates_linear_dLdL2priorfactor_uniform_prior_mass_2Drate_m1dL.hdf5', 'r')
frateh5 = h5.File('new_this_computer_Fixed_pdet_m1det_MeanKDEbuffer_with_input_transf_PdetCap0_1KDEpycode1000iterationsm1source_rates_linear_dLdL2priorfactor_uniform_prior_mass_2Drate_m1dLmax_pdet_cap_0.1.hdf5', 'r')

XX = frateh5['data_xx'][:]
YY = frateh5['data_yy'][:]
p1grid = XX[:, 0]
p2grid = YY[0, :]

################################################################################
m2_min = 5.0 # 5 is correct limit for BHs
beta = 1.26 #spectrial index for q  
#if opts.fpopchoice == 'rate':
#    pdet2D = np.zeros((Npoints, Npoints))
#    mgrid = p1grid
#    dLgrid = p2grid
#    #convert masses im detector frame to make sure we are correctly computing pdet on same masses as KDE grid masses
#    mdetgrid = get_massed_indetector_frame(dLgrid, mgrid)
#    for i, m1val in enumerate(mdetgrid):
#        for j, dLval in enumerate(dLgrid):
#            pdet2D[i, j] = u_pdet.pdet_of_m1_dL_powerlawm2(m1val, m2_min, dLval, beta=beta, classcall=g)
#
#    ## Set all values in `pdet` less than 0.1/0.03   to 0.1or 0.03
pdet2Dnofilter = frateh5['pdet2D'][:]
pdet2D = np.maximum(pdet2Dnofilter, 0.1) #trans[pose already taken
plt.figure(figsize=(10, 8))
plt.contourf(XX, YY, pdet2D, levels=[0.001,0.01,0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0], cmap='viridis', norm=Normalize(vmax=1))
plt.title(r'$p_\mathrm{det}, \,  q^{1.26}, \, \mathrm{with} \, max(0.1, p_\mathrm{det})$', fontsize=18)
#plt.title(r'$p_\mathrm{det}, \,  q^{1.26}$', fontsize=18)
plt.colorbar(label=r'$p_\mathrm{det}$')
plt.contour(XX, YY, pdet2D, colors='white', linestyles='dashed', levels=[0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0])
#plt.xlabel(r'$m_{1, detector} [M_\odot]$', fontsize=20)
plt.xlabel(r'$m_{1, source} [M_\odot]$', fontsize=20)
plt.ylabel(r'$d_L [Mpc]$', fontsize=20)
plt.loglog()
plt.savefig(opts.pathplot+"testpdet2Dpowerlaw_m2.png")
plt.close()
#plt.show()
### reweighting EM algorithm
Total_Iterations = 1000#int(opts.NIterations)
discard = 100#int(opts.buffer_start)   # how many iterations to discard default =5
Nbuffer = 100#int(opts.buffer_interval) #100 buffer [how many (x-numbers of ) previous iteration to use in reweighting with average of those past (x-numbers of ) iteration results

iterkde_list = []
iter2Drate_list = []
iterbwxlist = []
iterbwylist = []
iteralplist = []
print(frateh5['iteration_0'])

for i in range(Total_Iterations + discard):
    group = frateh5[f'iteration_{i}']
    # Save the data in the group
    rwsamples = group['rwsamples'][...]
    shiftedalp = group['alpha'][...]
    bwx = group['bwx'][...]
    bwy = group['bwy'][...]
    current_kdeval = group['kde'][:]
    current_kdeval = current_kdeval.reshape(XX.shape)
    iterkde_list.append(current_kdeval)
    iterbwxlist.append(bwx)
    iterbwylist.append(bwy)
    iteralplist.append(shiftedalp)
    current_rateval = len(rwsamples)*current_kdeval/pdet2D
    iter2Drate_list.append(current_rateval)
iterstep = 1001
#u_plot.histogram_datalist(iterbwxlist[100:], dataname='bwx', pathplot=opts.pathplot, Iternumber=iterstep)
#u_plot.histogram_datalist(iterbwylist[100:], dataname='bwy', pathplot=opts.pathplot, Iternumber=iterstep)
#u_plot.histogram_datalist(iteralplist[100:], dataname='alpha', pathplot=opts.pathplot, Iternumber=iterstep)
frateh5.close()

ratelists = iter2Drate_list[100:]
CI50 = np.percentile(ratelists, 50, axis=0)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm

# Assuming XX, YY, PDET, iter2Drate_list, sample1, sample2, opts.pathplot are already defined
fig, axl = plt.subplots(1, 1, figsize=(8, 6))
sample1, sample2  = meanxi1, meanxi2
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
axl.clabel(pdet_contour, inline=True, fontsize=15, fmt="%.2f", manual=contour_label_positions)  # Label the contour lines
max_density = np.max(CI50)
max_exp = np.floor(np.log10(max_density))  # Find the highest power of 10 below max_density
contourlevels = 10 ** (max_exp - np.arange(4))[::-1]

# CI50 colormap
p = axl.pcolormesh(
    XX, YY, CI50,
    cmap=plt.cm.get_cmap('Purples'),
    norm=LogNorm(vmin=contourlevels[0], vmax=contourlevels[-1]),
    shading='auto'  # Ensure no warning for mismatched grids
)
cbar = plt.colorbar(p, ax=axl)
cbar.set_label(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}d_L [\mathrm{Mpc}^{-1}\, (\mathrm{m}/{M}_\odot)^{-1}  \mathrm{yr}^{-1}]$', fontsize=18)
# CI50 contour lines
cs = axl.contour(
    XX, YY, CI50,
    colors='black',
    levels=contourlevels,
    linestyles='dashed',
    linewidths=1.5,
    norm=LogNorm()
)

axl.scatter(sample1, sample2,  marker="+", color="r", s=20)
axl.set_ylabel(r'$d_L\,[Mpc]$', fontsize=18)
axl.set_xlabel(r'$m_\mathrm{1, source} \,[M_\odot]$', fontsize=18)
axl.set_ylim(ymax=7000)
axl.semilogx()
#axl.semilogy()
fig.tight_layout()
plt.savefig("Special_pdetcontourlines_on_combined_average_Rate1000Iteration.png")
plt.close()
#plt.show()

#quit()

#u_plot.average2DlineardLrate_plot(meanxi1, meanxi2, XX, YY, iterkde_list[discard:], pathplot=opts.pathplot+'allKDEscombined_', titlename=1001, plot_label='KDE', x_label='m1', y_label='dL', show_plot= True)
#u_plot.average2DlineardLrate_plot(meanxi1, meanxi2, XX, YY, iter2Drate_list[discard:], pathplot=opts.pathplot+'allKDEscombined_', titlename=1001, plot_label='Rate', x_label='m1', y_label='dL', show_plot= True)

#alpha bw plots

iterstep = 1001
#u_plot.histogram_datalist(iterbwxlist[100:], dataname='bwx', pathplot=opts.pathplot, Iternumber=iterstep)
#u_plot.histogram_datalist(iterbwylist[100:], dataname='bwy', pathplot=opts.pathplot, Iternumber=iterstep)
#u_plot.histogram_datalist(iteralplist[100:], dataname='alpha', pathplot=opts.pathplot, Iternumber=iterstep)
#u_plot.bandwidth_correlation(iterbwxlist, number_corr=discard, error=0.02,  pathplot=opts.pathplot+'bwx_')
#u_plot.bandwidth_correlation(iterbwylist, number_corr=discard, error=0.02,  pathplot=opts.pathplot+'bwy_')
#u_plot.bandwidth_correlation(iteralplist, number_corr=discard,  error=0.02, param='alpha',pathplot=opts.pathplot, log=False)

##########volumefactor toget correct units and make median/offset plots



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


dLarray = np.array([300, 600 ,900, 1200, 1500, 1800, 2100, 2400, 2700, 3000])#, 3000, 3500, 4000, 4500, 5000])
xlab  = np.random.choice(np.array([20, 25,30, 35, 40, 50]), size=len(dLarray))
zarray = z_at_value(cosmo.luminosity_distance, dLarray*u.Mpc).value
norm = Normalize(vmin=zarray.min(), vmax=zarray.max())

rate_lnm1dLmed = np.percentile(iter2Drate_list[discard:], 50., axis=0)
rate_lnm1dL_5 = np.percentile(iter2Drate_list[discard:], 5., axis=0)
rate_lnm1dL_95 = np.percentile(iter2Drate_list[discard:], 95., axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
colormap = plt.cm.rainbow
#colormap = plt.cm.hsv
dLarray = np.array([300, 600 ,900, 1200, 1500, 1800, 2100, 2400, 2700, 3000])#, 3000, 3500, 4000, 4500, 5000])
xlab  = np.random.choice(np.array([20, 25,30, 35, 40, 50]), size=len(dLarray))
zarray = z_at_value(cosmo.luminosity_distance, dLarray*u.Mpc).value
norm = Normalize(vmin=zarray.min(), vmax=zarray.max())
y_offset = 0  # Starting offset for ridgelines
y_gap = 100  # Gap between ridgelines
for ik, val in enumerate(dLarray):
    color = colormap(norm(zarray[ik]))
    closest_index = np.argmin(np.abs(YY - val))
    fixed_dL_value = YY.flat[closest_index]
    print(fixed_dL_value)
    volume_factor = get_dVdz_factor(fixed_dL_value)
    indices = np.isclose(YY, fixed_dL_value)
    # Extract the slice of rate_lnm1dL for the specified dL
    rate_lnm1_slice50 = rate_lnm1dLmed[indices]
    rate_lnm1_slice5 = rate_lnm1dL_5[indices]
    rate_lnm1_slice95 = rate_lnm1dL_95[indices]
    rate50 =  rate_lnm1_slice50/volume_factor
    rate05 =  rate_lnm1_slice5/volume_factor
    rate95 =  rate_lnm1_slice95/volume_factor
    # Extract the corresponding values of lnm1 from XX
    m1_values = XX[indices]
    print(m1_values)

    #plt.figure(figsize=(8, 6))
    #plt.plot(m1_values, rate_lnm1_slice50,  linestyle='-', color='k', lw=2)
    #plt.plot(m1_values, rate_lnm1_slice5,  linestyle='--', color='r', lw=1.5)
    #plt.plot(m1_values, rate_lnm1_slice95,  linestyle='--', color='r', lw=1.5)
    #plt.plot(m1_values, rate50,  linestyle='-', color='k', lw=2)
    #plt.plot(m1_values, rate05,  linestyle='--', color='r', lw=1.5)
    #plt.plot(m1_values, rate95,  linestyle='--', color='r', lw=1.5)
    #plt.xlabel(r'$m_{1,\, source}$')
    #plt.ylabel(r'$\mathrm{d}\mathcal{R}/m_1\mathrm{d}d_L [\mathrm{Mpc}^{-1}\, (\mathrm{m}/{M}_\odot)^{-1}  \mathrm{yr}^{-1}]$',fontsize=18)
    #plt.ylabel(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}V_c [\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-1}]$ ',fontsize=18)
    #plt.title(r'$d_L=${0}[Mpc]'.format(val))
    ##plt.semilogx()
    #plt.semilogy()
    #plt.ylim(ymin=1e-6)
    #plt.grid(True)
    #plt.tight_layout()
    #plt.savefig(opts.pathplot+'OneD_rate_m1_slicedL{0:.1f}.png'.format(val))
    #plt.semilogx()
    #plt.tight_layout
    #plt.savefig(opts.pathplot+'OneD_rate_m1_slicedL{0:.1f}LogXaxis_ComovingVolume_Units.png'.format(val))
    #plt.close()
    #print("done")

    #ax.plot(m1_values, rate50, color=color,  lw=1.5, label=f'z={zarray[ik]:.1f}')
    #label_x = xlab[ik] # Choose a midpoint for the label
    #label_y = np.interp(label_x, m1_values, rate_lnm1_slice50)

    #for offset
    peak = rate50.max()
    mask = rate50 >= 5e-3 * peak

    p50_masked = np.where(mask, rate50, np.nan)  # Replace invalid values with NaN
    p5_masked = np.where(mask, rate05, np.nan)  # Replace invalid values with NaN
    p95_masked = np.where(mask, rate95, np.nan)

    ax.plot(m1_values, np.log10(p50_masked)+y_offset, color=color,  lw=1.5, label=f'z={zarray[ik]:.1f}')
    ax.fill_between(m1_values, np.log10(p5_masked) + y_offset, np.log10(p95_masked) + y_offset, color=color, alpha=0.3)
    label_x = xlab[ik] # Choose a midpoint for the label
    label_y = np.interp(label_x, m1_values, rate_lnm1_slice50)
    #plt.text(label_x, np.log(label_y), f"z={zarray[ik]:.1f}", color=color, fontsize=15, ha="left", va="bottom", rotation=0, bbox=dict(boxstyle="round,pad=0.2", edgecolor="none", facecolor="white", alpha=0.7))
    y_offset += 2

#quit()
from matplotlib import cm
sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label='$z$')
ax.set_xlabel(r"$m_1$")
#plt.semilogy()
#ax.set_ylim(ymin=1e-3)
ax.set_ylabel(r'$\mathrm{log}10(\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}V_c) [\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-1}]$ + offset',fontsize=18)
ax.set_ylabel(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}V_c [\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-1}]$ ',fontsize=18)
ax.set_ylabel(r'$\mathrm{log}10(\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}V_c) [\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-1}]$ + offset',fontsize=18)
#ax.set_ylabel(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}d_L [\mathrm{Mpc}^{-1}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-1}]$ ',fontsize=18)
#ax.settitle(f"dL = {dLval}[Mpc]")
#ax.legend()
ax.set_xlim(5, 105)
#ax.set_ylim(ymin=1e-4)
ax.set_ylim(ymin=-5)
plt.tight_layout()
#plt.savefig(opts.pathplot+'median_rate_m1_at_slice_dLplot_with_redshift_colors.png')
plt.savefig(opts.pathplot+'offset_rate_m1_at_slice_dLplot_with_redshift_colors.png')

plt.semilogx()
plt.tight_layout()
plt.savefig(opts.pathplot+'offset_rate_m1_at_slice_dLplot_with_redshift_colors_log_Xaxis.png')
#plt.savefig(opts.pathplot+'median_rate_m1_at_slice_dLplot_with_redshift_colors_log_Xaxis.png')
plt.show()

