#Use the file with reweighting samples and hyoer params of KDE fromoptimized 3D code and compute KDE given grids/ get PDET as well (make sure indexing issue)
#get Rates 
#No converison factoris used at the moment we just save KDE PDET results and plots of rates
#for paper plotconversion will be done in another postprocess script

import sys
import matplotlib
sys.path.append('pop-de/popde/')
import density_estimate as d
import adaptive_kde as ad
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import h5py as h5
import scipy
import numpy as np
from scipy.integrate import quad, simpson
from scipy.interpolate import RegularGridInterpolator
from matplotlib import rcParams
from matplotlib.colors import LogNorm, Normalize
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u
import utils_plot as u_plot
import o123_class_found_inj_general as u_pdet

# Set Matplotlib parameters for consistent plotting
rcParams.update({
    "text.usetex": True,
    "font.serif": "Computer Modern",
    "font.family": "Serif",
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.fontsize": 18,
    "axes.labelsize": 18,
    "axes.grid": True,
    "grid.color": 'grey',
    "grid.linewidth": 1.0,
    "grid.alpha": 0.6
})


#careful parsers 
parser = argparse.ArgumentParser(description=__doc__)
# Input files #maybe we should combine these three to one
parser.add_argument('--datafilename1', help='h5 file containing N samples for m1for all gw bbh event')
parser.add_argument('--datafilename2', help='h5  file containing N sample of parameter2 (m2) for each event, ')
parser.add_argument('--datafilename3', help='h5  file containing N sample of dL for each event')
parser.add_argument('--datafilename-redshift', help='h5  file containing N sample of redshift for each event')
parser.add_argument('--parameter1', help='name of parameter which we use for x-axis for KDE', default='m1')
parser.add_argument('--parameter2', help='name of parameter which we use for y-axis for KDE: m2', default='m2')
parser.add_argument('--parameter3', help='name of parameter which we use for y-axis for KDE [can be Xieff, dL]', default='dL')
# selection effect capping
parser.add_argument('--max-pdet', default=0.1, type=float, help='Capping value for small pdet to introduce regularization.')
# priors 
parser.add_argument('--m1-min', default=5.0, type=float, help='Minimum value for primary mass m1.')
parser.add_argument('--m1-max', default=100.0, type=float, help='Maximum value for primary mass m1.')
parser.add_argument('--Npoints', default=200, type=int, help='Number of points for KDE evaluation.')
parser.add_argument('--param2-min', default=4.95, type=float, help='Minimum value for parameter 2 if it is  m2, else if dL use 10')
parser.add_argument('--param2-max', default=100.0, type=float, help='Maximum value for parameter 2 if it is m2 else if dL  use 10000')
parser.add_argument('--param3-min', default=200., type=float, help='Minimum value for parameter 3 if it is  dL, else if Xieff use -1')
parser.add_argument('--param3-max', default=8000., type=float, help='Maximum value for parameter 3 if it is dL else if Xieff  use +1')

parser.add_argument('--discard', default=100, type=int, help=('discard first 100 iterations'))
parser.add_argument('--NIterations', default=1000, type=int, help='Total number of iterations for the reweighting process.')


#plots and saving data
parser.add_argument('--pathplot', default='./', help='public_html path for plots', type=str)
parser.add_argument('--iterative-result-filename', required=True, help='write a proper name of output hdf files based on analysis', type=str)
parser.add_argument('--output-filename', default='m1m2mdL3Danalysis_slice_dLoutput-filename', help='write a proper name of output hdf files based on analysis', type=str)
opts = parser.parse_args()
#####################################################################
###cosmology 
H0 = 67.9  # km/s/Mpc
omega_m = 0.3065
cosmo = FlatLambdaCDM(H0=H0, Om0=omega_m)

def get_mass_indetector_frame(dLMpc, mass):
    """
    Convert the mass of an object to its equivalent in the detector frame,
    accounting for cosmological redshift.

    This function computes the effective mass of an object observed at a
    certain luminosity distance, considering the cosmological redshift.

    Parameters:
    -----------
    dLMpc : float or array
        The luminosity distance to the object in megaparsecs (Mpc). If provided as a Quantity,
        it should have units of Mpc.

    mass : float or array
        The source frame mass
    Returns:
    --------
    mdet : float or array
        mass of the object in the detector frame (accounting for redshift).
        The returned valuescaled by the cosmological factor (1+z)
    """

    zcosmo = z_at_value(cosmo.luminosity_distance, dLMpc*u.Mpc).value
    mdet = mass*(1.0 + zcosmo)
    return mdet

#####################################
# get PDET  (m1, m2, dL)
run_fit = 'o3'
run_dataset = 'o3'
dmid_fun = 'Dmid_mchirp_fdmid_fspin' #'Dmid_mchirp_fdmid'
emax_fun = 'emax_exp'
alpha_vary = None
g = u_pdet.Found_injections(dmid_fun = 'Dmid_mchirp_fdmid_fspin', emax_fun='emax_exp', alpha_vary = None, ini_files = None, thr_far = 1, thr_snr = 10)

##################Only medians of m1-m2 and dL needed############
fz = h5.File(opts.datafilename_redshift, 'r')
#fz = h5.File('Final_noncosmo_GWTC3_redshift_datafile.h5', 'r')
dz = fz['randdata']
f1 = h5.File(opts.datafilename1, 'r')#m1
#f1 = h5.File('Final_noncosmo_GWTC3_m1srcdatafile.h5', 'r')#m1
d1 = f1['randdata']
#f2 = h5.File('Final_noncosmo_GWTC3_m2srcdatafile.h5', 'r')#m2
f2 = h5.File(opts.datafilename3, 'r')#m2
d2 = f2['randdata']
#f3 = h5.File('Final_noncosmo_GWTC3_dL_datafile.h5', 'r')#dL
f3 = h5.File(opts.datafilename3, 'r')#dL
d3 = f3['randdata']
medianlist1 = f1['initialdata/original_mean'][...]
sampleslists2 = []
medianlist2 = f2['initialdata/original_mean'][...]
medianlist3 = f3['initialdata/original_mean'][...]

f1.close()
f2.close()
f3.close()
fz.close()

meanxi1 = np.array(medianlist1)
meanxi2 = np.array(medianlist2)
meanxi3 = np.array(medianlist3)
sample = np.vstack((meanxi1, meanxi2, meanxi3)).T

####### to get KDEs we need eval points
if opts.m1_min is not None and opts.m1_max is not None:
    xmin, xmax = opts.m1_min, opts.m1_max
else:
    xmin, xmax = np.min(flat_samples1), np.max(flat_samples1)

if opts.param2_min is not None and opts.param2_max is not None:
    ymin, ymax = opts.param2_min, opts.param2_max
else:
    ymin, ymax = np.min(flat_samples2) , np.max(flat_samples2)

if opts.param3_min is not None and opts.param3_max is not None:
    zmin, zmax = opts.param3_min, opts.param3_max
else:
    zmin, zmax = np.min(flat_samples3) , np.max(flat_samples3)

# can use opts but for now using fixed
xmin, xmax = 5, 105  #m1
ymin, ymax = 5, 105 #m2
zmin, zmax = 200, 8000 #dL
Npoints = 200 #opts.Npoints

m1_src_grid = np.logspace(np.log10(xmin), np.log10(xmax), Npoints)
m2_src_grid = np.logspace(np.log10(ymin), np.log10(ymax), Npoints)
dL_grid = np.linspace(zmin, zmax, 150)  #we only need for fix m2 case

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
        ratem1[xid] = simpson(rate_vals, x=yval[y_valid])
    for yid, m2 in enumerate(m2val):
        x_valid = xval >= yval[yid]  # Only accept points with y <= x
        rate_vals = kde[x_valid, yid]
        ratem2[yid] = simpson(rate_vals,x= xval[x_valid])
    return ratem1

#File obtained with 3D iterative KDEs
hdf_file = opts.iterative_result_filename# Your HDF5 file path
hdf = h5.File(hdf_file, 'r')
Total_Iterations = int(opts.NIterations)
discard = int(opts.discard)
def get_kdes_for_fixedDL(dLval, savefilename):
    """
    for fixed dL eval KDEs using iterative results 
    and save data 
    """
    #save data of KDE (all iterations) and PDET  at eval points  
    savehf  =  h5.File(savefilename+'_fixed_dL{0}.hdf5'.format(dLval), 'w')
    p3grid = np.array([dLval])
    m1_det_grid = get_mass_indetector_frame(dLval, m1_src_grid)
    m2_det_grid = get_mass_indetector_frame(dLval, m2_src_grid)
    XX, YY, ZZ = np.meshgrid(m1_src_grid, m2_src_grid, p3grid, indexing='ij')
    eval_samples = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    #for detector frame mass
    XXd, YYd, ZZd = np.meshgrid(m1_det_grid, m2_det_grid, p3grid, indexing='ij')
    #PDET with masses in det frame
    PDET = u_pdet.get_pdet_m1m2dL(XXd, YYd, ZZd, classcall=g)
    PDET_slice = PDET[:, :, 0]
    savehf.create_dataset('PDET2D', data=PDET_slice)
    PDETfilter = np.maximum(PDET_slice, 0.1)
    m1_dLslice = XX[:, :, 0]
    m2_dLslice = YY[:, :, 0]
    savehf.create_dataset('xx2d', data=m1_dLslice)
    savehf.create_dataset('yy2d', data=m2_dLslice)
    savehf.create_dataset('PDET2Dfiltered01', data=PDETfilter)
    kde_list = []
    rate_list = []
    rate1Dlist = []
    for i in range(Total_Iterations+discard):#fix this as this can change
        iteration_name = f'iteration_{i}'
        if iteration_name not in hdf:
            print(f"Iteration {i} not found in file.")
            continue

        group = hdf[iteration_name]
        samples = group['rwsamples'][:]
        alpha = group['alpha'][()]
        bwx = group['bwx'][()]
        bwy = group['bwy'][()]
        bwz = group['bwz'][()]

        # Create the KDE with mass symmetry
        m1 = samples[:, 0]  # First column corresponds to m1
        m2 = samples[:, 1]  # Second column corresponds to m2
        dL = samples[:, 2]  # Third column corresponds to dL
        samples2 = np.vstack((m2, m1, dL)).T
        #Combine both samples into one array
        symmetric_samples = np.vstack((samples, samples2))
        train_kde = ad.AdaptiveBwKDE(
                symmetric_samples,
                None,
                input_transf=('log', 'log', 'none'),
                stdize=True,
                rescale=[1/bwx, 1/bwy, 1/bwz],
                alpha=alpha
            )
        # Evaluate the KDE on the evaluation samples
        eval_kde3d = train_kde.evaluate_with_transf(eval_samples)
        KDE_slice = eval_kde3d.reshape(m1_dLslice.shape)
        savehf.create_dataset("kde_iter{0}".format(i), data=KDE_slice)
        savehf.flush()
        current_rateval = len(m1)*KDE_slice/PDETfilter
        #get rate_m1_integrating over_m2
        rateOneD = IntegrateRm1m2_wrt_m2(m1_src_grid, m2_src_grid, current_rateval)
        kde_list.append(KDE_slice)
        rate_list.append(current_rateval)
        rate1Dlist.append(rateOneD)
    savehf.close()
    kde_array = np.array(kde_list)  #Shape:(num_iter, num_eval_pts)
    rate_array = np.array(rate_list)
    u_plot.average2Dkde_m1m2_plot(meanxi1, meanxi2, m1_dLslice, m2_dLslice, kde_list[discard:], pathplot=opts.pathplot, titlename=1001, plot_label='KDE', x_label='m1', y_label='m2', plottag='Combined_all_', dLval=dLval, correct_units=False)
    u_plot.average2Dkde_m1m2_plot(meanxi1, meanxi2, m1_dLslice, m2_dLslice, rate_list[discard:], pathplot=opts.pathplot, titlename=1001, plot_label='Rate', x_label='m1', y_label='m2', plottag='Combined_all_', dLval=dLval, correct_units=False)
   # we also want to get oneD integrate over m2 with removing m2>m1 vals and get 5th, 95th and50th percentile of those
    rate_density = np.percentile(rate1Dlist[discard:], 50.,  axis=0)
    rate_density95 = np.percentile(rate1Dlist[discard:], 95., axis=0)
    rate_density5 = np.percentile(rate1Dlist[discard:], 5., axis=0)
    plt.figure()
    plt.plot(m1_src_grid, rate_density, 'k',  lw=2.5)
    plt.plot(m1_src_grid, rate_density5, 'r', ls='--' , lw=1.5)
    plt.plot(m1_src_grid, rate_density95, 'r', ls='--' , lw=1.5)
    plt.xlabel(r"$m_1 \,  [M_\odot]$", fontsize=20)
    plt.title(r"$d_L= {0}$ [Mpc]".format(dLval), fontsize=20)
    plt.semilogy()
    plt.ylim(ymin=1e-5)
    plt.ylabel(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}d_L [\mathrm{Mpc}^{-1}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-1}]$',fontsize=18)
    plt.tight_layout()
    plt.savefig(opts.pathplot+'m1_rate_dL{0}.png'.format(dLval), bbox_inches='tight')
    plt.close()

def get_kdes_for_fixedm2(m2value, savefilename):
    m2_src_grid = np.array([m2value])
    savehf  =  h5.File(savefilename+'_fixed_m2{0}.hdf5'.format(m2value), 'w')
    XX, YY, ZZ = np.meshgrid(m1_src_grid, m2_src_grid, dL_grid, indexing='ij')
    eval_samples = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    #for pdet we need mass in detector  frame
    redshiftDLgrid = z_at_value(cosmo.luminosity_distance, ZZ*u.Mpc).value
    XXd = XX*(1.0 + redshiftDLgrid)
    YYd = YY*(1.0 + redshiftDLgrid)
    PDET = u_pdet.get_pdet_m1m2dL(XXd, YYd, ZZ, classcall=g)
    PDET_slice = PDET[:, 0, :]
    savehf.create_dataset('PDET2D', data=PDET_slice)
    PDETfilter = np.maximum(PDET_slice, 0.1)
    m1_slice = XX[:, 0, :]
    dL_slice = ZZ[:, 0, :]
    savehf.create_dataset('xx2d', data=m1_slice)
    savehf.create_dataset('yy2d', data=dL_slice)
    savehf.create_dataset('PDET2Dfiltered01', data=PDETfilter)
    u_plot.plot_pdet2D(m1_slice, dL_slice, PDETfilter, Maxpdet=0.1, pathplot=opts.pathplot, show_plot=False)
    kde_list = []
    kdev_list = []
    rate_list = []
    ratev_list = []
    rate1Dlist = []
    for i in range(Total_Iterations+discard):#fix this as this can change
        iteration_name = f'iteration_{i}'
        if iteration_name not in hdf:
            print(f"Iteration {i} not found in file.")
            continue
        group = hdf[iteration_name]
        samples = group['rwsamples'][:]
        alpha = group['alpha'][()]
        bwx = group['bwx'][()]
        bwy = group['bwy'][()]
        bwz = group['bwz'][()]

        # Create the KDE with mass symmetry
        m1 = samples[:, 0]  # First column corresponds to m1
        m2 = samples[:, 1]  # Second column corresponds to m2
        dL = samples[:, 2]  # Third column corresponds to dL
        samples2 = np.vstack((m2, m1, dL)).T
        #Combine both samples into one array
        symmetric_samples = np.vstack((samples, samples2))
        train_kde = ad.AdaptiveBwKDE(
                symmetric_samples,
                None,
                input_transf=('log', 'log', 'none'),
                stdize=True,
                rescale=[1/bwx, 1/bwy, 1/bwz],
                alpha=alpha
            )

        # Evaluate the KDE on the evaluation samples
        eval_kde3d = train_kde.evaluate_with_transf(eval_samples)
        KDE_slice = eval_kde3d.reshape(m1_slice.shape)
        savehf.create_dataset("kde_iter{0}".format(i), data=KDE_slice)
        savehf.flush()
        current_rateval = len(m1)*KDE_slice/PDETfilter
        kde_list.append(KDE_slice)
        rate_list.append(current_rateval)
    savehf.close()
    u_plot.average2DlineardLrate_plot(meanxi1, meanxi3, m1_slice, dL_slice, kde_list[discard:], pathplot='./', titlename=1, plot_label='KDE', x_label='m1', y_label='dL', show_plot=False)
    u_plot.average2DlineardLrate_plot(meanxi1, meanxi3, m1_slice, dL_slice, rate_list[discard:], pathplot='./', titlename=1, plot_label='Rate', x_label='m1', y_label='dL', show_plot=False)

for dLvalue  in [300]:#, 500, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3500, 4000, 4500, 5000]:
    get_kdes_for_fixedDL(dLvalue, opts.output_filename)

for m2_val in [10]:
    get_kdes_for_fixedm2(m2_val, opts.output_filename)
hdf.close()
