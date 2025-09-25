import sys
import numpy as np
import argparse
import h5py as h5
from scipy.integrate import quad, simpson
from scipy.interpolate import RegularGridInterpolator
from matplotlib import use; use('agg')
from matplotlib import rcParams
from popde import density_estimate as kde, adaptive_kde as akde
import utils_plot as u_plot

#'''
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
#'''

parser = argparse.ArgumentParser(description=__doc__)
# Input data
parser.add_argument('--iterative-result', required=True)
parser.add_argument('--samplesx1', help='h5 file containing N samples of m1 for each event')
parser.add_argument('--samplesx2', help='h5 file containing N samples of m2 for each event')
parser.add_argument('--samplesx3', help='h5 file containing N samples of chi_eff for each event')
parser.add_argument('--vt-file', required=True, help='VT grid hdf file')
parser.add_argument('--vt-multiplier', type=float, help='Multiplier to scale VTs up/down')

# Iterative options
parser.add_argument('--discard', default=100, type=int, help='discard first DISCARD iterations')
parser.add_argument('--start-iter', type=int, help='start at iteration START_ITER after discards')
parser.add_argument('--end-iter', type=int, help='end at iteration END_ITER after discards')

# Plots and saving data
parser.add_argument('--pathplot', default='./', help='public_html path for plots')
parser.add_argument('--output-tag', required=True)

opts = parser.parse_args()


#################Integration functions######################
def integral_wrt_chieff(cf_mesh, cf_grid, Rate_3d):
    """
    Compute moments of chieff distribution at each point over m1, m2.
    Rate_3d is stored with indexing 'ij' way
    """
    integm1m2 = simpson(Rate_3d, x=cf_grid, axis=2)
    integchi_m1m2 = simpson(Rate_3d * cf_mesh, x=cf_grid, axis=2)
    integchisq_m1m2 = simpson(Rate_3d * cf_mesh * cf_mesh, x=cf_grid, axis=2)

    return integm1m2, integchi_m1m2, integchisq_m1m2


def get_m_chieff_rate_at_fixed_q(m1grid, m2grid, chieffgrid, Rate_3d, q=1.0):
    """
    q must be <=1 as m2 = q * m1mesh
    """
    M, _ = np.meshgrid(m1grid, chieffgrid, indexing='ij')
    m2values = q * M
    Rate2Dfixed_q = np.zeros_like(M)
    interpolator = RegularGridInterpolator((m1grid, m2grid, chieffgrid), Rate_3d, bounds_error=False, fill_value=None)
    for ix, m1val in enumerate(m1grid):
        Rate2Dfixed_q[ix, :] = interpolator((m1val, m2_values[ix, :], chieffgrid))
    return Rate2Dfixed_q


def get_rate_m_oneD(m1_query, m2_query, Rate):
    ratem1 = np.zeros(len(m1_query))
    ratem2 = np.zeros(len(m2_query))
    for xid, m1 in enumerate(m1_query):
        y_valid = m2_query <= m1_query[xid]  # Only accept points with y <= x
        rate_vals = Rate[y_valid, xid]
        #print(rate_vals, m2_query[y_valid])
        ratem1[xid] = simpson(rate_vals, m2_query[y_valid])
    for yid, m2 in enumerate(m2_query):
        x_valid = m1_query >= m2_query[yid]
        rate_vals = Rate[x_valid, yid]
        ratem2[yid] = simpson(rate_vals,  m1_query[x_valid])
    return ratem1, ratem2


def get_rate_m_chieff2D(m1_query, m2_query, Rate):
    ratem1 = np.zeros((len(m1_query), Rate.shape[2]))
    ratem2 = np.zeros((len(m2_query), Rate.shape[2]))

    # Iterate over each slice along the third dimension
    for i in range(Rate.shape[2]):
        # Extract the 2D slice
        Rate_slice = Rate[:, :, i]

        # Compute ratem1
        for xid, m1 in enumerate(m1_query):
            y_valid = m2_query <= m1_query[xid]  # Only accept points with y <= x
            rate_vals = Rate_slice[y_valid, xid]
            ratem1[xid, i] = simpson(rate_vals, m2_query[y_valid])

        # Compute ratem2
        for yid, m2 in enumerate(m2_query):
            x_valid = m1_query >= m2_query[yid]  # Only accept points with y <= x
            rate_vals = Rate_slice[x_valid, yid]
            ratem2[yid, i] = simpson(rate_vals, m1_query[x_valid])

    return ratem1, ratem2


#####################################################################
# Get original mean sample points
with h5.File(opts.samplesx1, 'r') as f1:
    mean1 = f1['initialdata/original_mean'][...]
with h5.File(opts.samplesx2, 'r') as f2:
    mean2 = f2['initialdata/original_mean'][...]
with h5.File(opts.samplesx3, 'r') as f3:
    mean3 = f3['initialdata/original_mean'][...]

Nev = mean1.size  # Number of detections

VTdata = h5.File(opts.vt_file, 'r')
m1grid = VTdata['m1vals'][:]
m2grid = VTdata['m2vals'][:]
cfgrid = VTdata['xivals'][:]
VT_3d = VTdata['VT'][...] / 1e9  # units Gpc^3
VTdata.close()

if opts.vt_multiplier:  # Scale up for rough estimates if exact VT not available
    VT_3d *= opts.vt_multiplier

hdf = h5.File(opts.iterative_result, 'r')

###### KDE eval 3D grid #########################
XX, YY, ZZ = np.meshgrid(m1grid, m2grid, cfgrid, indexing='ij')
eval_samples = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])

######## For 2D plots ##############################
M, CF = np.meshgrid(m1grid, cfgrid, indexing='ij')
M1, M2 = np.meshgrid(m1grid, m2grid, indexing='ij')

threeDgrid = np.array([XX.ravel(), YY.ravel(), ZZ.ravel()]).T

############ Saving data in 3 files #################################
hfintegm1m2 = h5.File(opts.output_tag + "_int_dchieff.hdf5", "w")
hfintegm1chieff = h5.File(opts.output_tag + "_int_dm2.hdf5", "w")
hfintegm2chieff = h5.File(opts.output_tag + "_int_dm1.hdf5", "w")

hfintegm1m2.create_dataset("M1mesh", data=M1)
hfintegm1m2.create_dataset("M2mesh", data=M2)
hfintegm1chieff.create_dataset("Mmesh", data=M)
hfintegm1chieff.create_dataset("CFmesh", data=CF)
hfintegm2chieff.create_dataset("Mmesh", data=M)
hfintegm2chieff.create_dataset("CFmesh", data=CF)

rate_m1m2 = []
ratem1_arr = []
ratem2_arr = []

KDEM1chieff = []
KDEM2chieff = []
RateM1chieff = []
RateM2chieff = []
kde3d_list = [] #if needed

###############################Iterations and evaluating KDEs/Rate
boots_weighted = False
vt_weights = False  # New flag to control VT weighting

for i in range(opts.end_iter - opts.start_iter):
    it = i + opts.discard + opts.start_iter
    ilabel = i + opts.start_iter
    if it % 5 == 0: print(it)
    iter_name = f'iteration_{it}'
    if iter_name not in hdf:
        print(f"Iteration {it} not found in file.")
        continue
    group = hdf[iter_name]
    if 'bootstrap_weights' in group:
        boots_weighted = True
        poisson_weights = group['bootstrap_weights'][:]
        assert min(poisson_weights) > 0, "Some bootstrap weights are non-positive!"
    
    # Check if VT weighting should be used
    if 'rwvt_vals' in group:
        vt_weights = True
        vt_vals = group['rwvt_vals'][:]
    
    samples = group['rwsamples'][:]
    alpha = group['alpha'][()]
    bwx = group['bwx'][()]
    bwy = group['bwy'][()]
    bwz = group['bwz'][()]
    
    # Create the KDE with mass symmetry
    m1 = samples[:, 0]  # First column corresponds to m1
    m2 = samples[:, 1]  # Second column corresponds to m2
    cf = samples[:, 2]
    samples2 = np.vstack((m2, m1, cf)).T
    # Combine both samples into one array
    symmetric_samples = np.vstack((samples, samples2))
    
    # Determine weights based on vt_weights flag
    if boots_weighted:
        if vt_weights:
            weights_over_VT = poisson_weights / vt_vals
            # Duplicate weights for symmetric samples (m1 <-> m2)
            weights = np.tile(weights_over_VT, 2)
        else:
            weights = np.tile(poisson_weights, 2)
    else:
        weights = None
    
    # If per-point bandwidth exists use it directly, otherwise 
    # use the adaptive KDE algorithm
    if 'perpoint_bws' in group:
        per_point_bandwidth = group['perpoint_bws'][...]
        
        train_kde = kde.VariableBwKDEPy(
            symmetric_samples,
            weights,
            input_transf=('log', 'log', 'none'),
            stdize=True,
            rescale=[1/bwx, 1/bwy, 1/bwz],
            bandwidth=per_point_bandwidth
        )
    else:
        train_kde = akde.AdaptiveBwKDE(
            symmetric_samples,
            weights,
            input_transf=('log', 'log', 'none'),
            stdize=True,
            rescale=[1/bwx, 1/bwy, 1/bwz],
            alpha=alpha
        )
    
    # Evaluate KDE
    eval_kde3d = train_kde.evaluate_with_transf(eval_samples)
    KDE_3d = eval_kde3d.reshape(XX.shape)
    
    # Calculate merger rate based on vt_weights flag
    if vt_weights:
        Rate_3d = Nev * KDE_3d  # KDE kernels are already weighted by 1/VT
    else:
        Rate_3d = Nev * KDE_3d / VT_3d
    
    # Calculate marginals
    kdeM1chieff, kdeM2chieff = get_rate_m_chieff2D(m1grid, m2grid, KDE_3d)
    rateM1chieff, rateM2chieff = get_rate_m_chieff2D(m1grid, m2grid, Rate_3d)
    ratem1m2, ratechim1m2, ratechisqm1m2 = integral_wrt_chieff(CF, cfgrid, Rate_3d)

    KDEM1chieff.append(kdeM1chieff)
    KDEM2chieff.append(kdeM2chieff)
    RateM1chieff.append(rateM1chieff)
    RateM2chieff.append(rateM2chieff)
    rate_m1m2.append(ratem1m2)

    hfintegm1m2.create_dataset(f"rate_m1m2_iter{ilabel}", data=ratem1m2)
    hfintegm1m2.create_dataset(f"rate_chim1m2_iter{ilabel}", data=ratechim1m2)
    hfintegm1m2.create_dataset(f"rate_chisqm1m2_iter{ilabel}", data=ratechisqm1m2)
    hfintegm1chieff.create_dataset(f"rate_m1cf_iter{ilabel}", data=rateM1chieff)
    hfintegm2chieff.create_dataset(f"rate_m2cf_iter{ilabel}", data=rateM2chieff)

    hfintegm1chieff.create_dataset(f"kde_m1cf_iter{ilabel}", data=kdeM1chieff)
    hfintegm2chieff.create_dataset(f"kde_m2cf_iter{ilabel}", data=kdeM2chieff)

    # get oneD output
    rateM1, rateM2 = get_rate_m_oneD(m1grid, m2grid, ratem1m2)
    ratem1_arr.append(rateM1)
    ratem2_arr.append(rateM2)

    hfintegm1m2.create_dataset(f"rate_m1_iter{ilabel}", data=rateM1)
    hfintegm1m2.create_dataset(f"rate_m2_iter{ilabel}", data=rateM2)

    hfintegm1m2.flush()
    hfintegm1chieff.flush()
    hfintegm2chieff.flush()

hfintegm1m2.close()
hfintegm1chieff.close()
hfintegm2chieff.close()

print('Making plots')

iter_tag = f"iter{opts.start_iter}_{opts.end_iter}"
rate_m1m2_med = np.percentile(rate_m1m2, 50, axis=0)
u_plot.m1m2_contour(mean1, mean2, M1, M2, rate_m1m2_med, itertag=iter_tag, pathplot=opts.pathplot, plot_name='Rate')

u_plot.m_chieff_contour(mean1, mean3, M, CF, np.percentile(KDEM1chieff, 50, axis=0), itertag=iter_tag, pathplot=opts.pathplot, plot_name='KDE', xlabel='m_1')
u_plot.m_chieff_contour(mean2, mean3, M, CF, np.percentile(KDEM2chieff, 50, axis=0), itertag=iter_tag, pathplot=opts.pathplot, plot_name='KDE', xlabel='m_2')
u_plot.m_chieff_contour(mean1, mean3, M, CF, np.percentile(RateM1chieff, 50, axis=0), itertag=iter_tag, pathplot=opts.pathplot, plot_name='Rate', xlabel='m_1')
u_plot.m_chieff_contour(mean2, mean3, M, CF, np.percentile(RateM2chieff, 50, axis=0), itertag=iter_tag, pathplot=opts.pathplot, plot_name='Rate', xlabel='m_2')

# 1-d rate vs masses
u_plot.oned_rate_mass(m1grid, m2grid, ratem1_arr, ratem2_arr, tag=iter_tag, pathplot=opts.pathplot)

######offset chieff plot ######################
m1_slice_values = np.array([10, 15, 20, 25, 35, 45, 55, 70])
m2_slice_values = m1_slice_values * 2./3.
u_plot.chieff_offset_plot(m1grid, cfgrid, m1_slice_values, RateM1chieff, offset_increment=5, m_label='m_1', tag=iter_tag, pathplot=opts.pathplot)
u_plot.chieff_offset_plot(m1grid, cfgrid, m2_slice_values, RateM2chieff, offset_increment=5, m_label='m_2', tag=iter_tag, pathplot=opts.pathplot)

