import sys
sys.path.append('pop-de/popde/')
import density_estimate as d
import adaptive_kde as ad
import numpy as np
import argparse
import h5py as h5
from scipy.integrate import quad, simpson
from scipy.interpolate import RegularGridInterpolator
from matplotlib import rcParams
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
# Input files #maybe we should combine these three to one
parser.add_argument('--datafilename1', help='h5 file containing N samples for m1 for each event')
parser.add_argument('--datafilename2', help='h5 file containing N samples of parameter2 (m2) for each event')
parser.add_argument('--datafilename3', help='h5 file containing N samples of z for each event')
#parser.add_argument('--parameter1', help='name of parameter which we use for x-axis for KDE', default='m1')
#parser.add_argument('--parameter2', help='name of parameter which we use for y-axis for KDE: m2', default='m2')
#parser.add_argument('--parameter3', help='name of parameter which we use for z-axis for KDE [can be Xieff, dL]', default='chi_eff')
#parser.add_argument('--m1-min', default=3.0, type=float, help='Minimum value for primary mass m1.')
#parser.add_argument('--m1-max', default=105.0, type=float, help='Maximum value for primary mass m1.')
#parser.add_argument('--Npoints-masses', default=150, type=int, help='Number of points for KDE evaluation.')
#parser.add_argument('--Npoints-param3', default=100, type=int, help='Number of points for KDE evaluation.')
#parser.add_argument('--param2-min', default=3.0, type=float, help='Minimum value for parameter 2 if it is m2, else if dL use 10')
#parser.add_argument('--param2-max', default=105.0, type=float, help='Maximum value for parameter 2 if it is m2 else if dLuse 10000')
#parser.add_argument('--param3-min', default=-1., type=float, help='Minimum value for parameter 3 if it is dL, use 500 else if Xieff use -1')
#parser.add_argument('--param3-max', default=1., type=float, help='Maximum value for parameter 3 if it is dL use 8000 else if Xieff use +1')

parser.add_argument('--discard', default=100, type=int, help='discard first DISCARD iterations')
#parser.add_argument('--n-iterations', default=1000, type=int, help='Total number of iterations for the reweighting process.')
parser.add_argument('--start-iter', type=int, help='start at iteration START_ITER after discards')
parser.add_argument('--end-iter', type=int, help='end at iteration END_ITER after discards')

#plots and saving data
parser.add_argument('--pathplot', default='./', help='public_html path for plots')
parser.add_argument('--iterative-result', required=True)
parser.add_argument('--vt-file', required=True, help='VT grid hdf file')
parser.add_argument('--output-tag', required=True)
opts = parser.parse_args()

Nev = 69

#################Integration functions######################
def integral_wrt_Xieff(KDE3D, VT3D, Xieff_grid, Nevents):
    """
    KDE3d and VT3d are computed with indexing ='ij' way
    """
    Rate3D = Nevents * KDE3D / VT3D
    integm1m2 = simpson(Rate3D, x=Xieff_grid, axis=2)
    return integm1m2


def get_m_Xieff_rate_at_fixed_q(m1grid, m2grid, Xieffgrid, Rate3D, q=1.0):
    """
    q must be <=1  as m2 = q*m1mesh
    """
    M, _ = np.meshgrid(m1grid, Xieffgrid, indexing='ij')
    m2values = q * M
    Rate2Dfixed_q = np.zeros_like(M)
    interpolator = RegularGridInterpolator((m1grid, m2grid, Xieffgrid), Rate3D, bounds_error=False, fill_value=None)
    for ix, m1val in enumerate(m1grid):
        Rate2Dfixed_q[ix, :] = interpolator((m1val, m2_values[ix, :], Xieffgrid))
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


def get_rate_m_Xieff2D(m1_query, m2_query, Rate):
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
f1 = h5.File(opts.datafilename1, 'r')
d1 = f1['randdata']
f2 = h5.File(opts.datafilename2, 'r')
d2 = f2['randdata']
f3 = h5.File(opts.datafilename3, 'r')
d3 = f3['randdata']
medianlist1 = f1['initialdata/original_mean'][...]
medianlist2 = f2['initialdata/original_mean'][...]
medianlist3 = f3['initialdata/original_mean'][...]

f1.close()
f2.close()
f3.close()

med1 = np.array(medianlist1)
med2 = np.array(medianlist2)
med3 = np.array(medianlist3)

VTdata = h5.File(opts.vt_file, 'r')
m1grid = VTdata['m1vals'][:]
m2grid = VTdata['m2vals'][:]
cfgrid = VTdata['xivals'][:]
VT_3D = VTdata['VT'][...] /1e9  # Gpc^3
VTdata.close()

hdf = h5.File(opts.iterative_result, 'r')

###### KDE eval 3D grid #########################
XX, YY, ZZ = np.meshgrid(m1grid, m2grid, cfgrid, indexing='ij')
eval_samples = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])

######## For 2D plots ##############################
M, CF = np.meshgrid(m1grid, cfgrid, indexing='ij')
M1, M2 = np.meshgrid(m1grid, m2grid, indexing='ij')

threeDgrid = np.array([XX.ravel(), YY.ravel(), ZZ.ravel()]).T

############ Saving data in 3 files #################################
savehfintegm1m2 = h5.File(opts.output_tag + "_int_dchieff.hdf5", "w")
savehfintegm1Xieff = h5.File(opts.output_tag + "_int_dm2.hdf5", "w")
savehfintegm2Xieff = h5.File(opts.output_tag + "_int_dm1.hdf5", "w")

savehfintegm1m2.create_dataset("M1mesh", data=M1)
savehfintegm1m2.create_dataset("M2mesh", data=M2)
savehfintegm1Xieff.create_dataset("Mmesh", data=M)
savehfintegm1Xieff.create_dataset("CFmesh", data=CF)
savehfintegm2Xieff.create_dataset("Mmesh", data=M)
savehfintegm2Xieff.create_dataset("CFmesh", data=CF)

rate_m1m2IntXieff = []
ratem1_arr = [] #np.zeros(shape=[900, len(m1_src_grid)])
ratem2_arr = [] #np.zeros(shape=[900, len(m2_src_grid)])

KDEM1Xieff = []
KDEM2Xieff = []
RateM1Xieff = []
RateM2Xieff = []
kde3d_list = [] #if needed

###############################Iterations and evaluating KDEs/Rate
for i in range(opts.end_iter - opts.start_iter):
    it = i + opts.discard + opts.start_iter
    ilabel = i + opts.start_iter
    if it % 2 == 0: print(it)
    iter_name = f'iteration_{it}'
    if iter_name not in hdf:
        print(f"Iteration {it} not found in file.")
        continue

    group = hdf[iter_name]
    samples = group['rwsamples'][:]
    alpha = group['alpha'][()]
    bwx = group['bwx'][()]
    bwy = group['bwy'][()]
    bwz = group['bwz'][()]

    # Create the KDE with mass symmetry
    m1 = samples[:, 0]  # First column corresponds to m1
    m2 = samples[:, 1]  # Second column corresponds to m2
    z = samples[:, 2]
    samples2 = np.vstack((m2, m1, z)).T
    # Combine both samples into one array
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
    KDE_slice = eval_kde3d.reshape(XX.shape)
    #if we need 3D output
    #kde_list.append(KDE_slice)

    Rate3D = Nev * KDE_slice / VT_3D
    kdeM1Xieff, kdeM2Xieff = get_rate_m_Xieff2D(m1grid, m2grid, KDE_slice)
    rateM1Xieff, rateM2Xieff = get_rate_m_Xieff2D(m1grid, m2grid, Rate3D)
    Ratem1m2 = integral_wrt_Xieff(KDE_slice, VT_3D, cfgrid, Nev)

    KDEM1Xieff.append(kdeM1Xieff)
    KDEM2Xieff.append(kdeM2Xieff)
    RateM1Xieff.append(rateM1Xieff)
    RateM2Xieff.append(rateM2Xieff)
    rate_m1m2IntXieff.append(Ratem1m2)

    savehfintegm1m2.create_dataset(f"ratem1m2_iter{ilabel}", data=Ratem1m2)
    savehfintegm1Xieff.create_dataset(f"rate_m1xieff_iter{ilabel}", data=rateM1Xieff)
    savehfintegm2Xieff.create_dataset(f"rate_m2xieff_iter{ilabel}", data=rateM2Xieff)

    savehfintegm1Xieff.create_dataset(f"kde_m1xieff_iter{ilabel}", data=kdeM1Xieff)
    savehfintegm2Xieff.create_dataset(f"kde_m2xieff_iter{ilabel}", data=kdeM2Xieff)

    # get oneD output
    rateM1, rateM2 = get_rate_m_oneD(m1grid, m2grid, Ratem1m2)
    ratem1_arr.append(rateM1)
    ratem2_arr.append(rateM2)

    savehfintegm2Xieff.create_dataset(f"rate_m1_iter{ilabel}", data=rateM1)
    savehfintegm2Xieff.create_dataset(f"rate_m2_iter{ilabel}", data=rateM2)

    savehfintegm1m2.flush()
    savehfintegm1Xieff.flush()
    savehfintegm2Xieff.flush()

savehfintegm1m2.close()
savehfintegm1Xieff.close()
savehfintegm2Xieff.close()

print('Making plots')

# Note I did not save and make plot for m1-m2KDE
iter_tag = f"iter{opts.start_iter}_{opts.end_iter}"
u_plot.get_averagem1m2_plot(med1, med2, M1, M2, rate_m1m2IntXieff, itertag=iter_tag, pathplot=opts.pathplot, plot_name='Rate')

u_plot.get_m_Xieff_plot(med1, med3, M, CF, KDEM1Xieff, itertag=iter_tag, pathplot=opts.pathplot, plot_name='KDE', xlabel='m_1')
u_plot.get_m_Xieff_plot(med2, med3, M, CF, KDEM2Xieff, itertag=iter_tag, pathplot=opts.pathplot, plot_name='KDE', xlabel='m_2')
u_plot.get_m_Xieff_plot(med1, med3, M, CF, RateM1Xieff, itertag=iter_tag, pathplot=opts.pathplot, plot_name='Rate', xlabel='m_1')
u_plot.get_m_Xieff_plot(med2, med3, M, CF, RateM2Xieff, itertag=iter_tag, pathplot=opts.pathplot, plot_name='Rate', xlabel='m_2')

# 1-d rate vs masses
u_plot.Rate_masses(m1grid, m2grid, ratem1_arr, ratem2_arr, tag=iter_tag, pathplot=opts.pathplot)

######offset Xieff plot ######################
m1_slice_values = np.array([10, 15, 20, 25, 35, 45, 55, 70])
m2_slice_values = m1_slice_values * 2./3.
u_plot.Xieff_offset_plot(m1grid, cfgrid, m1_slice_values, RateM1Xieff, offset_increment=5, m_label='m_1', tag=iter_tag, pathplot=opts.pathplot)
u_plot.Xieff_offset_plot(m1grid, cfgrid, m2_slice_values, RateM2Xieff, offset_increment=5, m_label='m_2', tag=iter_tag, pathplot=opts.pathplot)

