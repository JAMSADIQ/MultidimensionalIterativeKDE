import sys
sys.path.append('pop-de/popde/')
import density_estimate as d
import adaptive_kde as ad
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import h5py as h5
from scipy.integrate import quad, simpson
from scipy.interpolate import RegularGridInterpolator
from matplotlib import rcParams
from matplotlib.colors import LogNorm, Normalize
import utils_plot as u_plot

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
parser.add_argument('--parameter1', help='name of parameter which we use for x-axis for KDE', default='m1')
parser.add_argument('--parameter2', help='name of parameter which we use for y-axis for KDE: m2', default='m2')
parser.add_argument('--parameter3', help='name of parameter which we use for y-axis for KDE [can be Xieff, dL]', default='Xieff')
parser.add_argument('--m1-min', default=5.0, type=float, help='Minimum value for primary mass m1.')
parser.add_argument('--m1-max', default=100.0, type=float, help='Maximum value for primary mass m1.')
parser.add_argument('--Npoints-masses', default=150, type=int, help='Number of points for KDE evaluation.')
parser.add_argument('--Npoints-param3', default=100, type=int, help='Number of points for KDE evaluation.')
parser.add_argument('--param2-min', default=4.95, type=float, help='Minimum value for parameter 2 if it is  m2, else if dL use 10')
parser.add_argument('--param2-max', default=100.0, type=float, help='Maximum value for parameter 2 if it is m2 else if dL  use 10000')
parser.add_argument('--param3-min', default=-1., type=float, help='Minimum value for parameter 3 if it is  dL, use 500 else if Xieff use -1')
parser.add_argument('--param3-max', default=1., type=float, help='Maximum value for parameter 3 if it is dL use 8000 else if Xieff  use +1')

parser.add_argument('--discard', default=100, type=int, help=('discard first 100 iterations'))
parser.add_argument('--NIterations', default=1000, type=int, help='Total number of iterations for the reweighting process.')


#plots and saving data
parser.add_argument('--pathplot', default='./', help='public_html path for plots', type=str)
parser.add_argument('--iterative-result-filename', required=True, help='write a proper name of output hdf files based on analysis', type=str)
parser.add_argument('--VTdata-filename', required=True, help='VToncoarsegrid use for interpolator', type=str)
parser.add_argument('--output-filename', default='m1m2mdL3Danalysis_slice_dLoutput-filename', help='write a proper name of output hdf files based on analysis', type=str)
opts = parser.parse_args()

#################Integration functions######################
def integral_wrt_Xieff(KDE3D, VT3D, Xieff_grid, Nevents=69):
    """
    KDE3d and VT3d are computed with indexing ='ij' way
    """
    Rate3D = Nevents*KDE3D/VT3D
    integm1m2 = simpson(Rate3D, x=Xieff_grid, axis=2)
    return integm1m2


def get_rate_m_oneD(m1_query, m2_query, Rate):
    ratem1 = np.zeros(len(m1_query))
    ratem2 = np.zeros(len(m2_query))
    for xid, m1 in enumerate(m1_query):
        y_valid = m2_query <= m1_query[xid]  # Only accept points with y <= x
        rate_vals = Rate[y_valid, xid]
        #print(rate_vals, m2_query[y_valid])
        ratem1[xid] = simpson(rate_vals, m2_query[y_valid])
    for yid, m2 in enumerate(m2_query):
        x_valid = m1_query >= m2_query[yid]  # Only accept points with y <=
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
f1 = h5.File(opts.datafilename1, 'r')#m1
#f1 = h5.File('Final_noncosmo_GWTC3_m1srcdatafile.h5', 'r')#m1
d1 = f1['randdata']
#f2 = h5.File('Final_noncosmo_GWTC3_m2srcdatafile.h5', 'r')#m2
f2 = h5.File(opts.datafilename2, 'r')#m2
d2 = f2['randdata']
#f3 = h5.File('Final_noncosmo_GWTC3_xieff_datafile.h5', 'r')#dL
f3 = h5.File(opts.datafilename3, 'r')#Xieff
d3 = f3['randdata']
medianlist1 = f1['initialdata/original_mean'][...]
sampleslists2 = []
medianlist2 = f2['initialdata/original_mean'][...]
medianlist3 = f3['initialdata/original_mean'][...]

f1.close()
f2.close()
f3.close()

meanxi1 = np.array(medianlist1)
meanxi2 = np.array(medianlist2)
meanxi3 = np.array(medianlist3)

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
zmin, zmax = -1, 1 #Xieff
Mpoints = opts.Npoints_masses #150
Xipoints = opts.Npoints_param3 #100

m1_src_grid = np.logspace(np.log10(5), np.log10(105), Mpoints)
m2_src_grid = np.logspace(np.log10(5), np.log10(105), Mpoints)
Xieff_grid = np.linspace(-1, 1, Xipoints)

######## Iterative result files must have 'iteration_{i} format data see below
hdf_file = opts.iterative_result_filename #
hdf = h5.File(hdf_file, 'r')
###### KDE eval 3D grid #########################
XX, YY, ZZ = np.meshgrid(m1_src_grid, m2_src_grid, Xieff_grid, indexing='ij')
eval_samples = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
######## For 2D plots ##############################
M, XIEFF = np.meshgrid(m1_src_grid, Xieff_grid, indexing='ij')
M1, M2 = np.meshgrid(m1_src_grid, m2_src_grid, indexing='ij')
#################### VT data ###########################
vt_file = opts.VTdata_filename #
VTdata = h5.File(vt_file, 'r')
m1vals = VTdata['m1'][...]
m2vals = VTdata['m2'][...]
Xieffvals = VTdata['Xieff'][...]
VT_3D = VTdata['VT'][...]/1e9     #Gpc3^ need 1/1e9
VTdata.close()
#####In case Interpolation is needed not using can be issue ####################
interp_VT = RegularGridInterpolator((m1_vals, m2_vals, Xieff_vals), VT_values, bounds_error=False, fill_value=np.nan) #nan can be an issue
threeDgrid = np.array([XX.ravel(), YY.ravel(), ZZ.ravel()]).T
#VT_3Dinterp = interp_VT(threeDgrid) #indexing ij can be issue here

############ Saving data in 3 files #################################
savehfintegm1m2 = h5.File(opts.output_filename+"CHI_PRIORsamples2Dm1m2_IntXieffdata1100Iterations.hdf5", "w")
savehfintegm1Xieff = h5.File(opts.output_filename+"CHI_PRIORsamples2Dm1Xieff_Intdata1100Iterations.hdf5", "w")
savehfintegm2Xieff = h5.File(opts.output_filename+"CHI_PRIORsamples2Dm1Xieff_m2Xieff_data1100Iterations.hdf5", "w")

savehfintegm1m2.create_dataset("M1mesh", data=M1)
savehfintegm1m2.create_dataset("M2mesh", data=M2)
savehfintegm1Xieff.create_dataset("Mmesh", data=M)
savehfintegm1Xieff.create_dataset("XIEFFmesh", data=XIEFF)
savehfintegm2Xieff.create_dataset("Mmesh", data=M)
savehfintegm2Xieff.create_dataset("XIEFFmesh", data=XIEFF)

rate_m1m2IntXieff_list = []
ratem1_arr= []#np.zeros(shape=[900, len(m1_src_grid)])
ratem2_arr =[] #np.zeros(shape=[900, len(m2_src_grid)])

KDEM1Xieff = []
KDEM2Xieff = []
RateM1Xieff = []
RateM2Xieff = []
kde3d_list = [] #if needed
###############################Iterations and evaluating KDEs/Rate
Total_Iterations = int(opts.NIterations)
discard = int(opts.discard)
for i in range(Total_Iterations+discard): #fix this as this can change
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
        KDE_slice = eval_kde3d.reshape(XX.shape)
        Rate3D  = 69*KDE_slice/VT_3D
        kdeM1Xieff, kdeM2Xieff = get_rate_m_Xieff2D(m1_src_grid, m2_src_grid, KDE_slice)
        rateM1Xieff, rateM2Xieff = get_rate_m_Xieff2D(m1_src_grid, m2_src_grid, Rate3D)
        KDEM1Xieff.append(kdeM1Xieff)
        KDEM2Xieff.append(kdeM2Xieff)
        RateM1Xieff.append(rateM1Xieff)
        RateM2Xieff.append(rateM2Xieff)
        Ratem1m2 = integral_wrt_Xieff(KDE_slice, VT_3D, Xieff_grid)
        rate_m1m2IntXieff_list.append(Ratem1m2)
        #I am not saving kde but rates only change it if u need kde
        savehfintegm1m2.create_dataset("ratem1m2_iter{0}".format(i), data=Ratem1m2)
        savehfintegm1Xieff.create_dataset("rate_m1xieff_iter{0}".format(i), data=rateM1Xieff)
        savehfintegm2Xieff.create_dataset("rate_m2xieff_iter{0}".format(i), data=rateM2Xieff)

        #get oneD output
        rateM1, rateM2 = get_rate_m_oneD(m1_src_grid, m2_src_grid, Ratem1m2)
        ratem1_arr.append(rateM1)
        ratem2_arr.append(rateM2)
        #if we need 3D output
        #kde_list.append(KDE_slice)
        savehfintegm2Xieff.create_dataset("rate_m1_iter{0}".format(i), data=rateM1)
        savehfintegm2Xieff.create_dataset("rate_m2_iter{0}".format(i), data=raterateM2)
        savehfintegm1m2.flush()
        savehfintegm1Xieff.flush()
        savehfintegm2Xieff.flush()

savehfintegm1m2.close()
savehfintegm1Xieff.close()
savehfintegm2Xieff.close()
#Plots only for final iterations if needed one can add this inside loop 

#Note I did not save and make plot for m1-m2KDE
u_plot.get_averagem1m2_plot(meanxi1, meanxi2, M1, M2, rate_m1m2IntXieff_list[100:], iterN=1, pathplot='./', plot_name='Rate')

u_plot.get_m_Xieff_plot(meanxi1, meanxi3, M, XIEFF, KDEM1Xieff[discard:], iterN=1, pathplot='./', plot_name='KDE', xlabel = 'm_1')
u_plot.get_m_Xieff_plot(meanxi2, meanxi3, M, XIEFF, KDEM2Xieff[discard:], iterN=1, pathplot='./', plot_name='KDE', xlabel = 'm_2')
u_plot.get_m_Xieff_plot(meanxi1, meanxi3, M, XIEFF, RateM1Xieff[discard:], iterN=1, pathplot='./', plot_name='Rate', xlabel = 'm_1')
u_plot.get_m_Xieff_plot(meanxi2, meanxi3, M, XIEFF, RateM2Xieff[discard:], iterN=1, pathplot='./', plot_name='Rate', xlabel = 'm_2')

#One D plot and Offset will go here
u_plot.Rate_masses(m1_src_grid, m2_src_grid, ratem1_arr, ratem2_arr, pathplot='./')
######offset Xieff plot ######################
m_slice_values = [10, 15, 20, 25, 35, 45, 55, 70]
u_plot.Xieff_offset_plot(m1_src_grid, Xieff_grid, m_slice_values, RateM1Xieff[discard:], offset_increment=5, m_label='m_1', pathplot='./')
u_plot.Xieff_offset_plot(m1_src_grid, Xieff_grid, m_slice_values, RateM2Xieff[discard:], offset_increment=5, m_label='m_2', pathplot='./')

    



