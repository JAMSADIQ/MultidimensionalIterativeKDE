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
parser.add_argument('--parameter3', help='name of parameter which we use for y-axis for KDE [can be Xieff, dL]', default='dL')
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
parser.add_argument('--VTdata-filename', required=True, help='VToncoarsegrid use for interpolator', type=str)
parser.add_argument('--output-filename', default='m1m2mdL3Danalysis_slice_dLoutput-filename', help='write a proper name of output hdf files based on analysis', type=str)
opts = parser.parse_args()
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
zmin, zmax = -1, 1 #dL
Npoints = 100 #opts.Npoints

m1_src_grid = np.logspace(np.log10(xmin), np.log10(xmax), Npoints)
m2_src_grid = np.logspace(np.log10(ymin), np.log10(ymax), Npoints)
Xieff_grid = np.linspace(zmin, zmax, 60)  #we only need for fix m2 case

##Need to verify this
def IntegrateRm1m2Xieff_wrt_m2(m1val, m2val, Xieffval, Ratem1m2Xieff):
    # Initialize the output array (size: 150 x 100)
    ratem1_Xieff = np.zeros((len(m1val), len(Xieffval)))
    # Iterate over m1 indices
    for xid, m1 in enumerate(m1val):
        # Mask for valid m2 values where m2 <= m1
        m2_valid = m2val <= m1  # Boolean mask
        rate_vals = Ratem1m2_Xieff[xid, m2_valid, :]  # Shape: (valid_m2, 100)
        
        # Perform Simpson integration along the m2 axis
        if np.sum(m2_valid) > 1:  # Ensure there are at least two points for integration
            ratem1_Xieff[xid, :] = simpson(rate_vals, x=m2val[m2_valid], axis=0)
        else:
            ratem1_Xieff[xid, :] = 0  # If no valid points, set to zero

    return ratem1_Xieff

#File obtained with 3D iterative KDEs
hdf_file = opts.iterative_result_filename# Your HDF5 file path
hdf = h5.File(hdf_file, 'r')
Total_Iterations = int(opts.NIterations)
discard = int(opts.discard)

XX, YY, ZZ = np.meshgrid(m1_src_grid, m2_src_grid, Xieff_grid, indexing='ij')
# Create a meshgrid for (m1_src_grid, Xieff_grid)
M1, XIEFF = np.meshgrid(m1_src_grid, Xieff_grid, indexing='ij')
eval_samples = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])

#############VT file and its plot ##################
csv_file = opts.VTdata_filename #"/home/jxs1805/Research/SISSA/cbc_pdet/cbc_pdet/VT_values_progress.csv"
df = pd.read_csv(csv_file)
df['VT'] = df['VT'].astype(float)  # Convert VT to float
# **Step 2: Extract unique grid values**
m1_vals = np.sort(df['m1'].unique())
m2_vals = np.sort(df['m2'].unique())
Xieff_vals = np.sort(df['Xieff'].unique())
VT_values = np.empty((len(m1_vals), len(m2_vals), len(Xieff_vals)))
print(VT_values.shape)
# Fill the 3D array with VT values
for _, row in df.iterrows():
    i = np.where(m1_vals == row['m1'])[0][0]
    j = np.where(m2_vals == row['m2'])[0][0]
    k = np.where(Xieff_vals == row['Xieff'])[0][0]
    VT_values[i, j, k] = row['VT']/1e9 #Gpc3

# **Step 4: Interpolate using RegularGridInterpolator*
interp_VT = RegularGridInterpolator((m1_vals, m2_vals, Xieff_vals), VT_values, bounds_error=False, fill_value=np.nan) #nan can be an issue
m2_fixed = 10 #change m2 here to 35
query_points = np.array([M1.ravel(), np.full(M1.size, m2_fixed), XIEFF.ravel()]).T
VT_interpolated = interp_VT(query_points).reshape(M1.shape)

#we want to save data for each 100 iteration to avoid very huge HDF file 
savehf = h5.File(opts.output_filename + "Save3DKDEm1m2_Xieff_outputs.hdf5", "w")
savehf.create_dataset("M1mesh", data=XX)
savehf.create_dataset("M2mesh", data=YY)
savehf.create_dataset("Xieffmesh", data=ZZ)
kde_list = []
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
        kde_list.append(KDE_slice)
        if i > 0  and i%100 == 0:
            u_plot.get_m1Xieff_at_m2_slice_plot(medianlist1, medianlist3, m2_src_grid, m2_fixed, M1, XIEFF, kde_list[-100:], VT_interpolated, iterN=i, plot_name='KDE', pathplot='./')
        savehf.create_dataset("kde_iter{0}".format(i), data=KDE_slice)
        savehf.flush()

savehf.close()
