import matplotlib
sys.path.append('pop-de/popde/')
import density_estimate as d
import adaptive_kde as ad
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
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

outputfilename = 'TwoDsliceCasem1fixed_slice_m1_65KDEm1m2_Xieff_outputs_all1100Iterations.hdf5'

xmin, xmax = 5, 105  #m1
ymin, ymax = 5, 105 #m2
zmin, zmax = -1, 1 #dL
Npoints = 150 #opts.Npoints


m1_src_grid = 65#np.logspace(np.log10(xmin), np.log10(xmax), Npoints)
m2_src_grid = np.logspace(np.log10(ymin), np.log10(ymax), Npoints)
Xieff_grid = np.linspace(zmin, zmax, 100)  #we only need for fix m2 case

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
hdf_file = 'output_fileVT_min_bw_Xieff0.01_optimize_code_test.hdf5' # Your HDF5 file path
hdf = h5.File(hdf_file, 'r')

Total_Iterations = 1000
discard = 100

XX, YY, ZZ = np.meshgrid(m1_src_grid, m2_src_grid, Xieff_grid, indexing='ij')
#choose 2D grid as needed
# Create a meshgrid for (m1_src_grid, Xieff_grid)
#M1, XIEFF = np.meshgrid(m1_src_grid, Xieff_grid, indexing='ij')
M2, XIEFF = np.meshgrid(m2_src_grid, Xieff_grid, indexing='ij')
#M1, M2 = np.meshgrid(m1_src_grid, m2_src_grid, indexing='ij')
eval_samples = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])

#m2_fixed = 65#change m2 here to 35
#xi_fixed = 0.0

savehf = h5.File(output_filename, "w")
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
        savehf.create_dataset("kde_iter{0}".format(i), data=KDE_slice)
        savehf.flush()

savehf.close()
