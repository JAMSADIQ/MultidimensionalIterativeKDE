import sys
sys.path.append('../../pop-de/popde/')
import adaptive_kde as ad
import numpy as np
import h5py as h5

input_hdf = sys.argv[1]
m1 = float(sys.argv[2])

# Output file
savehf = h5.File(f'kde_2d_slice_m1_{m1}_m1m2chieff_1000iter.hdf5', 'w')

xmin, xmax = 5, 105  # m1
ymin, ymax = 5, 105  # m2
zmin, zmax = -0.99, 0.99 # chieff


m1_src_grid = m1  #np.logspace(np.log10(xmin), np.log10(xmax), Npoints)
m2_src_grid = np.logspace(np.log10(ymin), np.log10(ymax), 150)
z_grid = np.linspace(zmin, zmax, 100)

# TD : This function is not used in the rest of the script
#def IntegrateRm1m2Xieff_wrt_m2(m1val, m2val, Xieffval, Ratem1m2Xieff):
#    # Initialize the output array (size: 150 x 100)
#    ratem1_Xieff = np.zeros((len(m1val), len(Xieffval)))
#    # Iterate over m1 indices
#    for xid, m1 in enumerate(m1val):
#        # Mask for valid m2 values where m2 <= m1
#        m2_valid = m2val <= m1  # Boolean mask
#        rate_vals = Ratem1m2_Xieff[xid, m2_valid, :]  # Shape: (valid_m2, 100)
#        
#        # Perform Simpson integration along the m2 axis
#        if np.sum(m2_valid) > 1:  # Ensure there are at least two points for integration
#            ratem1_Xieff[xid, :] = simpson(rate_vals, x=m2val[m2_valid], axis=0)
#        else:
#            ratem1_Xieff[xid, :] = 0  # If no valid points, set to zero
#
#    return ratem1_Xieff

#File obtained with 3D iterative KDEs
hdf = h5.File(input_hdf, 'r')

valid_iter = 1000
discard = 100

XX, YY, ZZ = np.meshgrid(m1_src_grid, m2_src_grid, z_grid, indexing='ij')
#choose 2D grid as needed
# Create a meshgrid for (m1_src_grid, Xieff_grid)
#M1, XIEFF = np.meshgrid(m1_src_grid, Xieff_grid, indexing='ij')
#M2, XIEFF = np.meshgrid(m2_src_grid, z_grid, indexing='ij')
#M1, M2 = np.meshgrid(m1_src_grid, m2_src_grid, indexing='ij')
eval_samples = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])

#m2_fixed = 65#change m2 here to 35
#xi_fixed = 0.0

savehf.create_dataset("M1mesh", data=XX)
savehf.create_dataset("M2mesh", data=YY)
savehf.create_dataset("Xieffmesh", data=ZZ)
kde_list = []

for i in range(discard, valid_iter+discard):
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
        m1 = samples[:, 0]
        m2 = samples[:, 1]
        z = samples[:, 2]
        samples2 = np.vstack((m2, m1, z)).T
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
        if i % 100 == 0: print(i)

savehf.close()
