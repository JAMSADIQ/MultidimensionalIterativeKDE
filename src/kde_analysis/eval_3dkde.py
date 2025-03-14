import sys
import matplotlib
#### set the path below where KDE code repo is
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
import pandas as pd
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
parser.add_argument('--m1-max', default=105.0, type=float, help='Maximum value for primary mass m1.')
parser.add_argument('--Npoints', default=200, type=int, help='Number of points for KDE evaluation.')
parser.add_argument('--param2-min', default=4.95, type=float, help='Minimum value for parameter 2 if it is  m2, else if dL use 10')
parser.add_argument('--param2-max', default=105.0, type=float, help='Maximum value for parameter 2 if it is m2 else if dL  use 10000')
parser.add_argument('--param3-min', default=200., type=float, help='Minimum value for parameter 3 if it is  dL, else if Xieff use -1')
parser.add_argument('--param3-max', default=8000., type=float, help='Maximum value for parameter 3 if it is dL else if Xieff  use +1')

parser.add_argument('--discard', default=100, type=int, help=('discard first 100 iterations'))
parser.add_argument('--NIterations', default=1000, type=int, help='Total number of iterations for the reweighting process.')


#plots and saving data
parser.add_argument('--pathplot', default='./', help='public_html path for plots', type=str)
parser.add_argument('--iterative-result-filename', required=True, help='write a proper name of output hdf files based on analysis', type=str)
parser.add_argument('--output-filename', default='m1_m2_chief_kde_eval_output-filename', help='write a proper name of output hdf files based on analysis', type=str)
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

mean_m1 = np.array(medianlist1)
mean_m2 = np.array(medianlist2)
mean_chieff = np.array(medianlist3)

################## To get KDE data on m1-Xieff or m2-Xieff wit integation over one of the mass
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

##############2D m-Xieff plots
def get_mXieff_plot(medianlist_m, medianlist_xieff, M, XIEFF, KDElist, iterN=1, mass_tag='m_1', pathplot='./', plot_name='KDE'):
    data_slice = np.percentile(KDElist, 50, axis=0)
    colorbar_label = r'$p('+mass_tag+', \chi_\mathrm{eff})$'
    max_density = np.nanmax(data_slice)
    max_exp = np.floor(np.log10(max_density))  # Find the highest power of 10 below max_density
    contourlevels = 10 ** (max_exp - np.arange(4))[::-1]
    contourlevels[-1] = max_density
    print("contourlevels =", contourlevels)
    plt.figure(figsize=(8, 6))
    norm_val = LogNorm(vmin=contourlevels[0], vmax=max_density)  # Apply log normalization
    pcm = plt.pcolormesh(M, XIEFF, data_slice, cmap='Purples', norm=norm_val, shading='auto')
    contours = plt.contour(M,  XIEFF, data_slice, levels=contourlevels, colors='black')

    # Colorbar
    cbar = plt.colorbar(pcm, label=colorbar_label)
    plt.scatter(medianlist_m, medianlist_xieff, color='r', marker='+', s=20)
    plt.ylabel(r"$\chi_\mathrm{effective}$")
    plt.xlabel(r"$m \,[M_\odot]$")
    plt.semilogx()
    plt.tight_layout()
    plt.savefig(pathplot+"Average"+plot_name+mass_tag+"Xieff_Iter{0}.png".format(iterN))
    #plt.close()
    plt.show()
    return 0


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
Npoints = opts.Npoints
chi_points = 100


m1_src_grid = np.logspace(np.log10(5), np.log10(105), Npoints)
m2_src_grid = np.logspace(np.log10(5), np.log10(105), Npoints)
Xieff_grid = np.linspace(-1, 1, chi_points)
#
M, XIEFF = np.meshgrid(m1_src_grid, Xieff_grid, indexing='ij')
hd2Dm2Xieff = h5.File(opts.output_filename+"integrate_m2_data_data.hdf5", "w")
hd2Dm1Xieff = h5.File(opts.output_filename+"integrate_m1_data.hdf5", "w")
#savehf = h5.File(opts.output_filename+"Full3D_data.hdf5", "w")

hdf_file = opts.iterative_result_filename# Your HDF5 file path
hdf = h5.File(hdf_file, 'r')
Total_Iterations = int(opts.NIterations)
discard = int(opts.discard)

XX, YY, ZZ = np.meshgrid(m1_src_grid, m2_src_grid, Xieff_grid, indexing='ij')
eval_samples = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])

hd2Dm2Xieff.create_dataset("Mmesh", data=M)
hd2Dm2Xieff.create_dataset("Xieffmesh", data=XIEFF)

hd2Dm1Xieff.create_dataset("Mmesh", data=M)
hd2Dm1Xieff.create_dataset("Xieffmesh", data=XIEFF)


#kde_list = []
kde_m1chieff = []
kde_m2chieff = []
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
        KDE_3d = eval_kde3d.reshape(XX.shape)
        #kde_list.append(KDE_3d)
        # do we want to save this?no
        #savehf.create_dataset("kde_iter{0}".format(i), data=KDE_slice)
        #savehf.flush()
#savehf.close()
        #integrate w r t m1
        # Mask where m1 < m2 (set to zero)
        mask_m1_lt_m2 = XX < YY
        KDE3D_m1_integrate = np.where(mask_m1_lt_m2, 0, KDE_3d)
        # Integrate over m1 where m1 >= m2
        integrated_m1 = simpson(KDE3D_m1_integrate, x=m1_src_grid, axis=0)
        hd2Dm2Xieff.create_dataset("kde_iter{0}".format(i), data=integrated_m1)
        kde_m2chieff.append(integrated_m1)

        #integaret w r t m2
        # Mask where m2 > m1 (set to zero)
        mask_m2_gt_m1 = YY > XX
        KDE3D_m2_integrate = np.where(mask_m2_gt_m1, 0, KDE_3d)
        # Integrate over m2 where m2 <= m1
        integrated_m2 = simpson(KDE3D_m2_integrate, x=m2_src_grid, axis=1)
        hd2Dm1Xieff.create_dataset("kde_iter{0}".format(i), data=integrated_m2)
        kde_m1chieff.append(integrated_m2)
        if (i+1) % discard==0:
            get_mXieff_plot(mean_m2, mean_chieff, M, XIEFF, kde_m2chieff[-discard:], iterN=i, mass_tag='m_1', pathplot=opts.pathplot, plot_name='KDE_Int_m1')
            get_mXieff_plot(mean_m1, mean_chieff, M, XIEFF, kde_m1chieff[-discard:], iterN=i, mass_tag='m_2', pathplot=opts.pathplot, plot_name='KDE_Int_m2')
        hd2Dm1Xieff.flush()
        hd2Dm2Xieff.flush()

hd2Dm1Xieff.close()
hd2Dm2Xieff.close()

get_mXieff_plot(mean_m2, mean_chieff, M, XIEFF, kde_m2chieff[discard:], iterN=1001, mass_tag='m_1', pathplot=opts.pathplot, plot_name='KDE_Int_m1')
get_mXieff_plot(mean_m1, mean_chieff, M, XIEFF, kde_m1chieff[discard:], iterN=1001, mass_tag='m_2', pathplot=opts.pathplot, plot_name='KDE_Int_m2')

