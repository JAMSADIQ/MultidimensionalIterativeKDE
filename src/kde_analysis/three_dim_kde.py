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
#rcParams["grid.linestyle"] = ':'
rcParams["grid.alpha"] = 0.6


import utils_plot as u_plot
import o123_class_found_inj_general as u_pdet
#careful parsers 
parser = argparse.ArgumentParser(description=__doc__)

# use code get_gwtc_data_samples.py in bin directory
parser.add_argument('--datafilename1', help='h5 file containing N samples for m1for all gw bbh event')
parser.add_argument('--datafilename2', help='h5  file containing N sample of parameter2 for each event, ')
parser.add_argument('--datafilename3', help='h5  file containing N sample of redshift for each event, ')
parser.add_argument('--parameter1', help='name of parameter which we use for x-axis for KDE', default='m1')
parser.add_argument('--parameter2', help='name of parameter which we use for y-axis for KDE [can be m2, Xieff, DL]', default='m2')
parser.add_argument('--parameter3', help='name of parameter which we use for y-axis for KDE [can be m2, Xieff, DL]', default='DL')
### 
parser.add_argument('--dl-prior-power', type=float, default=2.0, help='If set, perform KDE in logarithmic space.')
parser.add_argument('--redshift-prior-power ', type=float, default=2.0, help='If set, perform KDE in logarithmic space.')

# limits on KDE evulation: 
parser.add_argument('--m1-min', help='minimum value for primary mass m1', type=float)
parser.add_argument('--m1-max', help='maximum value for primary mass m1', type=float)
parser.add_argument('--Npoints', default=100, type=int, help='Total points on which to evaluate KDE')
#m2-min must be <= m1-min
parser.add_argument('--param2-min', default=2.95, type=float, help='minimum value of m2 ,chieff =-1, DL= 1Mpc, used, must be below m1-min')
parser.add_argument('--param2-max', default=100.0, type=float, help='max value of m2 used, could be m1-max for chieff +1, for DL 10000Mpc')
parser.add_argument('--param3-min', default=10, type=float, help='minimum value of m2 ,chieff =-1, DL= 1Mpc, used, must be below m1-min')
parser.add_argument('--param3-max', default=10000.0, type=float, help='max value of m2 used, could be m1-max for chieff +1, for DL 10000Mpc')
parser.add_argument('--Maxpdet', default=0.1, type=float, help='capping for small pdet to regularization')
parser.add_argument('--MaxbwdL', default=0.1, type=float, help='To avoid bias in dL dim use ax bw in this direction: rescale factor is 1/bw that we will code')
# analysis on mock data or gw data.
parser.add_argument('--type-data', choices=['gw_pe_samples', 'mock_data'], help='mock data for some power law with gaussian peak or gwtc  pe samples data. h5 files for two containing data for median and sigma for m1')
#only for  mock data we will need this 
parser.add_argument('--power-alpha', default=0.0, help='power in power law sample (other than gaussian samples) in true distribution', type=float)
parser.add_argument('--fp-gauss', default=0.6, help='fraction of gaussian sample in true distribution', type=float)
parser.add_argument('--fpopchoice', default='kde', help='choice of fpop to be rate or kde', type=str)
parser.add_argument('--mockdataerror', default='fixed', help='mockdata error = 5 if fixed otherwise use np.random.randint(minval, maxval)', type=str)

#EMalgorithm reweighting 
parser.add_argument('--reweightmethod', default='bufferkdevals', help='Only for gaussian sample shift method: we can reweight samples via buffered kdevals(bufferkdevals) or buffered kdeobjects (bufferkdeobject)', type=str)
parser.add_argument('--reweight-sample-option', default='reweight', help='choose either "noreweight" or "reweight" if reweight use fpop prob to get reweight sample (one sample for no bootstrap or no or multiple samples for poisson)', type=str)
parser.add_argument('--bootstrap-option', default='poisson', help='choose either "poisson" or "nopoisson" if None it will reweight based on fpop prob with single reweight sample for eaxh event', type=str)

#### buffer iteratio
parser.add_argument('--buffer-start', default=100, type=int, help='start of buffer in reweighting.')
parser.add_argument('--buffer-interval', default=100, type=int, help='interval of buffer that choose how many previous iteration resulkts we use in next iteration for reweighting.')
parser.add_argument('--NIterations', default=1000, type=int, help='Total Iterations in reweighting')

#plots and saving data
parser.add_argument('--pathplot', default='./', help='public_html path for plots', type=str)
parser.add_argument('--pathtag', default='re-weight-bootstrap_', help='public_html path for plots', type=str)
parser.add_argument('--output-filename', default='MassRedshift_with_reweight_output_data_', help='write a proper name of output hdf files based on analysis', type=str)
opts = parser.parse_args()

maxRescale_dLdim = round(1.0/ opts.MaxbwdL)
print(f'max rescal factor in dL dim = {maxRescale_dLdim}')
#############for ln paramereter we need these

H0 = 67.9  # km/s/Mpc
omega_m = 0.3065
cosmo = FlatLambdaCDM(H0=H0, Om0=omega_m)

def get_massed_indetector_frame(dLMpc, mass):
    zcosmo = z_at_value(cosmo.luminosity_distance, dLMpc*u.Mpc).value
    mdet = mass*(1.0 + zcosmo)
    return mdet

def prior_factor_function(samples, redshiftvals):
    """ 
    LVC use uniform-priors for masses 
    in linear scale. In reweighting we need to take 
    account for the prior factor if use masses 
    in log scale
    There is dL prior as well as masses 
    if they are in source frame
    if we use non-cosmo_files we need 
    dL^3 factor 
    Compute a prior factor for reweighting
    for dL and masses from redshift to source frame

    Args:
        samples (np.ndarray): Array of samples with shape (N, 2), where N is the number of samples.
        redshifts:  given (dL|mass)  one D list or array
             redshift fatcor for  source mass prior
    Returns:
        np.ndarray: Prior factor for each sample.
    """
    m1val, m2val , dLval = samples[:, 0], samples[:, 1], samples[:, 2]
    # log-masses handle in the input_transf

    dL_prior_factor = (dLval)**opts.dl_prior_power  
    redshift_prior_factor = (1. + redshift)**opts.redshift_prior_power

    return 1.0/(dL_prior_factor * redshift_prior_factor)


def get_random_sample(original_samples, bootstrap='poisson'):
    """without reweighting"""
    rng = np.random.default_rng()
    if bootstrap =='poisson':
        reweighted_sample = rng.choice(original_samples, np.random.poisson(1))
    else:
        reweighted_sample = rng.choice(original_samples)
    return reweighted_sample



def apply_max_cap_function(pdet_list, max_pdet_cap=opts.Maxpdet):
  """Applies the min(10, 1/pdet) or  max(0.1, pdet)
  function to each element in the given list.
  Returns:
    A new list containing the results of applying the function to each element.
  """

  result = []
  for pdet in pdet_list:
    result.append(max(max_pdet_cap, pdet))
  return np.array(result)

def get_reweighted_sample(original_samples, redshiftvals, pdet_vals, fpop_kde, bootstrap='poisson', prior_factor=prior_factor_function):
    """
    inputs 
    original_samples: list of PE sample of an event
    redshiftvals = for prior factor
    pdet_vals = pdet on PE samples redshiftvals
    fpop_kde: kde_object [GaussianKDE(opt alpha, opt bw and Cov=True)]
    kwargs:
    bootstrap: [poisson or no-poisson] from araparser option
    prior_factor: for ln parameter in we need to handle non uniform prior
    return: reweighted_sample 
    use in np.random.choice  on kwarg 
    """
    fkde_samples = fpop_kde.evaluate_with_transf(original_samples) / apply_max_cap_function(pdet_vals)

    # Adjust probabilities based on the prior factor
    frate_atsample = fkde_samples * prior_factor(original_samples, redshiftvals) 
    #Normalize :sum=1
    fpop_at_samples = frate_atsample/frate_atsample.sum()

    rng = np.random.default_rng()
    if bootstrap =='poisson':
        reweighted_sample = rng.choice(original_samples, np.random.poisson(1), p=fpop_at_samples)
    else:
        reweighted_sample = rng.choice(original_samples, p=fpop_at_samples)

    return reweighted_sample


def New_median_bufferkdelist_reweighted_samples(sample, redshiftvals, pdet_vals, meanKDEevent, bootstrap_choice='poisson', prior_factor=prior_factor_function):
    """
    sample: PE samples of an event
    pdet_vals: pdet_vals of PE samples
    meanKDEevent: using mean of KDEs on PE samples for previous 100 iterations
    for each iteration we get KDE on each sample, and than take mean for past 100 iteration so each point in mean KDE is mean value of KDE evaluated on a sample in previous 100 iterations
    kwargs:
    bootstrap: [poisson or no-poisson] from araparser option
    prior_factor: for ln parameter in we need to handle non uniform prior
    return: reweighted_sample 
    """
    kde_by_pdet = meanKDEevent/apply_max_cap_function(pdet_vals)
    # Adjust probabilities based on the prior factor
    kde_by_pdet  *= prior_factor(sample, redshiftvals)
    norm_mediankdevals = kde_by_pdet/sum(kde_by_pdet)
    rng = np.random.default_rng()
    if bootstrap_choice =='poisson':
        reweighted_sample = rng.choice(sample, np.random.poisson(1), p=norm_mediankdevals)
    else:
        reweighted_sample = rng.choice(sample, p=norm_mediankdevals)
    return reweighted_sample

#######################################################################
#To filter the large problematic dL 
injection_file = "endo3_bbhpop-LIGO-T2100113-v12.hdf5"
with h5.File(injection_file, 'r') as f:
    T_obs = f.attrs['analysis_time_s']/(365.25*24*3600) # years
    N_draw = f.attrs['total_generated']

    m1 = f['injections/mass1_source'][:]
    m2 = f['injections/mass2_source'][:]
    s1x = f['injections/spin1x'][:]
    s1y = f['injections/spin1y'][:]
    s1z = f['injections/spin1z'][:]
    s2x = f['injections/spin2x'][:]
    s2y = f['injections/spin2y'][:]
    s2z = f['injections/spin2z'][:]
    z = f['injections/redshift'][:]
    dLp = f["injections/distance"][:]
    m1_det = m1#*(1.0 +  z)
    p_draw = f['injections/sampling_pdf'][:]
    pastro_pycbc_bbh = f['injections/pastro_pycbc_bbh'][:]


# Calculate min and max for dL
min_dLp, max_dLp = min(dLp), max(dLp)
#####################################
#We want to get PDET 
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

#If we want to compute pdet as we dont have pdet file alreadt use below line and comment the next after it  "w"(if want new pdet file)  versus "r"(if have file)
fz = h5.File('Final_noncosmo_GWTC3_redshift_datafile.h5', 'r')
dz = fz['randdata']
f1 = h5.File('Final_noncosmo_GWTC3_m1srcdatafile.h5', 'r')#m1
d1 = f1['randdata']
f2 = h5.File('Final_noncosmo_GWTC3_m2srcdatafile.h5', 'r')#m2
d2 = f2['randdata']
f3 = h5.File('Final_noncosmo_GWTC3_dL_datafile.h5', 'r')#dL
d3 = f3['randdata']
print(d1.keys())
sampleslists1 = []
medianlist1 = []
eventlist = []
sampleslists2 = []
medianlist2 = []
sampleslists3 = []
medianlist3 = []
pdetlists = []
redshift_lists = []
for k in d1.keys():
    eventlist.append(k)
    if (k  == 'GW190719_215514_mixed-nocosmo' or k == 'GW190805_211137_mixed-nocosmo'):
        print(k)
        m1_values = d1[k][...]
        #we need detector frame mass for PDET
        m1det_values = d1[k][...]*(1.0 + dz[k][...])
        m2_values = d2[k][...]
        m2det_values = d2[k][...]*(1.0 + dz[k][...])
        d_Lvalues = d3[k][...]
        dL_indices = [i for i, dL in enumerate(d_Lvalues) if (dL < min_dLp  or dL > max_dLp)]
        m1_values = [m for i, m in enumerate(m1_values) if i not in  dL_indices]
        m1det_values = [m for i, m in enumerate(m1det_values) if i not in  dL_indices]
        m2_values = [m for i, m in enumerate(m2_values) if i not in  dL_indices]
        m2det_values = [m for i, m in enumerate(m2det_values) if i not in  dL_indices]
        d_Lvalues = [dL for i, dL in enumerate(d_Lvalues) if i not in dL_indices]
        pdet_values =  u_pdet.get_pdet_m1m2dL(np.array(m1det_values), np.array(m2det_values), np.array(d_Lvalues), classcall=g)
        #get bad PDET bad out
        pdetminIndex = np.where(np.array(pdet_values) < 5e-4)[0]
        m1_values = np.delete(m1_values, pdetminIndex).tolist()
        m2_values = np.delete(m2_values, pdetminIndex).tolist()
        m1det_values = np.delete(m1det_values, pdetminIndex).tolist()
        m2det_values = np.delete(m2det_values, pdetminIndex).tolist()
        d_Lvalues = np.delete(d_Lvalues, pdetminIndex).tolist()
        pdet_values = np.delete(pdet_values, pdetminIndex).tolist()
    else:
        m1_values = d1[k][...]
        m1det_values = d1[k][...]*(1.0 + dz[k][...])
        m2_values = d2[k][...]
        m2det_values = d2[k][...]*(1.0 + dz[k][...])
        d_Lvalues = d3[k][...]
        pdet_values = u_pdet.get_pdet_m1m2dL(np.array(m1det_values), np.array(m2det_values),np.array(d_Lvalues), classcall=g)    

    pdetlists.append(pdet_values)
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


meanxi1 = np.array(medianlist1)
meanxi2 = np.array(medianlist2)
meanxi3 = np.array(medianlist3)
flat_samples1 = np.concatenate(sampleslists1).flatten()
flat_samples2 = np.concatenate(sampleslists2).flatten()
flat_samples3 = np.concatenate(sampleslists3).flatten()
flat_pdetlist = np.concatenate(pdetlists).flatten()
print("min max m1 =", np.min(flat_samples1), np.max(flat_samples1))
print("min max m2 =", np.min(flat_samples2), np.max(flat_samples2))
print("min max dL =", np.min(flat_samples3), np.max(flat_samples3))

# Create the scatter plot for pdet
plt.figure(figsize=(8,6))
plt.scatter(flat_samples1, flat_samples3, c=flat_pdetlist, cmap='viridis', norm=LogNorm())
cbar = plt.colorbar(label=r'$p_\mathrm{det}$')
cbar.set_label(r'$p_\mathrm{det}$', fontsize=20)
#plt.xlim(min(flat_samples1), max(flat_samples1))
#plt.ylim(min(flat_samples2), max(flat_samples2))
plt.xlabel(r'$m_{1, source} [M_\odot]$', fontsize=20)
plt.ylabel(r'$d_L [Mpc]$', fontsize=20)
plt.loglog()
plt.title(r'$p_\mathrm{det}$', fontsize=20)
plt.tight_layout()
plt.savefig(opts.pathplot+"pdetscatter.png")
plt.close()
############### We need plotting  work in progress
# Create the scatter plot for pdet 
from mpl_toolkits.mplot3d import Axes3D
# 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
# Plot the data with logarithmic color scaling
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(
    flat_samples1, 
    flat_samples2, 
    flat_samples3, 
    c=flat_pdetlist, 
    cmap='viridis', 
    s=10, 
    norm=LogNorm()
)
plt.colorbar(sc, label=r'$p_\mathrm{det}(m_1, m_2, d_L)$')

# Set axis labels and limits
ax.set_xlabel(r'$m_{1, source} [M_\odot]$', fontsize=20)
ax.set_ylabel(r'$m_{2, source} [M_\odot]$', fontsize=20)
ax.set_zlabel(r'$d_L [Mpc]$', fontsize=20)
# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(opts.pathplot + "pdet3Dscatter.png")
plt.close()
##########################################
sampleslists = np.vstack((flat_samples1, flat_samples2, flat_samples3)).T
sample = np.vstack((meanxi1, meanxi2, meanxi3)).T
print(sampleslists.shape)
sample = np.vstack((meanxi1, meanxi2, meanxi3)).T
######################################################
if opts.m1_min is not None and opts.m1_max is not None:
    xmin, xmax = opts.m1_min, opts.m1_max
else:
    xmin, xmax = np.min(flat_samples1), np.max(flat_samples1)
#if  flat_samples1 is list of arrays
#xmin = min(a.min() for a in flat_samples1)
#xmax = max(a.max() for a in flat_samples1)

if opts.param2_min is not None and opts.param2_max is not None:
    ymin, ymax = opts.param2_min, opts.param2_max
else:
    ymin, ymax = np.min(flat_samples2) , np.max(flat_samples2)

if opts.param3_min is not None and opts.param3_max is not None:
    zmin, zmax = opts.param3_min, opts.param3_max
else:
    zmin, zmax = np.min(flat_samples3) , np.max(flat_samples3)

xmin, xmax = np.min(flat_samples1), np.max(flat_samples1)
ymin, ymax = np.min(flat_samples2) , np.max(flat_samples2)
zmin, zmax = 10, 5000#np.min(flat_samples3) , np.max(flat_samples3)
Npoints = 150 #opts.Npoints
#######################################################
################ We will be using masses in log scale so better to use
##### If we want to use log param we need proper grid spacing in log scale
p1grid = np.logspace(np.log10(xmin), np.log10(xmax), Npoints)
p2grid = np.logspace(np.log10(ymin), np.log10(ymax), Npoints)
p3grid = np.linspace(zmin, zmax, 50)#Npoints) 
#mesh grid points 
XX, YY, ZZ = np.meshgrid(p1grid, p2grid, p3grid,  indexing='ij')
#input_transf['log', 'log', None] will automatically do it
xy_grid_pts = np.array(list(map(np.ravel, [XX, YY, ZZ]))).T
################################################################################
def get_kde_obj_eval(sample, eval_pts, rescale_arr, alphachoice, input_transf=('log', 'log', 'none'), mass_symmetry=False, maxbw_dL = opts.MaxbwdL):
    maxRescale_dL = 1.0/maxbw_dL
    #Apply m1-m2 symmetry in the samples before fitting
    if mass_symmetry:
        m1 = sample[:, 0]  # First column corresponds to m1
        m2 = sample[:, 1]  # Second column corresponds to m2
        dL = sample[:, 2]  # Third column corresponds to dL
        sample2 = np.vstack((m2, m1, dL)).T
        #Combine both samples into one array
        symsample = np.vstack((sample, sample2))
        kde_object = ad.KDERescaleOptimization(symsample, stdize=True, rescale=rescale_arr, alpha=alphachoice, dim_names=['lnm1', 'lnm2', 'dL'], input_transf=input_transf)
    else:
        kde_object = ad.KDERescaleOptimization(sample, stdize=True, rescale=rescale_arr, alpha=alphachoice, dim_names=['lnm1', 'lnm2', 'dL'], input_transf=input_transf)
    dictopt, score = kde_object.optimize_rescale_parameters(rescale_arr, alphachoice, bounds=((0.01,100),(0.01, 100),(0.01, maxRescale_dL), (0, 1)), disp=True)#, xatol=1e-5, fatol=0.01)
    print("opt results = ", dictopt)
    optbwds = 1.0/dictopt[0:-1]
    print(optbwds)
    optalpha = dictopt[-1]

    print("opt results = ", dictopt)
    return  kde_object, optbwds, optalpha


##First median samples KDE
init_rescale_arr = [1., 1., 1.]
init_alpha_choice = [0.5]
current_kde, errorbBW, erroraALP = get_kde_obj_eval(sample, xy_grid_pts, init_rescale_arr, init_alpha_choice, mass_symmetry=True)
bwx, bwy, bwz = errorbBW[0], errorbBW[1], errorbBW[2]
print(errorbBW)

def get_sliced_data(xx, yy, kde3D,  dLgrid, dL_sliceval=500):
    dL_index = np.searchsorted(dLgrid,  dL_sliceval)#500Mpc
    dL_index_val = dLgrid[dL_index]
    KDE_slice = kde3D[:, :, dL_index]  # Sliced values of F at the chosen x3
    #Rate_slice = rate3D[:, :, dL_index]  # Sliced values of F at the chosen x3
    M1_slice, M2_slice = xx[:, :, dL_index], yy[:, :, dL_index]  
    return M1_slice, M2_slice, KDE_slice, Rate_slice

### reweighting EM algorithm
Total_Iterations = int(opts.NIterations)
discard = int(opts.buffer_start)   # how many iterations to discard default =5
Nbuffer = int(opts.buffer_interval) #100 buffer [how many (x-numbers of ) previous iteration to use in reweighting with average of those past (x-numbers of ) iteration results

iterbwxlist = []
iterbwylist = []
iterbwzlist = []
iteralplist = []
#### We want to save data for rate(m1, m2) in HDF file 
frateh5 = h5.File(opts.output_filename+'optimize_code_test.hdf5', 'a')
dsetxx = frateh5.create_dataset('data_xx', data=XX)
dsetxx.attrs['xname']='xx'
dsetyy = frateh5.create_dataset('data_yy', data=YY)
dsetyy.attrs['yname']='yy'
dsetzz = frateh5.create_dataset('data_zz', data=ZZ)
dsetzz.attrs['zname']='zz'

# Initialize buffer to store last 100 iterations of f(samples) for each event
samples_per_event =  [len(event_list) for event_list in sampleslists3]
num_events = len(meanxi1)
buffers = [[] for _ in range(num_events)]
for i in range(Total_Iterations + discard):
    print("i - ", i)
    rwsamples = []
    for eventid, (samplem1, samplem2, sample3, pdet_k) in enumerate(zip(sampleslists1, sampleslists2, sampleslists3,  pdetlists)):
        samples= np.vstack((samplem1, samplem2, sample3)).T
        event_kde = current_kde.evaluate_with_transf(samples)
        buffers[eventid].append(event_kde)
        if i < discard + Nbuffer :
            rwsample = get_reweighted_sample(samples, pdet_k, current_kde, bootstrap=opts.bootstrap_option)
        else:
            medians_kde_event =  np.median(buffers[eventid][-Nbuffer:], axis=0)
            #reweight base don events previous 100 KDEs median or mean
            rwsample= New_median_bufferkdelist_reweighted_samples(samples, pdet_k, medians_kde_event, bootstrap_choice=opts.bootstrap_option)
        rwsamples.append(rwsample)
    
    if opts.bootstrap_option =='poisson':
        rwsamples = np.concatenate(rwsamples)
    print("iter", i, "  totalsamples = ", len(rwsamples))
    current_kde, shiftedbw, shiftedalp = get_kde_obj_eval(np.array(rwsamples), xy_grid_pts, init_rescale_arr, init_alpha_choice, input_transf=('log', 'log', 'none'), mass_symmetry=True)
    bwx, bwy, bwz = shiftedbw[0], shiftedbw[1], shiftedbw[2]
    print("bwvalues", bwx, bwy, bwz)
    group = frateh5.create_group(f'iteration_{i}')

    # Save the data in the group
    group.create_dataset('rwsamples', data=rwsamples)
    group.create_dataset('alpha', data=shiftedalp)
    group.create_dataset('bwx', data=bwx)
    group.create_dataset('bwy', data=bwy)
    group.create_dataset('bwz', data=bwz)
    frateh5.flush()
    iterbwxlist.append(bwx)
    iterbwylist.append(bwy)
    iterbwzlist.append(bwz)
    iteralplist.append(shiftedalp)
    #if i > discard and i%Nbuffer==0:
    if i > 1 and i%Nbuffer==0:
        iterstep = int(i)
        print(iterstep)
        u_plot.histogram_datalist(iterbwxlist[-Nbuffer:], dataname='bwx', pathplot=opts.pathplot, Iternumber=iterstep)
        u_plot.histogram_datalist(iterbwylist[-Nbuffer:], dataname='bwy', pathplot=opts.pathplot, Iternumber=iterstep)
        u_plot.histogram_datalist(iterbwzlist[-Nbuffer:], dataname='bwz', pathplot=opts.pathplot, Iternumber=iterstep)
        u_plot.histogram_datalist(iteralplist[-Nbuffer:], dataname='alpha', pathplot=opts.pathplot, Iternumber=iterstep)
        #######need to work on plots
        #if opts.logkde:
frateh5.close()

u_plot.bandwidth_correlation(iterbwxlist, number_corr=discard, error=0.02,  pathplot=opts.pathplot+'bwx_')
u_plot.bandwidth_correlation(iterbwylist, number_corr=discard, error=0.02,  pathplot=opts.pathplot+'bwy_')
u_plot.bandwidth_correlation(iterbwzlist, number_corr=discard, error=0.02,  pathplot=opts.pathplot+'bwz_')
u_plot.bandwidth_correlation(iteralplist, number_corr=discard,  error=0.02, param='alpha',pathplot=opts.pathplot, log=False)
quit()

def compute_median_kde(hdf_file, start_iter, end_iter, eval_samples, XX, YY, dLgrid):
    kde_list = []
    # Open the HDF5 file
    with h5py.File(hdf_file, 'r') as hdf:
        for i in range(start_iter, end_iter + 1):
            iteration_name = f'iteration_{i}'
            # Check if the iteration exists in the file
            if iteration_name not in hdf:
                print(f"Iteration {i} not found in file.")
                continue
            
            # Load data for this iteration
            group = hdf[iteration_name]
            samples = group['samples'][:]
            alpha = group['alpha'][()]
            bwx = group['bwx'][()]
            bwy = group['bwy'][()]
            bwz = group['bwz'][()]
            
            # Create the KDE
            train_kde = ad.AdaptiveBwKDE(
                samples, 
                None, 
                input_transf=('log', 'log', 'none'),
                stdize=True, 
                rescale=[1/bwx, 1/bwy, 1/bwz],
                alpha=alpha
            )
            
            # Evaluate the KDE on the evaluation samples
            eval_kde3d = train_kde.evaluate_with_transf(eval_samples)
            #slice along dL dimension
            M1_slice, M2_slice, KDE_slice =  get_sliced_data(XX, YY, eval_kde3d, dLgrid, dL_sliceval=500)
            kde_list.append(KDE_slice)
    
    # Compute the median of all KDE evaluations
    kde_array = np.array(kde_list)  # Shape: (num_iterations, num_eval_points)
    median_kde = np.median(kde_array, axis=0)
    
    return kde_array #median_kde

# Define eval_samples
xmin, xmax, ymin, ymax, zmin, zmax, Npoints = 3, 100, 3, 100, 10, 5000, 150
p1grid = np.logspace(np.log10(xmin), np.log10(xmax), Npoints)
p2grid = np.logspace(np.log10(ymin), np.log10(ymax), Npoints)
p3grid = np.linspace(zmin, zmax, 50)  # Example: 50 points in Z
XX, YY, ZZ = np.meshgrid(p1grid, p2grid, p3grid, indexing='ij')
eval_samples = np.array(list(map(np.ravel, [XX, YY, ZZ]))).T

# Compute the median KDE
hdf_file = opts.output_filename+'del_test.hdf5'  # Your HDF5 file path
median_kde = compute_median_kde(hdf_file, 10, 20, eval_samples,  XX, YY, p3grid)
print("Median KDE computed for iterations 10-20.")

#once we get data how we want to plot?
u_plot2.average2Dkde_plot(meanxi1, meanxi2, M1_slice, M2_slice, iter2Dkde_list[discard:], pathplot=opts.pathplot, titlename=1001, plot_label='KDE', x_label='m1', y_label='m2', plottag='allKDEscombined_')
