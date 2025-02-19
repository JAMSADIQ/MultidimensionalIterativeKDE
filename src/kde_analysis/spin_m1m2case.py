import sys
import matplotlib
sys.path.append('pop-de/popde/')
import density_estimate as d
import adaptive_kde as ad
import priors as spin_prior
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import h5py as h5
import scipy
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator
from matplotlib import rcParams
from matplotlib.colors import LogNorm, Normalize
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u
import utils_plot as u_plot
import new_o123_class_found_inj_general as u_pdet

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
parser.add_argument('--datafilename3', help='h5  file containing N sample of Xieff for each event')
parser.add_argument('--datafilename-redshift', help='h5  file containing N sample of redshift for each event')
parser.add_argument('--datafilename-dL', help='h5  file containing N sample of dL for each event')
parser.add_argument('--parameter1', help='name of parameter which we use for x-axis for KDE', default='m1')
parser.add_argument('--parameter2', help='name of parameter which we use for y-axis for KDE: m2', default='m2')
parser.add_argument('--parameter3', help='name of parameter which we use for y-axis for KDE [can be Xieff, dL]', default='Xieff')
parser.add_argument('--injectionfile',  help='H5 file from GWTC3 public data for search sensitivity.', default='endo3_bbhpop-LIGO-T2100113-v12.hdf5')
# priors 
parser.add_argument('--xieff-prior-function', type=str, default='chi_effective_prior_from_isotropic_spins', help='prior function for xieff form priors.py file need to fix this.')
parser.add_argument('--redshift-prior-power', type=float, default=2.0, help='If set, perform KDE in logarithmic space.')
# KDE grid options: limits and resolution
parser.add_argument('--m1-min', default=5.0, type=float, help='Minimum value for primary mass m1.')
parser.add_argument('--m1-max', default=100.0, type=float, help='Maximum value for primary mass m1.')
parser.add_argument('--Npoints', default=200, type=int, help='Number of points for KDE evaluation.')
parser.add_argument('--param2-min', default=4.95, type=float, help='Minimum value for parameter 2 if it is  m2, else if dL use 10')
parser.add_argument('--param2-max', default=100.0, type=float, help='Maximum value for parameter 2 if it is m2 else if dL  use 10000')
parser.add_argument('--param3-min', default=-1., type=float, help='Minimum value for parameter 3 if it is  dL, else if Xieff use -1')
parser.add_argument('--param3-max', default=1., type=float, help='Maximum value for parameter 3 if it is dL else if Xieff  use +1')

# Rescaling factor bounds [bandwidth]
parser.add_argument('--min-bw-Xieffdim', default=0.01, type=float, help='Set the minimum bandwidth for the DL dimension. The value must be >= 0.3 for mmain analysis')
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
parser.add_argument('--output-filename', default='m1m2mXieff3Danalysis_', help='write a proper name of output hdf files based on analysis', type=str)
opts = parser.parse_args()
#####################################################################
maxRescale_Xieffdim = round(1.0/ opts.min_bw_Xieffdim)
print(f'max rescal factor in Xief dim = {maxRescale_Xieffdim}')

#set the prior factors correctly here before reweighting
prior_kwargs = {'redshift_prior_power': opts.redshift_prior_power}
print(f"prior powers: {prior_kwargs}")
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


def preprocess_data(m1_injection, dL_injection, pe_m1, pe_dL, num_bins=10):
    """
    Preprocess data by filtering invalid entries based on distance limits.

    Args:
        m1_injection (np.ndarray): Injected primary masses.
        dL_injection (np.ndarray): Injected luminosity distances.
        pe_m1 (np.ndarray): Primary mass posterior estimates.
        pe_dL (np.ndarray): Luminosity distance posterior estimates.
        num_bins (int): Number of bins for preprocessing.

    Returns:
        tuple: Filtered primary masses, distances, and their indices.
    """ 
    log_m1 = np.log10(m1_injection)
    pe_log_m1 = np.log10(pe_m1)
    bins = np.linspace(log_m1.min(), log_m1.max(), num_bins + 1)

    max_dL_per_bin = np.array([
        dL_injection[(log_m1 >= bins[i]) & (log_m1 < bins[i + 1])].max()
        if ((log_m1 >= bins[i]) & (log_m1 < bins[i + 1])).any() else -np.inf
        for i in range(len(bins) - 1)
    ])

    filtered_pe_m1 = []
    filtered_pe_dL = []
    filtered_indices = []

    for i in range(len(bins) - 1):
        bin_mask = (pe_log_m1 >= bins[i]) & (pe_log_m1 < bins[i + 1])
        max_dL = max_dL_per_bin[i]
        keep_mask = bin_mask & (pe_dL <= max_dL)

        filtered_pe_m1.extend(pe_m1[keep_mask])
        filtered_pe_dL.extend(pe_dL[keep_mask])
        filtered_indices.extend(np.where(keep_mask)[0])

    return (
        np.array(filtered_pe_m1),
        np.array(filtered_pe_dL),
        np.array(filtered_indices)
    )

#this is specific to m1-m2-dL 3D analysis 
def prior_factor_function(samples, redshift_vals, redshift_prior_power):
    """
    Compute a prior factor for reweighting for Xieff and masses from redshift to the source frame.
    For non-cosmo pe files:
    - If the source-frame mass is used, apply (1+z)^power for redshift scaling.

    Args:
        samples (np.ndarray): Array of samples with shape (N, 3), where N is the number of samples.
        redshift_vals:  (np.ndarray or list): Redshift values corresponding to the samples.
        redshift_prior_power (float, optional): Power to apply to the redshift prior. Default is 2.0
    Returns:
        np.ndarray: Prior factor for each sample, computed as 1 / (dL^power * (1+z)^power).
    """
    if samples.shape[1] != 3:
        raise ValueError("Samples array must have exactly three columns: [m1, m2, dL].")

    if len(redshift_vals) != len(samples):
        raise ValueError("Length of redshifts must match the number of samples.")

    # Extract values from samples
    m1_values,  m2_values, Xieff_values = samples[:, 0], samples[:, 1],  samples[:, 2]
    q_values =  m2_values/m1_values
    Xieff_prior = np.zeros(len(q_values))
    aMax = 0.999
    #compute prior from Callister etal code
    for i in range(len(q_values)):#enumerate(q_values):
        print(Xieff_values[i])
        Xieff_prior[i] = spin_prior.chi_effective_prior_from_isotropic_spins(q_values[i], aMax , Xieff_values[i])

    redshift_prior = (1. + redshift_vals)**redshift_prior_power
    # Compute and return the prior factor
    prior_factors = 1.0 / (Xieff_prior * redshift_prior)

    return prior_factors


def get_random_sample(original_samples, bootstrap='poisson'):
    """
    Generate a random sample from the provided original samples
    
    Parameters:
    -----------
    original_samples : array-like
        The array or list of original samples to draw from

    bootstrap : str, optional, default='poisson'
        The bootstrap method to use for resampling. Options:
        - 'poisson': The sample size will be drawn from a 
            Poisson distribution with mean = 1.

    Returns:
    --------
    random_sample : ndarray
         randomly selected sample/samples(/empty) from the original samples
    """
    rng = np.random.default_rng()
    if bootstrap =='poisson':
        reweighted_sample = rng.choice(original_samples, np.random.poisson(1))
    else:
        reweighted_sample = rng.choice(original_samples)
    return reweighted_sample


def get_reweighted_sample(original_samples, redshiftvals, vt_vals, fpop_kde, bootstrap='poisson', prior_factor=prior_factor_function,  prior_factor_kwargs=None):
    """
    Generate  reweighted random sample/samples  from the original PE samples
    
    This function adjusts the probability of each sample based on a kernel density estimate (KDE) for the population
    distribution, the prior factor, and detection probability values, and then performs the random resampling using 
    `np.random.choice`  with or without poisson sampling size

    Parameters:
    -----------
    original_samples : list or array-like
        The list or array of PE samples representing a set of events or observations.
        
    redshiftvals : array-like
        The redshift values corresponding to the `original_samples`, used to compute prior factors.
        
    vt_vals : array-like
        The sensitive volume time values for each sample, used to scale the KDE estimate.
        
    fpop_kde : KDE object
        A kernel density estimate (KDE) object, such as a `GaussianKDE`, that models the population distribution.
        It is used to calculate the KDE at the sample points.
        
    bootstrap : str, optional, default='poisson'
        The bootstrap method to use for resampling. Options:
        - 'poisson': Resampling is done with sample size drawn from a Poisson distribution with mean = 1.
        - Any other value: Uniform random sampling 
        
    prior_factor : callable, optional, default=prior_factor_function
        A function that calculates the prior factor for each sample, typically dependent on the redshift.
        It adjusts the sample probabilities based on a non-uniform prior.
        
    prior_factor_kwargs : dict, optional, default=None
        Additional keyword arguments to pass to the `prior_factor` function when calculating prior factors.
        

    Returns:
    --------
    reweighted_sample : ndarray
        randomly selected, reweighted sample/samples  from the `original_samples`. 
    """
    # Ensure prior_factor_kwargs is a dictionary
    if prior_factor_kwargs is None:
        prior_factor_kwargs = {}

    # Evaluate the KDE and apply the maximum detection probability cap
    fkde_samples = fpop_kde.evaluate_with_transf(original_samples) / vt_vals

    # Adjust probabilities based on the prior factor
    frate_atsample = fkde_samples * prior_factor(original_samples, redshiftvals, **prior_factor_kwargs) 
    # Normalize :sum=1
    fpop_at_samples = frate_atsample/frate_atsample.sum()

    # Initialize random number generator
    rng = np.random.default_rng()

    # Perform resampling with or without Poisson reweighting
    if bootstrap =='poisson':
        reweighted_sample = rng.choice(original_samples, np.random.poisson(1), p=fpop_at_samples)
    else:
        reweighted_sample = rng.choice(original_samples, p=fpop_at_samples)

    return reweighted_sample


def New_median_bufferkdelist_reweighted_samples(sample, redshiftvals, vt_vals, meanKDEevent, bootstrap_choice='poisson', prior_factor=prior_factor_function, prior_factor_kwargs=None):
    """
    Generate reweighted samples based on the median of estimated KDE train vals for a set of PE samples,
    adjusting for detection probability and cosmological redshift.

    The KDE values represent the mean of the kernel density estimates (KDE) for the last 100 iterations,
    and the weights are modified by detection probabilities and a non-uniform prior.

    Parameters:
    -----------
    sample : array-like
        The PE samples representing a set of events or observations. Typically, these could be numerical or parameter sets.
        
    redshiftvals : array-like
        The redshift values corresponding to the `sample`, used in the calculation of the prior factor.
        
    vt_vals : array-like
        The sensitive Volume * Times values for each PE sample, used to adjust the KDE weights.
        
    meanKDEevent : array-like
        The mean KDE values for the PE samples, which represent the average KDE values calculated 
        over the previous 100 iterations. These values are used as the basis for resampling probabilities.

    bootstrap_choice : str, optional, default='poisson'
        The bootstrap method to use for resampling. Options:
        - 'poisson': Resampling with sample size drawn from a Poisson distribution with mean = 1.
        - Any other value: Uniform random sampling without reweighting.
        
    prior_factor : callable, optional, default=prior_factor_function
        A function that calculates the prior factor for each sample, typically depending on redshift. It adjusts the sample probabilities based on a non-uniform prior.
        
    prior_factor_kwargs : dict, optional, default=None
        Additional keyword arguments for the `prior_factor` function when calculating prior factors.
        

    Returns:
    --------
    reweighted_sample : ndarray
        randomly selected, reweighted samples/sample from the `samples`, adjusted by the KDE values, prior factor,
        and detection probabilities.
    """
    # Ensure prior_factor_kwargs is a dictionary
    if prior_factor_kwargs is None:
        prior_factor_kwargs = {}

    # Compute KDE probabilities divide by regularized pdetvals
    kde_by_vt = meanKDEevent/vt_vals

    # Adjust probabilities based on the prior factor
    kde_by_vt  *= prior_factor(sample, redshiftvals, **prior_factor_kwargs)

    #Normalize:  sum=1
    norm_mediankdevals = kde_by_vt/sum(kde_by_vt)

    rng = np.random.default_rng()

    if bootstrap_choice =='poisson':
        reweighted_sample = rng.choice(sample, np.random.poisson(1), p=norm_mediankdevals)
    else:
        reweighted_sample = rng.choice(sample, p=norm_mediankdevals)
    return reweighted_sample

#######################################################################
# Main execution begins here
#STEP I: call the PE sample data and get PDET on PE samples using power law on m2
injection_file = opts.injectionfile
#see this link: https://zenodo.org/records/7890437:  "endo3_bbhpop-LIGO-T2100113-v12.hdf5"
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

f.close()

#####################################
# get PDET  (m1, m2, dL)
run_fit = 'o3'
run_dataset = 'o3'
dmid_fun = 'Dmid_mchirp_fdmid_fspin' #'Dmid_mchirp_fdmid'
emax_fun = 'emax_exp'
alpha_vary = None
g = u_pdet.Found_injections(dmid_fun = 'Dmid_mchirp_fdmid_fspin', emax_fun='emax_exp', alpha_vary = None, ini_files = None, thr_far = 1, thr_snr = 10)
g.load_inj_set(run_dataset)
g.get_opt_params(run_fit)
g.set_shape_params()


fz = h5.File(opts.datafilename_redshift, 'r')
#fz = h5.File('Final_noncosmo_GWTC3_redshift_datafile.h5', 'r')
dz = fz['randdata']
fdL = h5.File(opts.datafilename_dL, 'r')
#fdL = h5.File('Final_noncosmo_GWTC3_dL_datafile.h5', 'r')
ddL = fdL['randdata']


f1 = h5.File(opts.datafilename1, 'r')#m1
#f1 = h5.File('Final_noncosmo_GWTC3_m1srcdatafile.h5', 'r')#m1
d1 = f1['randdata']
#f2 = h5.File('Final_noncosmo_GWTC3_m2srcdatafile.h5', 'r')#m2
f2 = h5.File(opts.datafilename2, 'r')#m2
d2 = f2['randdata']
#f3 = h5.File('Final_noncosmo_GWTC3_dL_datafile.h5', 'r')#Xieff
f3 = h5.File(opts.datafilename3, 'r')#Xieff
d3 = f3['randdata']
sampleslists1 = []
medianlist1 = f1['initialdata/original_mean'][...]
eventlist = []
sampleslists2 = []
medianlist2 = f2['initialdata/original_mean'][...]
sampleslists3 = []
medianlist3 = f3['initialdata/original_mean'][...]
vth5file = h5.File("VTatposteriors_allevents.hdf5", "w")
vtlists = []
redshift_lists = []
for k in d1.keys():
    eventlist.append(k)
    if (k  == 'GW190719_215514_mixed-nocosmo' or k == 'GW190805_211137_mixed-nocosmo'):
        m1_values = d1[k][...]
        #we need detector frame mass for PDET
        m1det_values = d1[k][...]*(1.0 + dz[k][...])
        m2_values = d2[k][...]
        m2det_values = d2[k][...]*(1.0 + dz[k][...])
        d_Lvalues = ddL[k][...]
        Xieff_values = d3[k][...]
        #clean data
        m1_values, d_Lvalues, correct_indices =  preprocess_data(injection_m1, injection_dL, m1_values, d_Lvalues, num_bins=10)

        m2_values = m2_values[correct_indices]
        m1det_values = m1det_values[correct_indices]
        m2det_values = m2det_values[correct_indices]
        Xieff_values = Xieff_values[correct_indices]
        redshift_values = z_at_value(cosmo.luminosity_distance, d_Lvalues*u.Mpc).value
        vt_values = np.zeros(len(m1det_values))
        for ix in range(len(vt_values)):
            vt_values[ix] = g.total_sensitive_volume(m1_values[ix], m2_values[ix], chieff = Xieff_values[ix]) 
    else:
        m1_values = d1[k][...]
        m1det_values = d1[k][...]*(1.0 + dz[k][...])
        m2_values = d2[k][...]
        m2det_values = d2[k][...]*(1.0 + dz[k][...])
        d_Lvalues = ddL[k][...]
        Xieff_values = d3[k][...]
        redshift_values = z_at_value(cosmo.luminosity_distance, d_Lvalues*u.Mpc).value
        vt_values = np.zeros(len(m1det_values))
        for ix in range(len(vt_values)):
            vt_values[ix] = g.total_sensitive_volume(m1_values[ix], m2_values[ix], chieff = Xieff_values[ix])
    vth5file.create_dataset(k, data=np.array(vt_values))
    vtlists.append(vt_values)
    sampleslists1.append(m1_values)
    sampleslists2.append(m2_values)
    sampleslists3.append(Xieff_values)
    redshift_lists.append(redshift_values)

f1.close()
f2.close()
f3.close()
fz.close()
fdL.close()
vth5file.close()


meanxi1 = np.array(medianlist1)
meanxi2 = np.array(medianlist2)
meanxi3 = np.array(medianlist3)
flat_samples1 = np.concatenate(sampleslists1).flatten()
flat_samples2 = np.concatenate(sampleslists2).flatten()
flat_samples3 = np.concatenate(sampleslists3).flatten()
flat_vtlist = np.concatenate(vtlists).flatten()
flat_sample_z = np.concatenate(redshift_lists).flatten()
print("min max m1 =", np.min(flat_samples1), np.max(flat_samples1))
print("min max m2 =", np.min(flat_samples2), np.max(flat_samples2))
print("min max Xieff =", np.min(flat_samples3), np.max(flat_samples3))


# Create the scatter plot for pdet save with 3D analysis name
def eta_from_mass1_mass2(mass1, mass2):
    """Returns the symmetric mass ratio from mass1 and mass2."""
    return mass1*mass2 / (mass1 + mass2)**2.
def mchirp_from_mass1_mass2(mass1, mass2):
    """Returns the chirp mass from mass1 and mass2."""
    return eta_from_mass1_mass2(mass1, mass2)**(3./5) * (mass1 + mass2)

Mchirp = mchirp_from_mass1_mass2(flat_samples1, flat_samples2) 
#u_plot.plot_pdetscatter(flat_samples1, flat_samples3, flat_vtlist, xlabel=r'$m_{1, source} [M_\odot]$', ylabel=r'$\chi_\mathrm{effective}$', title=r'$VT$',save_name="VT_3Dm1m2dLXieff_correct_mass_frame_m1_Xieff_scatter.png", pathplot=opts.pathplot, show_plot=True)
u_plot.plot_pdetscatter(Mchirp, flat_samples3, flat_vtlist, xlabel=r'$\mathcal{M}_\mathrm{chirp}$', ylabel=r'$\chi_\mathrm{effective}$', title=r'$VT$',save_name="VT_3Dm1m2dLXieff_correct_mass_frame_m1_Xieff_scatter.png", pathplot=opts.pathplot, show_plot=True)
u_plot.plot_pdetscatter(flat_samples2/flat_samples1, flat_samples3, flat_vtlist, xlabel=r'$q$', ylabel=r'$\chi_\mathrm{effective}$', title=r'$VT$',save_name="VT_3Dm1m2dLXieff_correct_mass_frame_m1_Xieff_scatter.png", pathplot=opts.pathplot, show_plot=False)

#special plot with z on right y axis
u_plot.plot_pdetscatter_m1dL_redshiftYaxis(flat_samples1, flat_samples3, flat_vtlist, flat_sample_z, xlabel=r'$m_{1, source} [M_\odot]$', ylabel=r'$\chi_\mathrm{effective}$', title=r'$p_\mathrm{det}$',  save_name="pdet_m1Xieff_redshift_right_yaxis.png", pathplot=opts.pathplot, show_plot=False)

# Create the scatter plot for pdet 
u_plot.plotpdet_3Dm1m2dLscatter(flat_samples1, flat_samples2, flat_samples3, flat_vtlist, save_name="pdet_m1m2Xieff_3Dscatter.png", pathplot=opts.pathplot, show_plot=False)

##########################################
sampleslists = np.vstack((flat_samples1, flat_samples2, flat_samples3)).T
sample = np.vstack((meanxi1, meanxi2, meanxi3)).T
print(sampleslists.shape)
sample = np.vstack((meanxi1, meanxi2, meanxi3)).T
######################################################
################################################################################
def get_kde_obj_eval(sample, rescale_arr, alphachoice, input_transf=('log', 'log', 'none'), mass_symmetry=False, minbw_Xieff=0.01):
    maxRescale_Xieff = 1.0/minbw_Xieff
    #Apply m1-m2 symmetry in the samples before fitting
    if mass_symmetry:
        m1 = sample[:, 0]  # First column corresponds to m1
        m2 = sample[:, 1]  # Second column corresponds to m2
        Xieff = sample[:, 2]  # Third column corresponds to Xieff
        sample2 = np.vstack((m2, m1, Xieff)).T
        #Combine both samples into one array
        symsample = np.vstack((sample, sample2))
        kde_object = ad.KDERescaleOptimization(symsample, stdize=True, rescale=rescale_arr, alpha=alphachoice, dim_names=['lnm1', 'lnm2', 'Xieff'], input_transf=input_transf)
    else:
        kde_object = ad.KDERescaleOptimization(sample, stdize=True, rescale=rescale_arr, alpha=alphachoice, dim_names=['lnm1', 'lnm2', 'Xieff'], input_transf=input_transf)

    dictopt, score = kde_object.optimize_rescale_parameters(rescale_arr, alphachoice, bounds=((0.01,100),(0.01, 100),(0.01, maxRescale_Xieff), (0, 1)), disp=True)
    print("opt results = ", dictopt)
    optbwds = 1.0/dictopt[0:-1]
    print(optbwds)
    optalpha = dictopt[-1]

    print("opt results = ", dictopt)
    return  kde_object, optbwds, optalpha


##First median samples KDE
init_rescale_arr = [1., 1., 1.]
init_alpha_choice = [0.5]
current_kde, errorbBW, erroraALP = get_kde_obj_eval(sample,  init_rescale_arr, init_alpha_choice, mass_symmetry=True, input_transf=('log', 'log', 'none'), minbw_Xieff=opts.min_bw_Xieffdim)
bwx, bwy, bwz = errorbBW[0], errorbBW[1], errorbBW[2]
print(errorbBW)

def get_sliced_data(xx, yy, kde3D,  Xieffgrid, Xieff_sliceval=500):
    Xieff_index = np.searchsorted(Xieffgrid,  Xieff_sliceval)#500Mpc
    Xieff_index_val = Xieffgrid[Xieff_index]
    KDE_slice = kde3D[:, :, Xieff_index]  # Sliced values of F at the chosen x3
    #Rate_slice = rate3D[:, :, Xieff_index]  # Sliced values of F at the chosen x3
    M1_slice, M2_slice = xx[:, :, Xieff_index], yy[:, :, Xieff_index]  
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
frateh5 = h5.File(opts.output_filename+'VT_min_bw_Xieff'+str(opts.min_bw_Xieffdim)+'_optimize_code_test.hdf5', 'a')

# Initialize buffer to store last 100 iterations of f(samples) for each event
samples_per_event =  [len(event_list) for event_list in sampleslists3]
num_events = len(meanxi1)
buffers = [[] for _ in range(num_events)]
for i in range(Total_Iterations + discard):
    print("i - ", i)
    rwsamples = []
    for eventid, (samplem1, samplem2, sample3, redshiftvals, vt_k) in enumerate(zip(sampleslists1, sampleslists2, sampleslists3, redshift_lists, vtlists)):
        samples= np.vstack((samplem1, samplem2, sample3)).T
        event_kde = current_kde.evaluate_with_transf(samples)
        buffers[eventid].append(event_kde)
        if i < discard + Nbuffer :
            rwsample = get_reweighted_sample(samples, redshiftvals, vt_k, current_kde, bootstrap=opts.bootstrap_option,  prior_factor_kwargs=prior_kwargs)
        else:
            medians_kde_event =  np.median(buffers[eventid][-Nbuffer:], axis=0)
            #reweight base don events previous 100 KDEs median or mean
            rwsample= New_median_bufferkdelist_reweighted_samples(samples, redshiftvals, vt_k, medians_kde_event, bootstrap_choice=opts.bootstrap_option, prior_factor_kwargs=prior_kwargs)
        rwsamples.append(rwsample)
    
    if opts.bootstrap_option =='poisson':
        rwsamples = np.concatenate(rwsamples)
    print("iter", i, "  totalsamples = ", len(rwsamples))
    current_kde, shiftedbw, shiftedalp = get_kde_obj_eval(np.array(rwsamples), init_rescale_arr, init_alpha_choice, input_transf=('log', 'log', 'none'), mass_symmetry=True,  minbw_Xieff=opts.min_bw_Xieffdim)
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
