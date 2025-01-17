import sys
sys.path.append('pop-de/popde/')
import density_estimate as d
import adaptive_kde as ad
import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import LogNorm, Normalize
from matplotlib import rcParams
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u
import pickle
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

parser = argparse.ArgumentParser(description="Density estimation and reweighting pipeline.")
# Input files #maybe we should combine these three to one
parser.add_argument('--datafilename1', required=True, help='H5 file containing N samples for m1.')
parser.add_argument('--datafilename2', required=True, help='H5 file containing N samples for parameter2.')
parser.add_argument('--datafilename3', required=True, help='H5 file containing N samples of parameter3.')
# for removing bad samples based on sensitivity search injection study by LVK
parser.add_argument('--injectionfile',  help='H5 file from GWTC3 public data for search sensitivity.', default='endo3_bbhpop-LIGO-T2100113-v12.hdf5')
# for pdet calculation
parser.add_argument('--power-index-m2',  type=float, default=1.26, help='beta or power index for m2 distribution in pdet calculation')
parser.add_argument('--min-m2-integration',  type=float, default=5.0, help='minimum intgeration limit for m2. It can be problematic if min is smaller than smallest m1 value.')

# Parameters
parser.add_argument('--parameter1', default='m1', help='Name of parameter for x-axis of KDE.')
parser.add_argument('--parameter2', default='m2', help='Name of parameter for y-axis of KDE.')
parser.add_argument('--parameter3', default='dL', help='Name of parameter for z-axis of KDE.')
# Priors for reweighting 
parser.add_argument('--dl-prior-power', type=float, default=2.0, help='If set, perform KDE in logarithmic space.')
parser.add_argument('--redshift-prior-power ', type=float, default=2.0, help='If set, perform KDE in logarithmic space.')
# KDE grid options: limits and resolution
parser.add_argument('--m1-min', default=5.0, type=float, help='Minimum value for primary mass m1.')
parser.add_argument('--m1-max', default=100.0, type=float, help='Maximum value for primary mass m1.')
parser.add_argument('--Npoints', default=200, type=int, help='Number of points for KDE evaluation.')
parser.add_argument('--param2-min', default=4.95, type=float, help='Minimum value for parameter 2 if it is  m2, else if dL use 10')
parser.add_argument('--param2-max', default=100.0, type=float, help='Maximum value for parameter 2 if it is m2 else if dL  use 10000')
parser.add_argument('--param3-min', default=200., type=float, help='Minimum value for parameter 3 if it is  dL, else if Xieff use -1')
parser.add_argument('--param3-max', default=8000., type=float, help='Maximum value for parameter 3 if it is dL else if Xieff  use +1')

# Expectation Maximization Algorithm Reweighting
parser.add_argument('--fpopchoice', default='kde', help='choice of fpop to be rate or kde', type=str)
parser.add_argument('--reweight-method', default='bufferkdevals', type=str, help=('Reweighting method for Gaussian sample shift: ''"bufferkdevals" uses buffered KDE values, and "bufferkdeobject" uses buffered KDE objects.'))
parser.add_argument('--reweight-sample-option', default='reweight', type=str, help=('Option to reweight samples: choose "noreweight" to skip reweighting or "reweight" ''to use fpop probabilities for reweighting. If "reweight", one sample is used ''for no bootstrap, and multiple samples are used for Poisson.'))
parser.add_argument('--bootstrap-option', default='poisson', type=str, help=('Bootstrap method: choose "poisson" for Poisson resampling or "nopoisson" to skip it. ''If "None", reweighting will be based on fpop probabilities, with a single reweighted sample for each event.')
)
parser.add_argument('--buffer-start', default=100, type=int, help='The starting iteration for buffering in the reweighting process.')
parser.add_argument('--buffer-interval', default=100, type=int, help=('The interval of the buffer determines how many previous iteration results are used in the next iteration for reweighting.'))
parser.add_argument('--NIterations', default=1000, type=int, help='Total number of iterations for the reweighting process.')
# selection effect capping
parser.add_argument('--max-pdet', default=0.1, type=float, help='Capping value for small pdet to introduce regularization.')
# Rescaling factor bounds [bandwidth]
parser.add_argument('--min-bw-dLdim', default=0.01, type=float, help='Set the minimum bandwidth for the DL dimension. The value must be >= 0.3 for mmain analysis')
# Output and plotting
parser.add_argument('--pathplot', default='./', help='Path to save plots.')
parser.add_argument('--output-filename', default='output_data', help='Base name for output HDF5 files.')
opts = parser.parse_args()

#### min bw choice for dL
min_bw_dL = opts.min_bw_dLdim
print("min bw for dL = ", min_bw_dL
index_powerlaw_m2 = opts.power_index_m2
m2min = opts.min_m2_integration
print("powerlaw on m2 has index,  min m2 =",  index_powerlaw_m2, m2min)
# Define cosmology
H0 = 67.9  # km/s/Mpc
omega_m = 0.3065
cosmo = FlatLambdaCDM(H0=H0, Om0=omega_m)

def get_mass_in_detector_frame(dL_Mpc, mass):
    """
    Compute the redshift corresponding to a luminosity distance.
    and get detector frame mass
    Args:
        dL_Mpc (float/array): Luminosity distance in megaparsecs (Mpc).
        mass (float/array): source frame mass.

    Returns:
        float/array: Corresponding detector frame mass
    """
    zcosmo = z_at_value(cosmo.luminosity_distance, dL_Mpc * u.Mpc).value
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

#we are not using it now but will integrate it in future
def compute_pdet(mass_detector_frame, dL, beta, class_instance, classcall=None, min_m2=5.0):
    """
    Compute detection probability using given parameters.

    Args:
        mass_detector_frame (float): Mass in the detector frame.
        dL (float): Luminosity distance.
        beta (float): Power-law index.
        class_instance: Instance of a class with a method to compute pdet.
        min_m2 (float): minimum for integration over m2, default = 5.0

    Returns:
        float: Detection probability.
    """
    return class_instance.pdet_of_m1_dL_powerlawm2(
        mass_detector_frame, min_m2, dL, beta=beta, classcall=classcall
    )


def apply_max_cap_function(pdet_list, max_pdet_cap=0.1):
    """
    Applies the max(max_pdet_cap, pdet) function to each element in the given list.

    Args:
        pdet_list: A list of values.
        max_pdet_cap: The maximum allowed value for each element. 
                  Defaults to 0.1.

    Returns:
        A numpy array containing the results of applying the function to each element.
    """

    result = []
    for pdet in pdet_list:
        result.append(max(max_pdet_cap, pdet))
    return np.array(result)


#this is specific to m1-dL analysis note for 3D we need to fix this
def prior_factor_function(samples, redshift_vals, dl_prior_power=2.0, redshift_prior_power=2.0):
    """
    Compute a prior factor for reweighting for dL and masses from redshift to the source frame.
    For non-cosmo pe files:
    - Use dL^power (distance factor).
    - If the source-frame mass is used, apply (1+z)^power for redshift scaling.

    Args:
        samples (np.ndarray): Array of samples with shape (N, 2), where N is the number of samples.
        redshift_vals:  (np.ndarray or list): Redshift values corresponding to the samples.
        dl_prior_power (float, optional): Power to apply to the dL prior. Default is 2.0.
        redshift_prior_power (float, optional): Power to apply to the redshift prior. Default is 2.0
    Returns:
        np.ndarray: Prior factor for each sample, computed as 1 / (dL^power * (1+z)^power).
    """
    if samples.shape[1] != 2:
        raise ValueError("Samples array must have exactly two columns: [m1, dL].")
    
    if len(redshifts) != len(samples):
        raise ValueError("Length of redshifts must match the number of samples.")
        
    # Extract values from samples
    m1_values, dL_values = samples[:, 0], samples[:, 1]

    #compute prior
    dL_prior = (dL_values)**dl_prior_power
    redshift_prior = (1. + redshift_vals)**redshift_prior_power

    # Compute and return the prior factor
    prior_factors = 1.0 / (dL_prior * redshift_prior)    
    
    return prior_factors


def get_random_sample(original_samples, bootstrap='poisson'):
    """
    Draws random sample/samples from the given list of samples.
    without applying reweighting
    Args:
      original_samples: A list of samples.
      bootstrap: The type of bootstrapping to use.
                 'poisson': Draws a random number of samples from the
                           original list according to a Poisson distribution
                           with a mean of 1. 
                 Default: 'poisson'.
                 If 'none', draws a single random sample from the list.
    Returns:
      random sample[/samples (nothing) if 'poisson' bootstrap] drawn from the original_samples list.
    """
    rng = np.random.default_rng()

    if bootstrap == 'poisson':
        random_unweighted_sample = rng.choice(original_samples, np.random.poisson(1))
    else:
        random_unweighted_sample = rng.choice(original_samples)
    return random_unweighted_sample


def get_reweighted_sample(original_samples, redshiftvals, pdetvals, fpop_kde, bootstrap='poisson', prior_factor=prior_factor_function, prior_factor_kwargs=None, , max_pdet_cap=0.1):
    """
    Generate  reweighted sample/samples from the original samples based on their probabilities under a given kernel density estimate (KDE) and optional transformations.

    Parameters:
    ----------
    original_samples : list or array-like
        A collection of PE samples of an event

    redshiftvals: list or array-like values
         for each original sample (mass/dL)

    pdet_vals : array-like
        selection effects based on injections

    fpop_kde : object
        A KDE object (e.g., GaussianKDE) that computes the probability density
        of a sample. This object must support a method `evaluate_with_transf`.

    bootstrap : str, optional
        Sampling method for bootstrapping. Options are:
        - 'poisson' (default): Uses Poisson-distributed weights during sampling.
        - 'nopoisson': Direct sampling without Poisson weights.

    prior_factor : callable, optional
        A function to compute a prior adjustment factor for the KDE probabilities.
        This is especially relevant when handling non-uniform priors, such as
        logarithmic priors/ mass-frame/ dL 

    prior_factor_kwargs (dict, optional): Keyword arguments for the `prior_factor_function`. Default is None.

    max_pdet_cap: capping on pdet to avoid 0/0 or divergences for 
    small values

    Returns:
    -------
    reweighted_sample : list or array-like
        A new sample of the same type as `original_samples`, reweighted based on
        the computed probabilities. The number of samples depends on the chosen
        bootstrap method:
        - For 'poisson', the number is drawn from a Poisson distribution.
        - For 'nopoisson', the original sample size is preserved.

    Notes:
    -----
    - The function first computes probabilities for the `original_samples` using the
      `fpop_kde.evaluate_with_transf` method and normalizes them by applying
      `apply_max_cap_function` to `pdetvals`.
      applied to adjust the KDE probabilities.
    - Sampling is performed using `np.random.choice`, with probabilities normalized
      to sum to 1.

    """
    # Ensure prior_factor_kwargs is a dictionary
    if prior_factor_kwargs is None:
        prior_factor_kwargs = {}
    # Compute KDE probabilities divide by regularized pdetvals
    fkde_samples = fpop_kde.evaluate_with_transf(original_samples) / apply_max_cap_function(pdetvals, max_pdet_cap)

    # Adjust probabilities based on the prior factor
    frate_atsample = fkde_samples * prior_factor(original_samples, redshiftvals , **prior_factor_kwargs)

    # Normalize probabilities to sum to 1
    fpop_at_samples = frate_atsample / frate_atsample.sum()

    # Initialize random generator
    rng = np.random.default_rng()

    # Perform reweighted sampling based on bootstrap method
    if bootstrap == 'poisson':
        reweighted_sample = rng.choice(original_samples, np.random.poisson(1), p=fpop_at_samples)
    else:
        reweighted_sample = rng.choice(original_samples, p=fpop_at_samples)

    return reweighted_sample


def mean_bufferkdelist_reweighted_samples(original_samples, redshiftvals, pdetvals, mean_kde_interp, bootstrap_choice='poisson', prior_factor=prior_factor_function, prior_factor_kwargs=None, max_pdet_cap=0.1):
    """
    Generate reweighted samples using interpolator
    prepared on mean of a KDE buffer list
    (e.g., the mean KDE over the last 100 iterations)
    .
    This function replaces the previously used method of 
    using the median of the buffer KDE 
    with the mean. 
    Thisinterpolator is used to compute the probabilities
    for the `original_samples`.

    The KDE probabilities are divided by detection probabilities (`pdetvals`) 
    with regularization applied via `max_pdet_cap`. The reweighted sampling process 
    follows the same approach as in `get_reweighted_sample`.

    Parameters:
    ----------
    original_samples : list or array-like
        A collection of samples representing the mean of each event.
    
    redshiftvals: list or array-like values
         for each original sample (mass/dL)

    pdetvals : array-like
        Detection probabilities used for scaling the KDE values. Regularized 
        with the `max_pdet_cap` parameter to avoid extreme values.

    mean_kde_interp : callable
        An interpolator (e.g., `RegularGridInterpolator`) based on the mean KDE 
        values from previous iterations. It computes the KDE probability density 
        for the given `original_samples`.

    bootstrap_choice : str, optional
        Sampling method for bootstrapping. Options are:
        - 'poisson' (default): Uses Poisson-distributed weights during sampling.
        - 'nopoisson': Direct sampling without Poisson weights.

    prior_factor : callable, optional
        A function to compute a prior adjustment factor for the KDE probabilities. 
        This is particularly relevant when handling non-uniform priors.

    prior_factor_kwargs (dict, optional): Keyword arguments for the `prior_factor_function`. Default is None.

    max_pdet_cap : float, optional
        A regularization cap for detection probabilities (`pdetvals`). Values 
        above this cap are clipped to avoid numerical instability. Default is 0.1.

    Returns:
    -------
    reweighted_sample : list or array-like
        A new sample of the same type as `original_samples`, reweighted based on 
        the computed probabilities. The number of samples depends on the chosen 
        bootstrap method:
        - For 'poisson', the number is drawn from a Poisson distribution.
        - For 'nopoisson', the original sample size is preserved.

    Change
    #median_kde_values = np.percentile(kdelist, 50, axis=0)
    to
    #mean_kde_values= np.mean(iterkde_list[-Nbuffer:], axis=0)
    #interp = RegularGridInterpolator((p1val, p2val), mean_kde_values.T, bounds_error=False, fill_value=0.0)
    """
    #interp = RegularGridInterpolator((m1val, m2val), median_kde_values.T, bounds_error=False, fill_value=0.0)
    
    # Ensure prior_factor_kwargs is a dictionary
    if prior_factor_kwargs is None:
        prior_factor_kwargs = {}

    kde_interp_vals = mean_kde_interp(original_samples)/apply_max_cap_function(pdetvals, max_pdet_cap)

    kde_interp_vals  *= prior_factor(original_samples, redshiftvals, **prior_factor_kwargs)
    norm_mediankdevals = kde_interp_vals/sum(kde_interp_vals)

    rng = np.random.default_rng()
    if bootstrap_choice =='poisson':
        reweighted_sample = rng.choice(original_samples, np.random.poisson(1), p=norm_mediankdevals)
    else:
        reweighted_sample = rng.choice(original_samples, p=norm_mediankdevals)
    return reweighted_sample

# we should make bounds inside arguments to keep bwz max limit
#we shouls add input transf in parser to make it generic too
def get_kde_obj_eval(sample, eval_pts, rescale_arr, alphachoice, input_transf=('log', 'none'), min_bw_dL=0.01):
    """
    Create a KDE object with rescaling optimization, optimize its parameters, 
    and evaluate it at given points.

    This function constructs a KDE object using `ad.KDERescaleOptimization`, performs 
    optimization to rescale parameters for better fitting, and evaluates the KDE at 
    specified evaluation points. It also returns the optimized bandwidths and alpha 
    parameter from the optimization process.

    Parameters:
    ----------
    sample : array-like
        Input sample data used to create the KDE object. Each row represents a data 
        point, and each column corresponds to a feature.

    eval_pts : array-like
        Points at which to evaluate the KDE. The dimensions of `eval_pts` must match 
        those of `sample`.

    rescale_arr : array-like
        Initial rescaling factors for the KDE optimization. These factors are applied 
        to the input data to standardize or transform the scale.

    alphachoice : float
        Initial alpha value for the KDE optimization. This parameter controls the 
        regularization or smoothing effect in the KDE.

    input_transf : tuple of str, optional
        Transformation applied to the input data before KDE computation. The default 
        is `('log', 'none')`, which applies a logarithmic transformation to the first 
        dimension and no transformation to the second.
    min_bw_dL: to adjust lower bound on bw choice of dL default in 0.01
    this need to be include in bounds

    Returns:
    -------
    kde_object : object
        The constructed and optimized KDE object from `ad.KDERescaleOptimization`.

    kde_vals : array-like
        KDE values computed at the specified `eval_pts`.

    optbwds : list of float
        Optimized bandwidths for the KDE in each dimension. Computed as the inverse 
        of the optimized rescaling factors.

    optalpha : float
        Optimized alpha parameter from the KDE optimization process.

    """
    dLmax_rescale_bound = 1.0/min_bw_dL
    kde_object = ad.KDERescaleOptimization(sample, stdize=True, rescale=rescale_arr, alpha=alphachoice, dim_names=['lnm1', 'dL'], input_transf=input_transf)
    #dictopt, score = kde_object.optimize_rescale_parameters(rescale_arr, alphachoice, bounds=((0.01,100),(0.01, 100),(0,1)), disp=True)#, xatol=0.01, fatol=0.1)
    dictopt, score = kde_object.optimize_rescale_parameters(rescale_arr, alphachoice, bounds=((0.01,100),(0.01, dLmax_rescale_bound),(0,1)), disp=True)
    kde_vals = kde_object.evaluate_with_transf(eval_pts)
    optbwds = [1.0/dictopt[0], 1.0/dictopt[1]]
    optalpha = dictopt[-1]
    print("opt results = ", dictopt)
    return  kde_object, kde_vals, optbwds, optalpha



# Main execution begins here
#STEP I: call the PE sample data and get PDET on PE samples using power law on m2
injection_file = opts.injectionfile 
#see this link: https://zenodo.org/records/7890437
injection_file = "/home/jxs1805/Research/CITm1dL/endo3_bbhpop-LIGO-T2100113-v12.hdf5"
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
#here we need pdet
run_fit = 'o3'
run_dataset = 'o3'
dmid_fun = 'Dmid_mchirp_fdmid_fspin' #'Dmid_mchirp_fdmid'
emax_fun = 'emax_exp'
alpha_vary = None
g = u_pdet.Found_injections(dmid_fun = 'Dmid_mchirp_fdmid_fspin', emax_fun='emax_exp', alpha_vary = None, ini_files = None, thr_far = 1, thr_snr = 10)

fz = h5.File(opts.datafilename3, 'r') #redshift values need for mdet
#fz = h5.File('/home/jxs1805/Research/CITm1dL/PEfiles/Final_noncosmo_GWTC3_redshift_datafile.h5', 'r')
dz1 = fz['randdata']

fm1 = h5.File(opts.datafilename1, 'r')
#fm1 = h5.File('/home/jxs1805/Research/CITm1dL/PEfiles/Final_noncosmo_GWTC3_m1srcdatafile.h5', 'r')
d1 = fm1['randdata']

f2dL = h5.File(opts.datafilename2, 'r')
#f2dL = h5.File('/home/jxs1805/Research/CITm1dL/PEfiles/Final_noncosmo_GWTC3_dL_datafile.h5', 'r')
d2 = f2dL['randdata']

# to save samples for iterative reweighting
sampleslists1 = []
medianlist1 = []
eventlist = []
sampleslists2 = []
medianlist2 = []
redshift_lists = []
pdetlists = []

for k in d1.keys():
    eventlist.append(k)
    if (k  == 'GW190719_215514_mixed-nocosmo' or k == 'GW190805_211137_mixed-nocosmo'):
        print(k)
        d_Lvalues = d2[k][...]
        m_values = d1[k][...]
        mdet_values = d1[k][...]*(1.0 + dz1[k][...])
        m_values, d_Lvalues, correct_indices =  preprocess_data(injection_m1, injection_dL, m_values, d_Lvalues, num_bins=10)
        mdet_values = mdet_values[correct_indices]
        redshift_values = z_at_value(cosmo.luminosity_distance, d_Lvalues*u.Mpc).value 
        pdet_values =  np.zeros(len(d_Lvalues))
        #print("minm2, beta are", index_powerlaw_m2, minm2,  "used in pdet")
        for i in range(len(d_Lvalues)):
            pdet_values[i] = u_pdet.pdet_of_m1_dL_powerlawm2(mdet_values[i], minm2, d_Lvalues[i], beta=index_powerlaw_m2, classcall=g)
    else:
        m_values = d1[k][...]
        mdet_values = d1[k][...]*(1.0 + dz1[k][...])
        d_Lvalues = d2[k][...]
        redshift_values = z_at_value(cosmo.luminosity_distance, d_Lvalues*u.Mpc).value
        pdet_values =  np.zeros(len(d_Lvalues))
        for i in range(len(d_Lvalues)):
            pdet_values[i] = u_pdet.pdet_of_m1_dL_powerlawm2(mdet_values[i], minm2, d_Lvalues[i], beta=index_powerlaw_m2, classcall=g)
    pdetlists.append(pdet_values)
    sampleslists1.append(m_values)
    sampleslists2.append(d_Lvalues)
    redshift_lists.append(redshift_values)
    medianlist1.append(np.percentile(m_values, 50))
    medianlist2.append(np.percentile(d_Lvalues, 50))

fm1.close()
f2dL.close()
fz.close()

meanxi1 = np.array(medianlist1)
meanxi2 = np.array(medianlist2)
flat_samples1 = np.concatenate(sampleslists1).flatten()
flat_samples2 = np.concatenate(sampleslists2).flatten()
flat_pdetlist = np.concatenate(pdetlists).flatten()
print("for 69 events the total clean pe samples  = ", len(flat_pdetlist))
#plot of pdet as function of m1-dL after removing bad samples
u_plot.plot_pdetscatter(flat_samples1, flat_samples2, flat_pdetlist, xlabel=r'$m_{1, source} [M_\odot]$', ylabel=r'$d_L [Mpc]$', title=r'$p_\mathrm{det}$',save_name="pdet_power_law_m2_correct_mass_frame_m1_dL_scatter.png", pathplot=opts.pathplot, show_plot=False)


#STEP II: get KDE  from median PE samples & plot it
sampleslists = np.vstack((flat_samples1, flat_samples2)).T
sample = np.vstack((meanxi1, meanxi2)).T
print("shape of train samples =", sampleslists.shape)
#eval grid 
if opts.m1_min is not None and opts.m1_max is not None:
    xmin, xmax = opts.m1_min, opts.m1_max
else:
    xmin, xmax = min([a.min() for a in sampleslists]), max([a.max() for a in sampleslists])
#### we can fix in this analysis 
xmin, xmax = 5, 105
if opts.param2_min is not None and opts.param2_max is not None:
    ymin, ymax = opts.param2_min, opts.param2_max
else:
    ymin, ymax = min([a.min() for a in sampleslists]), max([a.max() for a in sampleslists])
#############we are using fixed dL grid 200-8000Mpc
ymin, ymax = 200, 8000
Npoints = opts.Npoints #200 bydefault
#we need log space points for m1, linear in dL
p1grid = np.logspace(np.log10(xmin), np.log10(xmax), Npoints)
p2grid = np.linspace(ymin, ymax, 150) #here we are using 150 points

XX, YY = np.meshgrid(p1grid, p2grid)
xy_grid_pts = np.array(list(map(np.ravel, [XX, YY]))).T

sample = np.vstack((meanxi1, meanxi2)).T
##First median samples KDE
init_rescale_arr = [1., 1.]
init_alpha_choice = [0.5]
current_kde, errorkdeval, errorbBW, erroraALP = get_kde_obj_eval(sample, xy_grid_pts, init_rescale_arr, init_alpha_choice,  min_bw_dL=min_bw_dL)
bwx, bwy = errorbBW[0], errorbBW[1]
# reshape KDE to XX grid shape 
ZZ = errorkdeval.reshape(XX.shape)
u_plot.new2DKDE(XX, YY,  ZZ,  meanxi1, meanxi2, saveplot=True,  show_plot=False, pathplot=opts.pathplot, plot_label='KDE', title='median')
#STEP III: get pdet on same grid as KDE eval and plot contours of pdet
# try on top of scatter of m1-dL vals
if opts.fpopchoice == 'rate':
    pdet2D = np.zeros((len(p1grid), len(p2grid)))
    m1_source_grid = p1grid.copy()
    dL_grid = p2grid.copy()
    #convert masses im detector frame to make sure we are correctly computing pdet on same masses as KDE grid masses
    m1_det_grid = get_mass_in_detector_frame(dL_grid, m1_source_grid) 
    for i, m1val in enumerate(m1_det_grid):
        for j, dLval in enumerate(dL_grid):
            pdet2D[i, j] = u_pdet.pdet_of_m1_dL_powerlawm2(m1val, m2min, dLval, beta=index_powerlaw_m2, classcall=g)

capped_pdet2D = np.maximum(pdet2D, opts.max_pdet)
u_plot.plot_pdet2D(XX, YY, pdet2D, Maxpdet=opts.max_pdet, pathplot=opts.pathplot, show_plot=False)

#get rates
current_rateval = len(meanxi1)*ZZ/capped_pdet2D.T
u_plot.new2DKDE(XX, YY,  current_rateval, meanxi1, meanxi2 , saveplot=True,plot_label='Rate', title='median', show_plot=False, pathplot=opts.pathplot)
#save data in hdf5 file
frateh5 = h5.File(opts.output_filename+'dL2priorfactor_uniform_prior_mass_2Drate_m1'+opts.parameter2+'max_pdet_cap_'+str(opts.max_pdet)+'min_bw_dL'+str(opts.min_bw_dLdim)+'.hdf5', 'w')
dsetxx = frateh5.create_dataset('data_xx', data=XX)
dsetxx.attrs['xname']='xx'
dsetyy = frateh5.create_dataset('data_yy', data=YY)
dsetxx.attrs['yname']='yy'
med_group = frateh5.create_group(f'median_iteration')
    # Save the data in the group
med_group.create_dataset('rwsamples', data=sample)
med_group.create_dataset('alpha', data=erroraALP)
med_group.create_dataset('bwx', data=bwx)
med_group.create_dataset('bwy', data=bwy)
med_group.create_dataset('kde', data=errorkdeval)
frateh5.create_dataset('pdet2D', data=pdet2D.T)
frateh5.create_dataset('pdet2Dwithcap', data=capped_pdet2D.T)
#STEP IV Iterative reweighting main method: saving data make final plots average of 1000 KDEs/Rates/bwd
Total_Iterations = int(opts.NIterations)
discard = int(opts.buffer_start)   # how many iterations to discard default =5
Nbuffer = int(opts.buffer_interval) #100 buffer [how many (x-numbers of ) previous iteration to use in reweighting with average of those past (x-numbers of ) iteration results
iterkde_list = []
iter2Drate_list = []
iterbwxlist = []
iterbwylist = []
iteralplist = []

#set the prior factors correctly here before reweighting
prior_kwargs = {'dl_prior_power': opts.dl_prior_power, 'redshift_prior_power': opts.redshift_prior_power}

for i in range(Total_Iterations + discard):
    print("i - ", i)
    if i >= discard + Nbuffer:
        buffer_kdes_mean = np.mean(iterkde_list[-Nbuffer:], axis=0)
        buffer_interp = RegularGridInterpolator((p1grid, p2grid), buffer_kdes_mean.T, bounds_error=False, fill_value=0.0)
    rwsamples = []
    for samplem1, samplem2, redshiftvals, pdet_k in zip(sampleslists1, sampleslists2, redshift_lists, pdetlists):
        samples= np.vstack((samplem1, samplem2)).T
        if i < discard + Nbuffer :
            rwsample = get_reweighted_sample(samples, redshiftvals, pdet_k, current_kde, bootstrap=opts.bootstrap_option, prior_factor_kwargs=prior_kwargs, max_pdet_cap=opts.max_pdet)
        else:
            rwsample= mean_bufferkdelist_reweighted_samples(samples, redshiftvals, pdet_k, buffer_interp, bootstrap_choice=opts.bootstrap_option, prior_factor_kwargs=prior_kwargs, max_pdet_cap=opts.max_pdet)
        rwsamples.append(rwsample)
    if opts.bootstrap_option =='poisson':
        rwsamples = np.concatenate(rwsamples)
    print("iter", i, "  totalsamples = ", len(rwsamples))
    current_kde, current_kdeval, shiftedbw, shiftedalp = get_kde_obj_eval(np.array(rwsamples), xy_grid_pts, init_rescale_arr, init_alpha_choice,  input_transf=('log', 'none'),  min_bw_dL=min_bw_dL)
    bwx, bwy = shiftedbw[0], shiftedbw[1]
    print("bwvalues", bwx, bwy)
    current_kdeval = current_kdeval.reshape(XX.shape)
    iterkde_list.append(current_kdeval)
    iterbwxlist.append(bwx)
    iterbwylist.append(bwy)
    iteralplist.append(shiftedalp)
    group = frateh5.create_group(f'iteration_{i}')
    # Save the data in the group
    group.create_dataset('rwsamples', data=rwsamples)
    group.create_dataset('alpha', data=shiftedalp)
    group.create_dataset('bwx', data=bwx)
    group.create_dataset('bwy', data=bwy)
    group.create_dataset('kde', data=current_kdeval)
    frateh5.flush()
    if opts.fpopchoice == 'rate':
        current_kdeval = current_kdeval.reshape(XX.shape)
        current_rateval = len(rwsamples)*current_kdeval/capped_pdet2D.T
        iter2Drate_list.append(current_rateval)

    if i > 1 and i%Nbuffer==0:
        iterstep = int(i)
        print(iterstep)
        u_plot.histogram_datalist(iterbwxlist[-Nbuffer:], dataname='bwx', pathplot=opts.pathplot, Iternumber=iterstep)
        u_plot.histogram_datalist(iterbwylist[-Nbuffer:], dataname='bwy', pathplot=opts.pathplot, Iternumber=iterstep)
        u_plot.histogram_datalist(iteralplist[-Nbuffer:], dataname='alpha', pathplot=opts.pathplot, Iternumber=iterstep)
        u_plot.average2DlineardLrate_plot(meanxi1, meanxi2, XX, YY, iterkde_list[-Nbuffer:], pathplot=opts.pathplot, titlename=i, plot_label='KDE', x_label='m1', y_label='dL', show_plot= False)
        if opts.fpopchoice == 'rate':
             u_plot.average2DlineardLrate_plot(meanxi1, meanxi2, XX, YY, iter2Drate_list[-Nbuffer:], pathplot=opts.pathplot, titlename=i, plot_label='Rate', x_label='m1', y_label='dL', show_plot= False)
frateh5.close()
u_plot.average2DlineardLrate_plot(meanxi1, meanxi2, XX, YY, iterkde_list[discard:], pathplot=opts.pathplot+'allKDEscombined_', titlename=1001, plot_label='KDE', x_label='m1', y_label='dL', show_plot= False)
u_plot.average2DlineardLrate_plot(meanxi1, meanxi2, XX, YY, iter2Drate_list[discard:], pathplot=opts.pathplot+'allKDEscombined_', titlename=1001, plot_label='Rate', x_label='m1', y_label='dL', show_plot= False)

#alpha bw plots
u_plot.bandwidth_correlation(iterbwxlist, number_corr=discard, error=0.02,  pathplot=opts.pathplot+'bwx_')
u_plot.bandwidth_correlation(iterbwylist, number_corr=discard, error=0.02,  pathplot=opts.pathplot+'bwy_')
u_plot.bandwidth_correlation(iteralplist, number_corr=discard,  error=0.02, param='alpha',pathplot=opts.pathplot, log=False)
########################### One D rate pltos with correct Units of comoving Volume 
#STEP V: plot rate(m1) with error bars  (90th percentile 5th-95th percentile)
rate_lnm1dLmed = np.percentile(iter2Drate_list[discard:], 50., axis=0)
rate_lnm1dL_5 = np.percentile(iter2Drate_list[discard:], 5., axis=0)
rate_lnm1dL_95 = np.percentile(iter2Drate_list[discard:], 95., axis=0)




for val in [300, 500, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3500, 4000]:
    closest_index = np.argmin(np.abs(YY - val))
    fixed_dL_value = YY.flat[closest_index]
    print(fixed_dL_value)
    indices = np.isclose(YY, fixed_dL_value)

    # Extract the slice of rate_lnm1dL for the specified dL
    rate_lnm1_slice50 = rate_lnm1dLmed[indices]
    rate_lnm1_slice5 = rate_lnm1dL_5[indices]
    rate_lnm1_slice95 = rate_lnm1dL_95[indices]

    # Extract the corresponding values of lnm1 from XX
    m1_values = XX[indices]
    print(m1_values)

    plt.figure(figsize=(8, 6))
    plt.plot(m1_values, rate_lnm1_slice50,  linestyle='-', color='k', lw=2)
    plt.plot(m1_values, rate_lnm1_slice5,  linestyle='--', color='r', lw=1.5)
    plt.plot(m1_values, rate_lnm1_slice95,  linestyle='--', color='r', lw=1.5)
    plt.xlabel(r'$m_{1,\, source}$')
    plt.ylabel(r'$\mathrm{d}\mathcal{R}/m_1\mathrm{d}d_L [\mathrm{Mpc}^{-1}\, (\mathrm{m}/{M}_\odot)^{-1}  \mathrm{yr}^{-1}]$',fontsize=18)
    plt.title(r'$d_L=${0}[Mpc]'.format(val))
    plt.semilogx()
    plt.semilogy()
    plt.ylim(ymin=1e-6)
    plt.grid(True)
    plt.savefig(opts.pathplot+'OneD_rate_m1_slicedL{0:.1f}.png'.format(val))
    plt.close()
    print("done")
