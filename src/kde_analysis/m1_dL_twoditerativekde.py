import sys
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
# Input files #maybe we can combine these three to one
parser.add_argument('--datafilename1', required=True, help='H5 file containing N samples for m1.')
parser.add_argument('--datafilename2', required=True, help='H5 file containing N samples for parameter2.')
parser.add_argument('--datafilename3', required=True, help='H5 file containing N samples of parameter3.')
# Parameters
parser.add_argument('--parameter1', default='m1', help='Name of parameter for x-axis of KDE.')
parser.add_argument('--parameter2', default='m2', help='Name of parameter for y-axis of KDE.')
parser.add_argument('--parameter3', default='dL', help='Name of parameter for z-axis of KDE.')
parser.add_argument('--logkde', action='store_true', help='If set, perform KDE in logarithmic space.')

# KDE grid options: limits and resolution
parser.add_argument('--m1-min', default=5.0, type=float, help='Minimum value for primary mass m1.')
parser.add_argument('--m1-max', default=100.0, type=float, help='Maximum value for primary mass m1.')
parser.add_argument('--Npoints', default=200, type=int, help='Number of points for KDE evaluation.')
parser.add_argument('--param2-min', default=4.95, type=float, help='Minimum value for parameter 2 if it is  m2, else if dL use 10')
parser.add_argument('--param2-max', default=100.0, type=float, help='Maximum value for parameter 2 if it is m2 else if dL  use 10000')
parser.add_argument('--param3-min', default=10., type=float, help='Minimum value for parameter 3 if it is  dL, else if Xieff use -1')
parser.add_argument('--param3-max', default=10000., type=float, help='Maximum value for parameter 3 if it is dL else if Xieff  use +1')

#Expectation Maximization Algorithm Reweighting
parser.add_argument('--reweight-method', default='bufferkdevals', type=str, help=('Reweighting method for Gaussian sample shift: ''"bufferkdevals" uses buffered KDE values, and "bufferkdeobject" uses buffered KDE objects.'))
parser.add_argument('--reweight-sample-option', default='reweight', type=str, help=('Option to reweight samples: choose "noreweight" to skip reweighting or "reweight" ''to use fpop probabilities for reweighting. If "reweight", one sample is used ''for no bootstrap, and multiple samples are used for Poisson.'))
parser.add_argument('--bootstrap-option', default='poisson', type=str, help=('Bootstrap method: choose "poisson" for Poisson resampling or "nopoisson" to skip it. ''If "None", reweighting will be based on fpop probabilities, with a single reweighted sample for each event.')
)
parser.add_argument('--buffer-start', default=100, type=int, help='The starting iteration for buffering in the reweighting process.')
parser.add_argument('--buffer-interval', default=100, type=int, help=('The interval of the buffer determines how many previous iteration results are used in the next iteration for reweighting.'))
parser.add_argument('--NIterations', default=1000, type=int, help='Total number of iterations for the reweighting process.')
parser.add_argument('--Maxpdet', default=0.1, type=float, help='Capping value for small pdet to introduce regularization.')
# Output and plotting
parser.add_argument('--pathplot', default='./', help='Path to save plots.')
parser.add_argument('--output-filename', default='output_data', help='Base name for output HDF5 files.')
opts = parser.parse_args()


# Define cosmology
H0 = 67.9  # km/s/Mpc
omega_m = 0.3065
cosmo = FlatLambdaCDM(H0=H0, Om0=omega_m)

def get_redshift_from_distance(dL_Mpc):
    """
    Compute the redshift corresponding to a luminosity distance.

    Args:
        dL_Mpc (float): Luminosity distance in megaparsecs (Mpc).

    Returns:
        float: Corresponding redshift.
    """
    return z_at_value(cosmo.luminosity_distance, dL_Mpc * u.Mpc)


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


def compute_pdet(mass_detector_frame, dL, beta, class_instance):
    """
    Compute detection probability using given parameters.

    Args:
        mass_detector_frame (float): Mass in the detector frame.
        dL (float): Luminosity distance.
        beta (float): Power-law index.
        class_instance: Instance of a class with a method to compute pdet.

    Returns:
        float: Detection probability.
    """
    return class_instance.pdet_of_m1_dL_powerlawm2(
        mass_detector_frame, 5.0, dL, beta=beta
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


def prior_factor_function(samples, logkde=False):
    """
    Compute a prior factor for reweighting based on uniform priors.

    Args:
        samples (np.ndarray): Array of samples with shape (N, 2), where N is the number of samples.
        logkde (bool): If True, the parameters are logarithmic.
        Note in new code prior conversion is done inside code
        so here we only need d_L prior factor
    Returns:
        np.ndarray: Prior factor for each sample.
    """
    m1_values, dL_values = samples[:, 0], samples[:, 1]
    if logkde:
        return 1.0 / (dL_values ** 2)
    return np.ones_like(m1_values)


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


def get_reweighted_sample(original_samples, pdetvals, fpop_kde, bootstrap='poisson', prior_factor=prior_factor_function, max_pdet_cap=0.1):
    """
    Generate  reweighted sample/samples from the original samples based on their probabilities under a given kernel density estimate (KDE) and optional transformations.

    Parameters:
    ----------
    original_samples : list or array-like
        A collection of PE samples of an event

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
        logarithmic priors.

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
    - If `opts.logkde` is set (assumed to be a global variable), the prior factor is
      applied to adjust the KDE probabilities.
    - Sampling is performed using `np.random.choice`, with probabilities normalized
      to sum to 1.

    """
    # Compute KDE probabilities divide by regularized pdetvals
    fkde_samples = fpop_kde.evaluate_with_transf(original_samples) / apply_max_cap_function(pdetvals, max_pdet_cap)

    # Adjust probabilities based on the prior factor, if opts.logkde is set
    if opts.logkde:
        frate_atsample = fkde_samples * prior_factor(original_samples)
    else:
        frate_atsample = fkde_samples

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


def mean_bufferkdelist_reweighted_samples(original_samples, pdetvals, mean_kde_interp, bootstrap_choice='poisson', prior_factor=prior_factor_function, max_pdet_cap=0.1):
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
    kde_interp_vals = mean_kde_interp(original_samples)/apply_max_cap_function(pdetvals, max_pdet_cap)
    if opts.logkde:
        kde_interp_vals  *= prior_factor(original_samples)
    norm_mediankdevals = kde_interp_vals/sum(kde_interp_vals)
    rng = np.random.default_rng()
    if bootstrap_choice =='poisson':
        reweighted_sample = rng.choice(original_samples, np.random.poisson(1), p=norm_mediankdevals)
    else:
        reweighted_sample = rng.choice(original_samples, p=norm_mediankdevals)
    return reweighted_sample


def get_kde_obj_eval(sample, eval_pts, rescale_arr, alphachoice, input_transf=('log', 'none')):
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
    kde_object = ad.KDERescaleOptimization(sample, stdize=True, rescale=rescale_arr, alpha=alphachoice, dim_names=['lnm1', 'dL'], input_transf=input_transf)
    dictopt, score = kde_object.optimize_rescale_parameters(rescale_arr, alphachoice, bounds=((0.01,100),(0.01, 100),(0,1)), disp=True, tol=0.1)#, xatol=0.01, fatol=0.1)
    kde_vals = kde_object.evaluate_with_transf(eval_pts)
    optbwds = [1.0/dictopt[0], 1.0/dictopt[1]]
    optalpha = dictopt[-1]
    print("opt results = ", dictopt)
    return  kde_object, kde_vals, optbwds, optalpha



# Main execution begins here
#STEP I: call the PE sample data and get PDET on PE samples using power law on m2

#plot of pdet as function of m1-dL after removing bad samples

#STEP II: get median PE sample KDE plot it


#STEP III: get pdet on same grid as KDE eval and plot contours of pdet
# try on top of scatter of m1-dL vals


#STEP IV Iterative reweighting main method: saving data make final plots average of 1000 KDEs/Rates/bwd


#STEP V rate(m1) with error bars  (90th percentile 5th-95th percentile)
