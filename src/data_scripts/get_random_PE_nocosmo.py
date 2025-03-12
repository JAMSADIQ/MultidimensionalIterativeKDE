#python  get_gwtc_data_samples.py  --parameter1 mass_1_source --rsample  100  --o2filesname  GWTC1/*.h5  --o3afilesname  GWTC2_1/NONcosmo/*.h5  --useO3b  --o3bfilesname GWTC3/NONcosmo*.h5  --tag GWTC3  --eventsType  BBH 

import argparse
import os 
import numpy as np
import h5py 
import ntpath

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--parameters", nargs='+', default=['mass_1_source', 'mass_2_source', 'luminosity_distance', 'redshift', 'chi_eff'], help="names of params to retrieve from PE samples")
parser.add_argument("--rsample", default=100, type=int, help="number of random samples")
parser.add_argument("--seed", default=12345, type=int, help="Random seed")
parser.add_argument("--o2filesname", nargs='+', help="List of paths to O1 and O2 PE eventname.h5 files")
parser.add_argument("--o3afilesname", nargs='+', help="List of paths to O3a PE eventname.h5 files only comoving frame")
parser.add_argument("--o3bfilesname", nargs='+', help="List of paths to O3b PE eventname.h5 files only comoving frame")
parser.add_argument("--tag", default="GWTC3", required=True, help="String to label output file")
parser.add_argument("--inverse-chieff-prior-weight", action='store_true', help="If given, weight samples by the inverse of the PE prior over chi_eff")
parser.add_argument("--max-a", type=float, default=0.999, help="Max spin magnitude for chieff prior calculation. Default 0.999")
parser.add_argument("--eventsType", default="BBH", required=True, help="description of type of events used. AllCompactObject or BBH")
parser.add_argument("--min-median-mass", default=3.0, type=float, required=True, help="If eventType BBH is chosen, remove events with one or both component median masses below given value")
parser.add_argument('--pathdata', default='./', help='directory to save data file')
opts = parser.parse_args()

# Empty list if any set of files is not specified
if opts.o2filesname is None:
    opts.o2filesname = []
if opts.o3afilesname is None:
    opts.o3afilesname = []
if opts.o3bfilesname is None:
    opts.o3bfilesname = []


def compute_statistics(data):
    """Compute median, mean, and standard deviation of the data."""
    return {
        "median": np.percentile(data, 50),
        "mean": np.mean(data),
        "std": np.std(data)
    }


def sample_random_indices(data_length, num_samples, seed, weights=None):
    """Get random sample indices with a consistent seed."""
    rng = np.random.default_rng(seed)

    if weights is None:
        return rng.choice(np.arange(data_length), num_samples)
    # Use weights : normalize for safety
    sample_p = weights / np.sum(weights, dtype=float)
    return rng.choice(np.arange(data_length), num_samples, p=sample_p)


# Initialize storage
par_names = {
             'mass_1_source': 'm1src',
             'mass_2_source': 'm2src',
             'luminosity_distance': 'dL',
             'redshift': 'redshift',
             'chi_eff': 'chieff',
}

event_data = {"names": [],}
for p in opts.parameters:
    # nested dicts
    event_data[par_names[p]] = {"median": [], "mean": [], "std": [], "samples": []}


def path_leaf(path):
    """
    from given long path extract filename 
    in data directory
    """
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def sampledata(d):
    """
    given a dictionary d
    return sample list of values 
    from all keys combined 
    """
    nd = dict(d)
    return np.concatenate([nd[key] for key in nd.keys()])

# TD : function does not seem to be used
# For bootstrap case:
def randomsampledata(d):
    """
    given a dictionary d
    return list random  values 
    from all keys such that 
    some keys all values can be repeated 
     multiple times
    """
    nd = dict(d)
    #create an ndarray arrays of array
    arr = np.array([nd[key] for key in nd.keys()])
    rng = np.random.default_rng()
    random_arr = rng.choice(arr, len(arr))
    #choose randomly set of arrays in list of arrays
    return np.concatenate(random_arr)


def gwdata_save(datfilename, eventslist, pesamplelists, meanxi, sigmaxi):
    """
    give file name and from input options of gw files
    return h5 file
    with one group of mean value of each  event pe sample
    one group with each event errored sample that uses
    events std and add error to the mean values
    and third group to give 100 pe samples using seed we define
    """
    #open an h5 file
    dataf = h5py.File(datfilename, 'w')
    grp_data = dataf.create_group('randdata')
    grp_initial = dataf.create_group('initialdata')
    grp_error = dataf.create_group('erroreddata')
    grp_sigma = dataf.create_group('sigmaerror')
    error_meanxi = np.zeros_like(meanxi)
    for i in range(len(meanxi)):
        error_meanxi[i] = np.random.normal(meanxi[i], sigmaxi[i])
        grp_data.create_dataset(eventslist[i], data=pesamplelists[i])
    grp_initial.create_dataset('original_mean', data=np.array(meanxi))
    grp_error.create_dataset('errored_mean', data=error_meanxi)
    grp_sigma.create_dataset('errored_sigma', data=sigmaxi)
    dataf.close()
    return meanxi, error_meanxi, pesamplelists 


def Neff(weights):
    """
    Effective sample size in importance sampling with given weights
    """
    w = np.array(weights)
    return w.sum() ** 2. / (w ** 2.).sum()


for f in opts.o2filesname:
    eventnamef = os.path.splitext(path_leaf(f))[0]
    dat = h5py.File(f, 'r')[eventnamef+'/posterior_samples']
    m1vals, m2vals = dat['mass_1_source'][:], dat['mass_2_source'][:]

    # If the BBH option is given masses must satisfy a criterion otherwise continue
    if opts.eventsType == 'BBH':
        m1med, m2med = np.percentile(m1vals, 50), np.percentile(m2vals, 50)
        if not(m1med >= opts.min_median_mass and m2med >= opts.min_median_mass):
            print(f"This is a nonBBH event {eventnamef} with m1, m2 = {m1med:.3f}, {m2med:.3f}")
            continue  # don't process the file
        else:
            print(f"This is a BBH event {eventnamef} with m1, m2 = {m1med:.2f}, {m2med:.2f}")

    # Generate random indices once for all parameters
    if opts.inverse_chieff_prior_weight:
        import sys ; sys.path.append('../kde_analysis')
        from priors_vectorize import chi_effective_prior_from_isotropic_spins as chieff_prior_iso_vec
        chivals = dat['chi_eff'][:]
        weights = 1./chieff_prior_iso_vec(m2vals/m1vals, opts.max_a, chivals)
        print(f'chi_eff based weights cover {weights.min():.2f} - {weights.max():.1f}, '
              f'Neff {Neff(weights):.1f}')
    else:
        weights = None
    random_idx = sample_random_indices(len(m1vals), opts.rsample, opts.seed, weights)

    # Process each parameter
    for p in opts.parameters:
        k = par_names[p]
        data = dat[p].astype(float)
        stats = compute_statistics(data)

        event_data[k]["median"].append(stats["median"])
        event_data[k]["mean"].append(stats["mean"])
        event_data[k]["std"].append(stats["std"])
        event_data[k]["samples"].append(data[random_idx])

    event_data["names"].append(eventnamef)

for f in opts.o3afilesname:
    eventnamef = os.path.splitext(path_leaf(f))[0]
    dat = h5py.File(f, 'r')['C01:Mixed/posterior_samples']
    m1vals, m2vals = dat['mass_1_source'][:], dat['mass_2_source'][:]

    # If the BBH option is given masses must satisfy a criterion otherwise continue
    if opts.eventsType == 'BBH':
        m1med, m2med = np.percentile(m1vals, 50), np.percentile(m2vals, 50)
        if not(m1med >= opts.min_median_mass and m2med >= opts.min_median_mass):
            print(f"This is a nonBBH event {eventnamef} with m1, m2 = {m1med:.3f}, {m2med:.3f}")
            continue  # don't process the file
        else:
            print(f"This is a BBH event {eventnamef} with m1, m2 = {m1med:.2f}, {m2med:.2f}")

    # Generate random indices once for all parameters
    if opts.inverse_chieff_prior_weight:
        chivals = dat['chi_eff'][:]
        weights = 1./chieff_prior_iso_vec(m2vals/m1vals, opts.max_a, chivals)
        print(f'chi_eff based weights cover {weights.min():.2f} - {weights.max():.1f}, '
              f'Neff {Neff(weights):.1f}')
    else:
        weights = None
    random_idx = sample_random_indices(len(m1vals), opts.rsample, opts.seed, weights)

    # Process each parameter
    for p in opts.parameters:
        k = par_names[p]
        data = dat[p].astype(float)
        stats = compute_statistics(data)

        event_data[k]["median"].append(stats["median"])
        event_data[k]["mean"].append(stats["mean"])
        event_data[k]["std"].append(stats["std"])
        event_data[k]["samples"].append(data[random_idx])

    event_data["names"].append(eventnamef)

for f in opts.o3bfilesname:
    eventnamef = os.path.splitext(path_leaf(f))[0]
    dat = h5py.File(f, 'r')['C01:Mixed/posterior_samples']
    m1vals, m2vals = dat['mass_1_source'][:], dat['mass_2_source'][:]

    # If the BBH option is given masses must satisfy a criterion otherwise continue
    if opts.eventsType == 'BBH':
        m1med, m2med = np.percentile(m1vals, 50), np.percentile(m2vals, 50)
        if not(m1med >= opts.min_median_mass and m2med >= opts.min_median_mass):
            print(f"This is a nonBBH event {eventnamef} with m1, m2 = {m1med:.3f}, {m2med:.3f}")
            continue  # don't process the file
        else:
            print(f"This is a BBH event {eventnamef} with m1, m2 = {m1med:.2f}, {m2med:.2f}")

    # Generate random indices once for all parameters
    if opts.inverse_chieff_prior_weight:
        chivals = dat['chi_eff'][:]
        weights = 1./chieff_prior_iso_vec(m2vals/m1vals, opts.max_a, chivals)
        print(f'chi_eff based weights cover {weights.min():.2f} - {weights.max():.1f}, '
              f'Neff {Neff(weights):.1f}')
    else:
        weights = None
    random_idx = sample_random_indices(len(m1vals), opts.rsample, opts.seed, weights)

    # Process each parameter
    for p in opts.parameters:
        k = par_names[p]
        data = dat[p].astype(float)
        stats = compute_statistics(data)

        event_data[k]["median"].append(stats["median"])
        event_data[k]["mean"].append(stats["mean"])
        event_data[k]["std"].append(stats["std"])
        event_data[k]["samples"].append(data[random_idx])

    event_data["names"].append(eventnamef)


gwdata_save(opts.tag+'_m1src.h5', event_data["names"], event_data["m1src"]["samples"], event_data["m1src"]["mean"], event_data["m1src"]["std"])
gwdata_save(opts.tag+'_m2src.h5', event_data["names"], event_data["m2src"]["samples"], event_data["m2src"]["mean"], event_data["m2src"]["std"])
gwdata_save(opts.tag+'_dl.h5', event_data["names"], event_data["dL"]["samples"], event_data["dL"]["mean"], event_data["dL"]["std"])
gwdata_save(opts.tag+'_redshift.h5', event_data["names"], event_data["redshift"]["samples"], event_data["redshift"]["mean"], event_data["redshift"]["std"])
gwdata_save(opts.tag+'_chieff.h5', event_data["names"], event_data["chieff"]["samples"], event_data["chieff"]["mean"], event_data["chieff"]["std"])
