#python  get_gwtc_data_samples.py  --parameter1 mass_1_source --rsample  100  --o2filesname  GWTC1/*.h5  --o3afilesname  GWTC2_1/NONcosmo/*.h5  --useO3b  --o3bfilesname GWTC3/NONcosmo*.h5  --tag GWTC3  --eventsType  BBH 

import argparse
import os 
import numpy as np
import h5py 
import ntpath

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--parameter1", type=str, default='mass_1_source', help="name of param")
parser.add_argument("--parameter2", type=str, default='mass_2_source', help="name of param")
parser.add_argument("--parameter3", type=str, default='luminosity_distance', help="name of param")
parser.add_argument("--parameter4", type=str, default='redshift', help="name of param")
parser.add_argument("--parameter5", type=str, default='xi_eff', help="name of param")
parser.add_argument("--rsample", default=100, type=int, help="number of random samples")
parser.add_argument("--seed", default=12345, type=int, help="Random seed")
parser.add_argument('--pathh5files', type=str, default='/home/jam.sadiq/PopModels/projectKDE/Analysis/Data/public_data/latest_O3a_O3b_public_data/', help="directory where .h5 files of latest PE data is saved")
parser.add_argument("--o2filesname", nargs="+", type=str, required=True, help="List of all O1 and O2 PE  eventname.h5   files")
parser.add_argument("--o3afilesname", nargs="+", type=str, required=True, help="List of all O3a PE eventname.h5   files  only comoving frame")
parser.add_argument("--useO3b", action='store_true',help="if we want O3b files use this command otherwise dont use this option")
parser.add_argument("--o3bfilesname", nargs="+", type=str,help="List of all O3b PE eventname.h5   files  only comoving frame")
parser.add_argument("--tag", default="GWTC3", type=str, required=True, help="initial name of outputfile based on what catalog events has been used")
parser.add_argument("--eventsType", default="BBH", type=str, required=True, help="description of type of events used. AllCompactObject or BBH")
parser.add_argument("--min-median-mass", default=3.0, type=float, required=True, help="For only BBH events use 3.0 or large, for all events including BNS or NSBH use smaller than 3.0 value what is suitable 1.0")
parser.add_argument('--pathdata', default='./', type=str, help='directory to save data file')
opts = parser.parse_args()


def compute_statistics(data):
    """Compute median, mean, and standard deviation of the data."""
    return {
        "median": np.percentile(data, 50),
        "mean": np.mean(data),
        "std": np.std(data)
    }

def sample_random_indices(data_length, num_samples, seed):
    """Get random sample indices with a consistent seed."""
    rng = np.random.default_rng(seed)
    return rng.choice(np.arange(data_length), num_samples)

# Initialize storage
event_data = {
    "names": [],
    "datam1src": {"median": [], "mean": [], "std": [], "samples": []},
    "datam2src": {"median": [], "mean": [], "std": [], "samples": []},
    "data_dL": {"median": [], "mean": [], "std": [], "samples": []},
    "data_redshift": {"median": [], "mean": [], "std": [], "samples": []},
    "data_xieff": {"median": [], "mean": [], "std": [], "samples": []}
}


def path_leaf(path):
    """
    from given long path extract filename 
    in data directory
    """
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

#### Function for 100 PE samples KDEs ###
def sampledata(d):
    """
    given a dictionary d
    return sample list of values 
    from all keys combined 
    """
    nd = dict(d)
    return np.concatenate([nd[key] for key in nd.keys()])

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

##################
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


for f in opts.o2filesname:
    eventnamef  = os.path.splitext(path_leaf(opts.pathh5files + f))[0]
    print(eventnamef)
    dat = h5py.File(opts.pathh5files + f, 'r')[eventnamef+'/posterior_samples']
    #if BBH it must satisfy a criteria  otherwise continue
    if opts.eventsType == 'BBH':
        m1vals, m2vals = dat['mass_1_source'], dat['mass_2_source']
        m1med, m2med = np.percentile(m1vals,100*0.50), np.percentile(m2vals,100*0.50)
        if (m1med >= opts.min_median_mass and m2med >= opts.min_median_mass):
            print("This is a BBH event with GW name with m1, m2  = ", eventnamef, m1med, m2med)
            # Generate random indices once for all parameters
            random_indices = sample_random_indices(len(dat[opts.parameter1]), opts.rsample, opts.seed)

            # Process each parameter
            for param, key in zip(
                [opts.parameter1, opts.parameter2, opts.parameter3, opts.parameter4, opts.parameter5],
                ["datam1src", "datam2src", "data_dL", "data_redshift", "data_xieff"]
            ):
                data = dat[param].astype(float)
                stats = compute_statistics(data)
                samples = data[random_indices]
                
                event_data[key]["median"].append(stats["median"])
                event_data[key]["mean"].append(stats["mean"])
                event_data[key]["std"].append(stats["std"])
                event_data[key]["samples"].append(samples)
            
            event_data["names"].append(eventnamef)

        else:
            print("This is a nonBBH events with GW name with m1, m2 = ", eventnamef, m1med, m2med)
            continue
    else:
        print("check event type option")

for f in opts.o3afilesname:
    print("f", f)
    eventnamef  = os.path.splitext(path_leaf(opts.pathh5files + f))[0]
    print(eventnamef)
    data = h5py.File(f, 'r')
    dat = data['C01:Mixed/posterior_samples']
    #if BBH it must satisfy a criteria  otherwise continue
    if opts.eventsType == 'BBH':
        m1vals, m2vals = dat['mass_1_source'], dat['mass_2_source']
        m1med, m2med = np.percentile(m1vals,100*0.50), np.percentile(m2vals,100*0.50)
        if (m1med >= opts.min_median_mass and m2med >= opts.min_median_mass):
            print("This is a BBH event with GW name with m1, m2  = ", eventnamef, m1med, m2med)
            # Generate random indices once for all parameters
            random_indices = sample_random_indices(len(dat[opts.parameter1]), opts.rsample, opts.seed)
            
            # Process each parameter
            for param, key in zip(
                [opts.parameter1, opts.parameter2, opts.parameter3, opts.parameter4, opts.parameter5],
                ["datam1src", "datam2src", "data_dL", "data_redshift", "data_xieff"]
            ):
                data = dat[param].astype(float)
                stats = compute_statistics(data)
                samples = data[random_indices]
                
                event_data[key]["median"].append(stats["median"])
                event_data[key]["mean"].append(stats["mean"])
                event_data[key]["std"].append(stats["std"])
                event_data[key]["samples"].append(samples)
            
            event_data["names"].append(eventnamef)
        else:
            print("This is a nonBBH events with GW name with m1, m2 = ", eventnamef, m1med, m2med)
            continue
    else:
        print("check event type option")

if opts.useO3b:
    for f in opts.o3bfilesname:
        eventnamef  = os.path.splitext(path_leaf(opts.pathh5files + f))[0]
        dat = h5py.File(f, 'r')['C01:Mixed/posterior_samples']
        ##if BBH it must satisfy a criteria  otherwise continue
        if opts.eventsType == 'BBH':
            m1vals, m2vals = dat['mass_1_source'], dat['mass_2_source']
            m1med, m2med = np.percentile(m1vals,100*0.50), np.percentile(m2vals,100*0.50)
            if (m1med >= opts.min_median_mass and m2med >= opts.min_median_mass):
                print("This is a BBH event with GW name with m1, m2  = ", eventnamef, m1med, m2med)
                # Generate random indices once for all parameters
                random_indices = sample_random_indices(len(dat[opts.parameter1]), opts.rsample, opts.seed)

                # Process each parameter
                for param, key in zip(
                    [opts.parameter1, opts.parameter2, opts.parameter3, opts.parameter4, opts.parameter5],
                    ["datam1src", "datam2src", "data_dL", "data_redshift", "data_xieff"]
                ):
                    data = dat[param].astype(float)
                    stats = compute_statistics(data)
                    samples = data[random_indices]

                    event_data[key]["median"].append(stats["median"])
                    event_data[key]["mean"].append(stats["mean"])
                    event_data[key]["std"].append(stats["std"])
                    event_data[key]["samples"].append(samples)

                event_data["names"].append(eventnamef)

            else:
                print("This is a nonBBH events with GW name with m1, m2 = ", eventnamef, m1med, m2med)
                continue
        else:
            print("check event type option")

# Save data files
def save_gw_data(filename, names, samples, means, stds):
    """Save GW data into a file."""
    gwdata_save(
        filename,
        names,
        samples,
        means,
        stds
    )

save_gw_data(opts.tag+'_m1src.h5', event_data["names"], event_data["datam1src"]["samples"], event_data["datam1src"]["mean"], event_data["datam1src"]["std"])
save_gw_data(opts.tag+'_m2src.h5', event_data["names"], event_data["datam2src"]["samples"], event_data["datam2src"]["mean"], event_data["datam2src"]["std"])
save_gw_data(opts.tag+'_dl.h5', event_data["names"], event_data["data_dL"]["samples"], event_data["data_dL"]["mean"], event_data["data_dL"]["std"])
save_gw_data(opts.tag+'_redshift.h5', event_data["names"], event_data["data_redshift"]["samples"], event_data["data_redshift"]["mean"], event_data["data_redshift"]["std"])
save_gw_data(opts.tag+'_chieff.h5', event_data["names"], event_data["data_xieff"]["samples"], event_data["data_xieff"]["mean"], event_data["data_xieff"]["std"])
