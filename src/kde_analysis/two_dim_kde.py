# Jam Sadiq: script for m1-dL KDe with power law on mass ratio
#log spacing in m1-grid and linear-spacing in dL grid
#in KDE estimate KDEpy will use lnm1, lnm2 and reconvert the results in input frames
#so we need only uniform prior in reweighting do not 1/m factor for non uniform prior
#nb: dL must have a prior of 1/dL^2 though
#Clean data with no samples very far form the rest should be used
#make sure pdet and KDE are compute on source mass [for pdet convert source -> det mass]
#Save data of KDE/ PDET (not rate)  byt also reweighted  samples and KDE params which can be used to recomputeKDE for each iterative steps?:
#Data in one HDF file is saved in AnalysisCodes


import sys
sys.path.append('/home/jam.sadiq/PopModels/selectioneffects/cbc_pdet/pop-de/popde/')
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
rcParams["grid.linestyle"] = ':'
rcParams["grid.alpha"] = 0.6


import utils_plot as u_plot
import o123_class_found_inj_general as u_pdet

#careful parsers 
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--datafilename1', help='h5 file containing N samples for m1for all gw bbh event')
parser.add_argument('--datafilename2', help='h5  file containing N sample of parameter2 for each event, ')
parser.add_argument('--datafilename3', help='h5  file containing N sample of redshift for each event, ')
parser.add_argument('--parameter1', help='name of parameter which we use for x-axis for KDE', default='m1')
parser.add_argument('--parameter2', help='name of parameter which we use for y-axis for KDE [can be m2, Xieff, DL]', default='m2')
### For KDE in log parameter we need to add --logkde 
parser.add_argument('--logkde', action='store_true',help='if True make KDE in log params but results will be in onlog')
# limits on KDE evulation: 
parser.add_argument('--m1-min', help='minimum value for primary mass m1', type=float)
parser.add_argument('--m1-max', help='maximum value for primary mass m1', type=float)
parser.add_argument('--Npoints', default=200, type=int, help='Total points on which to evaluate KDE')
#m2-min must be <= m1-min
parser.add_argument('--param2-min', default=2.95, type=float, help='minimum value of m2 ,chieff =-1, DL= 1Mpc, used, must be below m1-min')
parser.add_argument('--param2-max', default=100.0, type=float, help='max value of m2 used, could be m1-max for chieff +1, for DL 10000Mpc')

parser.add_argument('--fpopchoice', default='kde', help='choice of fpop to be rate or kde', type=str)
parser.add_argument('--type-data', choices=['gw_pe_samples', 'mock_data'], help='mock data for some power law with gaussian peak or gwtc  pe samples data. h5 files for two containing data for median and sigma for m1')
#EMalgorithm reweighting 
parser.add_argument('--reweightmethod', default='bufferkdevals', help='Only for gaussian sample shift method: we can reweight samples via buffered kdevals(bufferkdevals) or buffered kdeobjects (bufferkdeobject)', type=str)
parser.add_argument('--reweight-sample-option', default='reweight', help='choose either "noreweight" or "reweight" if reweight use fpop prob to get reweight sample (one sample for no bootstrap or no or multiple samples for poisson)', type=str)
parser.add_argument('--bootstrap-option', default='poisson', help='choose either "poisson" or "nopoisson" if None it will reweight based on fpop prob with single reweight sample for eaxh event', type=str)

parser.add_argument('--buffer-start', default=500, type=int, help='start of buffer in reweighting.')
parser.add_argument('--buffer-interval', default=100, type=int, help='interval of buffer that choose how many previous iteration resulkts we use in next iteration for reweighting.')
parser.add_argument('--NIterations', default=1000, type=int, help='Total Iterations in reweighting')
parser.add_argument('--Maxpdet', default=0.1, type=float, help='capping for small pdet to regularization')

parser.add_argument('--pathplot', default='./', help='public_html path for plots', type=str)
parser.add_argument('--pathfile', default='AnalysisCodes/', help='public_html path for plots', type=str)
parser.add_argument('--pathtag', default='re-weight-bootstrap_', help='public_html path for plots', type=str)
parser.add_argument('--output-filename', default='MassRedshift_with_reweight_output_data_', help='write a proper name of output hdf files based on analysis', type=str)
opts = parser.parse_args()



#############for ln paramereter we need these
def prior_factor_function(samples):
    """ 
    LVC use uniform-priors for masses 
    in linear scale. so
    reweighting need a constant factor

   note that in the reweighting function 
   if we use input masses/dL in log
   form so when we need to factor
   we need non-log mass/dL  so we take exp
    if we use non-cosmo_files we need 
    dL^3 factor 
    """
    m1val, dLval = samples[:, 0], samples[:, 1]
    if opts.logkde:
        factor = 1.0/(dLval)**2 
    else:
        factor = np.ones_like(m1val)
    return factor


def get_random_sample(original_samples, bootstrap='poisson'):
    """without reweighting"""
    rng = np.random.default_rng()
    if bootstrap =='poisson':
        reweighted_sample = rng.choice(original_samples, np.random.poisson(1))
    else:
        reweighted_sample = rng.choice(original_samples)
    return reweighted_sample


def apply_max_cap_function(pdet_list, max_pdet_cap=opts.Maxpdet):
  """Applies the max (0.1, pdet)  or min(10, 1/pdet) function to each element in the given list.

  Args:
    pdet_list: A list of values.

  Returns:
    A new list containing the results of applying the function to each element.
  """

  result = []
  for pdet in pdet_list:
    #result.append(min(10, 1 / pdet))
    result.append(max(max_pdet_cap, pdet))
  return np.array(result)

def get_reweighted_sample(original_samples, vtvals, fpop_kde, bootstrap='poisson', prior_factor=prior_factor_function):
    """
    inputs 
    original_samples: list of mean of each event samples 
    fpop_kde: kde_object [GaussianKDE(opt alpha, opt bw and Cov=True)]
    kwargs:
    bootstrap: [poisson or nopoisson] from araparser option
    prior: for ln parameter in we need to handle non uniform prior
    return: reweighted_sample 
    one or array with poisson choice
    if reweight option is used
    using kde_object compute probabilty on original samples and 
    we compute rate using Vt as samples 
    and apply 
    use in np.random.choice  on kwarg 
    """
    fkde_samples = fpop_kde.evaluate_with_transf(original_samples) / apply_max_cap_function(vtvals)

    if opts.logkde:
        frate_atsample = fkde_samples * prior_factor(original_samples) 
    else:
        frate_atsample = fkde_samples
    fpop_at_samples = frate_atsample/frate_atsample.sum() # normalize
    rng = np.random.default_rng()
    if bootstrap =='poisson':
        reweighted_sample = rng.choice(original_samples, np.random.poisson(1), p=fpop_at_samples)
    else:
        reweighted_sample = rng.choice(original_samples, p=fpop_at_samples)

    return reweighted_sample


def median_bufferkdelist_reweighted_samples(sample, vtvals, interp, bootstrap_choice='poisson', prior_factor=prior_factor_function):
    """
    added a prior factor to handle non uniform prior factor
    for ln parameter kde or rate
    inputs
    and based on what choice of bootstrap is given
    """
    #Take the medians of kde and use it in interpolator
    #median_kde_values = np.percentile(kdelist, 50, axis=0)
    #print("shape of median for interpolator", median_kde_values.shape)
    #interp = RegularGridInterpolator((m1val, m2val), median_kde_values.T, bounds_error=False, fill_value=0.0)
    kde_interp_vals = interp(sample)/apply_max_cap_function(vtvals)
    if opts.logkde:
        kde_interp_vals  *= prior_factor(sample)
    norm_mediankdevals = kde_interp_vals/sum(kde_interp_vals)
    rng = np.random.default_rng()
    if bootstrap_choice =='poisson':
        reweighted_sample = rng.choice(sample, np.random.poisson(1), p=norm_mediankdevals)
    else:
        reweighted_sample = rng.choice(sample, p=norm_mediankdevals)
    return reweighted_sample


#######################################################################
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
    m1_det = m1*(1.0 +  z)
    p_draw = f['injections/sampling_pdf'][:]
    pastro_pycbc_bbh = f['injections/pastro_pycbc_bbh'][:]

# Calculate min and max for dLp
min_dLp, max_dLp = min(dLp), max(dLp)

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
fz = h5.File('Nnoncosmo_GWTC3_redshift_datafile.h5', 'r')
dz1 = fz['randdata']
f1 = h5.File(opts.datafilename1, 'r')
d1 = f1['randdata']
f2 = h5.File(opts.datafilename2, 'r')
d2 = f2['randdata']
print(d1.keys())
sampleslists1 = []
medianlist1 = []
eventlist = []
sampleslists2 = []
medianlist2 = []
pdetlists = []
for k in d1.keys():
    eventlist.append(k)
    if (k  == 'GW190719_215514_mixed-nocosmo' or k == 'GW190805_211137_mixed-nocosmo'):
        print(k)
        d_Lvalues = d2[k][...]
        m_values = d1[k][...]#*(1.0 + dz1[k][...])
        mdet_values = d1[k][...]*(1.0 + dz1[k][...])
        dL_indices = [i for i, dL in enumerate(d_Lvalues) if (dL < min_dLp  or dL > max_dLp)]
        m_values = [m for i, m in enumerate(m_values) if i not in  dL_indices]
        mdet_values = [m for i, m in enumerate(mdet_values) if i not in  dL_indices]
        d_Lvalues = [dL for i, dL in enumerate(d_Lvalues) if i not in dL_indices]
        pdet_values =  np.zeros(len(d_Lvalues))
        for i in range(len(d_Lvalues)):
            pdet_values[i] = u_pdet.pdet_of_m1_dL_powerlawm2(mdet_values[i], 3.0, d_Lvalues[i], beta=1.26, classcall=g)

        #still some bad indices
        pdetminIndex = np.where(np.array(pdet_values) < 0.0001)[0]
        m_values = np.delete(m_values, pdetminIndex).tolist()
        d_Lvalues = np.delete(d_Lvalues, pdetminIndex).tolist()
        pdet_values = np.delete(pdet_values, pdetminIndex).tolist()
    else:
        m_values = d1[k][...]
        mdet_values = d1[k][...]*(1.0 + dz1[k][...])
        d_Lvalues = d2[k][...]
        # if we want to compute pdet use line after below line 
        #pdet_values = fpdet[k][...]
        pdet_values =  np.zeros(len(d_Lvalues))
        for i in range(len(d_Lvalues)):
            pdet_values[i] = u_pdet.pdet_of_m1_dL_powerlawm2(mdet_values[i], 3.0, d_Lvalues[i], beta=1.26, classcall=g)
    pdetlists.append(pdet_values)
    sampleslists1.append(m_values)
    sampleslists2.append(d_Lvalues)
    medianlist1.append(np.percentile(m_values, 50)) 
    medianlist2.append(np.percentile(d_Lvalues, 50))

f1.close()
f2.close()
fz.close()

meanxi1 = np.array(medianlist1)
meanxi2 = np.array(medianlist2)
flat_samples1 = np.concatenate(sampleslists1).flatten()
flat_samples2 = np.concatenate(sampleslists2).flatten()
flat_pdetlist = np.concatenate(pdetlists).flatten()


# Create the scatter plot for pdet 
plt.figure(figsize=(8,6))
plt.scatter(flat_samples1, flat_samples2, c=flat_pdetlist, cmap='viridis', norm=LogNorm())
cbar = plt.colorbar(label=r'$p_\mathrm{det}$')
cbar.set_label(r'$p_\mathrm{det}$', fontsize=20)
#plt.xlim(min(flat_samples1), max(flat_samples1))
#plt.ylim(min(flat_samples2), max(flat_samples2))
#plt.xlabel(r'$m_{1, detector} [M_\odot]$', fontsize=20)
plt.xlabel(r'$m_{1, source} [M_\odot]$', fontsize=20)
plt.ylabel(r'$d_L [Mpc]$', fontsize=20)
plt.loglog()
plt.title(r'$p_\mathrm{det}$', fontsize=20)
plt.tight_layout()
plt.savefig(opts.pathplot+"correctpdet_PEsamplesscatter.png")
plt.close()


plt.figure(figsize=(8,6))
for samplem1, samplem2 in zip(sampleslists1, sampleslists2):
    plt.scatter(samplem1, samplem2, marker='+')
#plt.xlabel(r'$m_{1, detector} [M_\odot]$', fontsize=20)
plt.xlabel(r'$m_{1, source} [M_\odot]$', fontsize=20)
plt.ylabel(r'$d_L [Mpc]$', fontsize=20)
plt.semilogx()
plt.title("100 PE samples per event")
plt.tight_layout()
plt.savefig(opts.pathplot+'Scattered_color_event_data_m1_dL.png')
plt.close()


sampleslists = np.vstack((flat_samples1, flat_samples2)).T
sample = np.vstack((meanxi1, meanxi2)).T
print(sampleslists.shape)
#to make KDE (2DKDE)  # we use same limits on m1 and m2 
if opts.m1_min is not None and opts.m1_max is not None:
    xmin, xmax = opts.m1_min, opts.m1_max
else:
    xmin, xmax = min([a.min() for a in sampleslists]), max([a.max() for a in sampleslists])
#use PE sample based limit
xmin, xmax = np.min(flat_samples1) , np.max(flat_samples1)

if opts.param2_min is not None and opts.param2_max is not None:
    ymin, ymax = opts.param2_min, opts.param2_max
else:
    ymin, ymax = min([a.min() for a in sampleslists]), max([a.max() for a in sampleslists])

ymin, ymax = np.min(flat_samples2) , np.max(flat_samples2)
Npoints = opts.Npoints

#######################################################
##### If we want to use log param we need proper grid spacing in log scale
p1grid = np.logspace(np.log10(xmin), np.log10(xmax), Npoints) 
p2grid = np.linspace(ymin, ymax, Npoints) 
XX, YY = np.meshgrid(p1grid, p2grid)
xy_grid_pts = np.array(list(map(np.ravel, [XX, YY]))).T
sample = sample = np.vstack((meanxi1, meanxi2)).T

############### save data in HDF file:
frateh5 = h5.File(opts.pathfile+'Final_'+opts.output_filename+'dL2priorfactor_uniform_prior_mass_2Drate_m1'+opts.parameter2+'.hdf5', 'w')
dsetxx = frateh5.create_dataset('data_xx', data=XX)
dsetxx.attrs['xname']='xx'
dsetyy = frateh5.create_dataset('data_yy', data=YY)
dsetxx.attrs['yname']='yy'
################################################################################

def get_kde_obj_eval(sample, eval_pts, rescale_arr, alphachoice, input_transf=('log', 'none')):
    kde_object = ad.KDERescaleOptimization(sample, stdize=True, rescale=rescale_arr, alpha=alphachoice, dim_names=['lnm1', 'dL'], input_transf=input_transf)
    dictopt, score = kde_object.optimize_rescale_parameters(rescale_arr, alphachoice, bounds=((0.01,100),(0.01, 100),(0,1)), disp=True, tol=0.1)#, xatol=0.01, fatol=0.1)
    kde_vals = kde_object.evaluate_with_transf(eval_pts)
    optbwds = [1.0/dictopt[0], 1.0/dictopt[1]]
    optalpha = dictopt[-1]
    print("opt results = ", dictopt)
    return  kde_object, kde_vals, optbwds, optalpha

##First median samples KDE
init_rescale_arr = [1., 1.]
init_alpha_choice = [0.5]
current_kde, errorkdeval, errorbBW, erroraALP = get_kde_obj_eval(sample, xy_grid_pts, init_rescale_arr, init_alpha_choice)
bwx, bwy = errorbBW[0], errorbBW[1]
# reshape KDE to XX grid shape 
ZZ = errorkdeval.reshape(XX.shape)
if opts.logkde:
    if opts.parameter2=='dL':
        print("dL case we need to use prior factor but in reweighting steps not here?")
        nlZZ = ZZ/YY**2
    else:
        nlZZ = ZZ #*nl_1_over_XX*nl_1_over_YY # for m1-m2 case

u_plot.new2DKDE(XX, YY,  ZZ,  meanxi1, meanxi2, saveplot=True,  show_plot=True, pathplot=opts.pathplot, plot_label='KDE', title='median')

med_group = frateh5.create_group(f'median_iteration')
    # Save the data in the group
med_group.create_dataset('rwsamples', data=sample)
med_group.create_dataset('alpha', data=erroraALP)
med_group.create_dataset('bwx', data=bwx)
med_group.create_dataset('bwy', data=bwy)
med_group.create_dataset('kde', data=errorkdeval)

########Physicalparams ofr power law in m2
m2_min = 3.0 # 5 is correct limit for BHs
beta = 1.26 #spectrial index for q  
if opts.fpopchoice == 'rate':
    pdet2D = np.zeros((Npoints, Npoints))
    mgrid = p1grid
    dLgrid = p2grid
    #convert masses im detector frame to make sure we are correctly computing pdet on same masses as KDE grid masses
    mdetgrid = get_massed_indetector_frame(dLgrid, mgrid)
    for i, m1val in enumerate(mdetgrid):
        for j, dLval in enumerate(dLgrid):
            pdet2D[i, j] = u_pdet.pdet_of_m1_dL_powerlawm2(m1val, m2_min, dLval, beta=beta, classcall=g)

    ## Set all values in `pdet` less than 0.1/0.03   to 0.1or 0.03
    dsetpdet = frateh5.create_dataset('pdet2D', data=pdet2D.T)
    dsetpdet.attrs['pdet']='pdet2D_m2min5'
    #pdetcapping
    pdet2D = np.maximum(pdet2D, opts.Maxpdet)
    current_rateval = len(meanxi1)*ZZ/pdet2D.T
    u_plot.new2DKDE(XX, YY,  current_rateval, meanxi1, meanxi2 , saveplot=True,plot_label='Rate', title='median', show_plot=True, pathplot=opts.pathplot)

plt.figure(figsize=(10, 8))
if opts.Maxpdet =='0.03':
    plt.contourf(XX, YY, pdet2D.T, levels=[ 0.03, 0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0], cmap='viridis', norm=Normalize(vmax=1))
    plt.title(r'$p_\mathrm{det}, \,  q^{1.26}, \, \mathrm{with} \, max(0.03, p_\mathrm{det})$', fontsize=18)
else:
    plt.contourf(XX, YY, pdet2D.T, levels=[0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0], cmap='viridis', norm=Normalize(vmax=1))
    plt.title(r'$p_\mathrm{det}, \,  q^{1.26}, \, \mathrm{with} \, max(0.1, p_\mathrm{det})$', fontsize=18)
plt.colorbar(label=r'$p_\mathrm{det}$')
plt.contour(XX, YY, pdet2D.T, colors='white', linestyles='dashed', levels=[0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0])
#plt.xlabel(r'$m_{1, detector} [M_\odot]$', fontsize=20)
plt.xlabel(r'$m_{1, source} [M_\odot]$', fontsize=20)
plt.ylabel(r'$d_L [Mpc]$', fontsize=20)
plt.loglog()
plt.savefig(opts.pathplot+"testpdet2Dpowerlaw_m2.png")
plt.close()

### reweighting EM algorithm
Total_Iterations = int(opts.NIterations)
discard = int(opts.buffer_start)   # how many iterations to discard default =5
Nbuffer = int(opts.buffer_interval) #100 buffer [how many (x-numbers of ) previous iteration to use in reweighting with average of those past (x-numbers of ) iteration results

iterkde_list = []
iter2Drate_list = []
iterbwxlist = []
iterbwylist = []
iteralplist = []

for i in range(Total_Iterations + discard):
    print("i - ", i)
    if i >= discard + Nbuffer:
        #use previous 100 iteration KDEs
        #bufffer_kdes_median = np.percentile(iterkde_list[-Nbuffer:], 50, axis=0)
        #buffer_interp = RegularGridInterpolator((p1grid, p2grid), bufffer_kdes_median.T, bounds_error=False, fill_value=0.0)
        buffer_kdes_mean = np.mean(iterkde_list[-Nbuffer:], axis=0)
        buffer_interp = RegularGridInterpolator((p1grid, p2grid), buffer_kdes_mean.T, bounds_error=False, fill_value=0.0)
    rwsamples = []
    for samplem1, samplem2, pdet_k in zip(sampleslists1, sampleslists2, pdetlists):
        samples= np.vstack((samplem1, samplem2)).T
        #if opts.logkde:
        #    samples = np.vstack((np.log(samplem1), samplem2)).T
        if i < discard + Nbuffer :
            rwsample = get_reweighted_sample(samples, pdet_k, current_kde, bootstrap=opts.bootstrap_option)
        else:
            rwsample= median_bufferkdelist_reweighted_samples(samples, pdet_k, buffer_interp, bootstrap_choice=opts.bootstrap_option)
        rwsamples.append(rwsample)
    if opts.bootstrap_option =='poisson':
        rwsamples = np.concatenate(rwsamples)
    print("iter", i, "  totalsamples = ", len(rwsamples))
    current_kde, current_kdeval, shiftedbw, shiftedalp = get_kde_obj_eval(np.array(rwsamples), xy_grid_pts, init_rescale_arr, init_alpha_choice,  input_transf=('log', 'none'))
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
        current_rateval = len(rwsamples)*current_kdeval/pdet2D.T
        iter2Drate_list.append(current_rateval)

    #if i > discard and i%Nbuffer==0:
    if i > 1 and i%Nbuffer==0:
        iterstep = int(i)
        print(iterstep)
        u_plot.histogram_datalist(iterbwxlist[-Nbuffer:], dataname='bwx', pathplot=opts.pathplot, Iternumber=iterstep)
        u_plot.histogram_datalist(iterbwylist[-Nbuffer:], dataname='bwy', pathplot=opts.pathplot, Iternumber=iterstep)
        u_plot.histogram_datalist(iteralplist[-Nbuffer:], dataname='alpha', pathplot=opts.pathplot, Iternumber=iterstep)
        if opts.logkde:
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


rate_lnm1dLmed = np.percentile(iter2Drate_list[discard:], 50., axis=0)
rate_lnm1dL_5 = np.percentile(iter2Drate_list[discard:], 5., axis=0)
rate_lnm1dL_95 = np.percentile(iter2Drate_list[discard:], 95., axis=0)

for val in [300, 500, 900, 1200, 1500, 2000, 4000]:
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
    plt.plot(m1_values, rate_lnm1_slice50,  linestyle='-', color='r', lw=2)
    plt.plot(m1_values, rate_lnm1_slice5,  linestyle='--', color='r', lw=1.5)
    plt.plot(m1_values, rate_lnm1_slice95,  linestyle='--', color='r', lw=1.5)
    plt.xlabel(r'$m_{1,\, source}$')
    plt.ylabel(r'$\mathrm{d}\mathcal{R}/m_1\mathrm{d}d_L [\mathrm{Mpc}^{-1}\, (\mathrm{m}/{M}_\odot)^{-1}  \mathrm{yr}^{-1}]$',fontsize=18)
    plt.title(r'$d_L=${0}[Mpc]'.format(val))
    plt.semilogx()
    plt.semilogy()
    plt.ylim(ymin=1e-6)
    plt.grid(True)
    plt.savefig(opts.pathplot+'OneD_rate_m1_slicedL{0}.png'.format(fixed_dL_value))
    plt.close()
    print("done")
