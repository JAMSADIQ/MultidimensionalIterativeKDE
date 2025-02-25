#to run use  #10, 20, 25, 50 and 65 are available
#python  python Ratem1plots.py 65  
import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from scipy.interpolate import RegularGridInterpolator
from matplotlib import rcParams
from matplotlib.colors import LogNorm, Normalize
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

##################Only medians of m1-m2 and dL needed############
f1 = h5.File('/home/jxs1805/Research/CITm1dL/PEfiles/Final_noncosmo_GWTC3_m1srcdatafile.h5', 'r')#m1
d1 = f1['randdata']
f2 = h5.File('/home/jxs1805/Research/CITm1dL/PEfiles/Final_noncosmo_GWTC3_m2srcdatafile.h5', 'r')#m2
d2 = f2['randdata']
f3 = h5.File('/home/jxs1805/Research/CITm1dL/PEfiles/Final_noncosmo_GWTC3_xieff_datafile.h5', 'r')#dL
d3 = f3['randdata']
medianlist1 = f1['initialdata/original_mean'][...]
sampleslists2 = []
medianlist2 = f2['initialdata/original_mean'][...]
medianlist3 = f3['initialdata/original_mean'][...]

f1.close()
f2.close()
f3.close()

meanxi1 = np.array(medianlist1)
meanxi2 = np.array(medianlist2)
meanxi3 = np.array(medianlist3)

param = 'm1' # 'm1', 'm2'
paramval = int(sys.argv[1]) #10, 20, 35, 50, 65
df = pd.read_csv("grid2D50by40fixedm1_"+str(paramval)+"_VT_values_progress.csv")
savehf = h5.File("TwoDsliceCasem1fixed_slice_m1_"+str(paramval)+"KDEm1m2_Xieff_outputs_all1100Iterations.hdf5",  "r")

m1_vals = np.sort(df['m1'].unique())
m2_vals = np.sort(df['m2'].unique())
Xieff_vals = np.sort(df['Xieff'].unique())

VT_values = np.empty((len(m1_vals), len(m2_vals), len(Xieff_vals)))
# Fill the 3D array with VT values
for _, row in df.iterrows():
    i = np.where(m1_vals == row['m1'])[0][0]
    j = np.where(m2_vals == row['m2'])[0][0]
    k = np.where(Xieff_vals == row['Xieff'])[0][0]
    VT_values[i, j, k] = row['VT']/1e9

interp_VT = RegularGridInterpolator((m1_vals, m2_vals, Xieff_vals), VT_values, bounds_error=False, fill_value=np.nan)


XX = savehf["M1mesh"][...]
YY = savehf["M2mesh"][...]
ZZ = savehf["Xieffmesh"][...]
m1_query = paramval  #np.logspace(np.log10(5), np.log10(105), 150)  # More resolution
m2_query = np.logspace(np.log10(5), np.log10(105), 150)  # More resolution
Xieff_query = np.linspace(-1, 1, 100)


kde_list = []
for i in range(1100):#fix this as this can change
        KDE_slice = savehf["kde_iter{0}".format(i)][...]
        kde_list.append(KDE_slice)

M2 = YY[ 0, :, :]
XIEFF = ZZ[0, :, :]
query_points = np.array([np.full(M2.size, paramval), M2.ravel(), XIEFF.ravel()]).T
VT_interpolated = interp_VT(query_points).reshape(M2.shape)

u_plot.get_m2Xieff_at_m1_slice_plot(meanxi2, meanxi3, paramval, paramval, M2, XIEFF, kde_list[100:], VT_interpolated,  iterN=1001, pathplot='./', plot_name='KDE')
u_plot.get_m2Xieff_at_m1_slice_plot(meanxi2, meanxi3, paramval, paramval, M2, XIEFF, kde_list[100:], VT_interpolated,  iterN=1001, pathplot='./', plot_name='Rate')


rateXieff = []
import scipy
for kde in kde_list:
    Rate2D = 69*kde[0, :, :]/VT_interpolated
    #masking the data for integration
    mask = M2 <= paramval
    masked_Rate2D = np.where(mask, Rate2D, 0)
    #rateXieff.append(scipy.integrate.simpson(y=Rate2D, x=M2, axis=0))
    rateXieff.append(scipy.integrate.simpson(y=masked_Rate2D, x=M2, axis=0))
print(rateXieff)
median = np.percentile(rateXieff, 50., axis=0)

m05 = np.percentile(rateXieff, 5., axis=0)
m95 = np.percentile(rateXieff, 95., axis=0)
plt.figure(figsize=(8, 6))
plt.plot(Xieff_query, median, 'k', lw=2)
plt.plot(Xieff_query,  m05, 'r--')
plt.plot(Xieff_query, m95, 'r--')
plt.xlabel(r'$\chi_\mathrm{eff}$')
plt.ylabel(r'$\mathcal{{R}}(\chi_\mathrm{{eff}}, m1={0})$'.format(paramval))
plt.tight_layout()
plt.show()
