import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import scipy
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


def TwoDplot(medianlist_m1, medianlist_xieff, M1, XIEFF, twoDlist,  iterN=1, pathplot='./', plot_name='KDE'):
    #get median 
    data_slice = np.percentile(twoDlist, 50, axis=0)
    if plot_name=='Rate':
        #data_slice *= 69 #69 is numbe of observed BBH signals         
        colorbar_label = r'$d \mathcal{R}/dm_2 d\chi_\mathrm{eff}[\mathrm{Gpc}^{-3} \mathrm{yr}^{-1} M_\odot^{-1}] $'
    else:
        colorbar_label = r'$p(m_2, \chi_\mathrm{eff})$'
    max_density = np.nanmax(data_slice)
    max_exp = np.floor(np.log10(max_density))
    contourlevels = 10 ** (max_exp - np.arange(4))[::-1]
    contourlevels[-1] = max_density
    vmin, vmax = contourlevels[0], contourlevels[-1]# np.nanmax(KDE_slice)  # Min and max values for KDE
    print("vmin, vmax is =", vmin, vmax)
    # Plot
    plt.figure(figsize=(8, 6))
    norm = LogNorm(vmin=vmin, vmax=vmax)  # Apply log normalization
    #pcm = plt.pcolormesh(M1, XIEFF, data_slice, cmap='viridis', norm=norm, shading='auto')
    pcm = plt.pcolormesh(M1, XIEFF, data_slice, cmap='Purples', norm=norm, shading='auto')
    contours = plt.contour(M1,  XIEFF, data_slice, levels=contourlevels, colors='black', linewidths=1.5)

    #cbar = plt.colorbar(pcm, label='')#colorbar_label)
    #cbar.set_ticks(contourlevels)
    plt.scatter(medianlist_m1, medianlist_xieff, color='r', marker='+', s=20)
    plt.ylabel(r"$\chi_\mathrm{eff}$")
    plt.xlabel(r"$m_\mathrm{2} \,[M_\odot]$")
    plt.semilogx()
    plt.tight_layout()
    #plt.savefig(pathplot+"Average"+plot_name+"m1Xieffatm2_{1}_Iter{0}.png".format(iterN, m2_target))
    #plt.close()
    plt.show()
    return 0


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
VTdata  =  "finer_gridVT_values_progress.csv" #40 by 40 by 30 values
df = df = pd.read_csv(VTdata)
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
interp_VT = RegularGridInterpolator((m1_vals, m2_vals, Xieff_vals), VT_values, bounds_error=False, fill_value=1.0)


#savehf = h5.File("fIRSToutput_fileSave3DKDEm1m2_XieffFirst100Iter.hdf5",  "r")
savehf = h5.File("sECONDoutput_fileSave3DKDEm1m2_Xieff_outputs_100to200Iterations.hdf5",  "r")
XX = savehf["M1mesh"][...]
YY = savehf["M2mesh"][...]
ZZ = savehf["Xieffmesh"][...]

mask = XX < YY
m1 = XX[:, 0, 0]
m2 = YY[0, :, 0]
Xieff = ZZ[0, 0, :]
print("m1", len(m1))
print("m2", m2)
print("chi",len( Xieff))

#interpolate to et VT o same grid as KDE
query_points = np.array([XX.ravel(), YY.ravel(), ZZ.ravel()]).T
VT_interpolated = interp_VT(query_points).reshape(YY.shape)

#We we will get 2D kde integrating over m2
kde2d_list = []
rate2d_list = []
for i in range(100, 200, 1):#fix this as this can change
        KDE_3D = savehf["kde_iter{0}".format(i)][...]
        Rate_3D = 69 * KDE_3D/VT_interpolated #69bbh events
        #integrate with mask on m2
        KDE_masked = np.where(mask, 0, KDE_3D)
        rate_masked = np.where(mask, 0, Rate_3D)
        print(rate_masked) #this has nan
        # Integrate maskedKDE with respect to m1
        KDE_integral_result = scipy.integrate.simpson(KDE_masked, x=m1, axis=0)
        #there is trouble due to nan
        Rate_integral_result = scipy.integrate.simpson(rate_masked, x=m1, axis=0)
        print(Rate_integral_result)
        kde2d_list.append(KDE_integral_result)
        rate2d_list.append(Rate_integral_result)

M2, XIeff = np.meshgrid(m2, Xieff, indexing='ij')
TwoDplot(medianlist2, medianlist3, M2, XIeff, kde2d_list,  iterN=100, pathplot='./', plot_name='KDE')
TwoDplot(medianlist2, medianlist3, M2, XIeff, rate2d_list,  iterN=100, pathplot='./', plot_name='Rate')
