import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from scipy.integrate import quad
from matplotlib.colors import LogNorm, Normalize
from matplotlib import rcParams
from astropy.cosmology import FlatLambdaCDM, z_at_value, Planck15
import astropy.units as u

rcParams.update({
    "text.usetex": True,
    "font.serif": "Computer Modern",
    "font.family": "Serif",
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.fontsize": 20,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "axes.grid": True,
    "grid.color": 'grey',
    "grid.linewidth": 1.0,
    "grid.alpha": 0.6
})

#we need cosmology and comoving volume_factor
def hubble_parameter(z, H0, Omega_m):
    """
    Calculate the Hubble parameter H(z) for a flat Lambda-CDM cosmology.
    """
    Omega_Lambda = 1.0 - Omega_m
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)

def comoving_distance(z, H0, Omega_m):
    """
    Compute the comoving distance D_c(z) in Mpc.
    """
    integrand = lambda z_prime: c / hubble_parameter(z_prime, H0, Omega_m)
    D_c, _ = quad(integrand, 0, z)
    return D_c

def comoving_distance_derivative(z, H0, Omega_m):
    """
    Compute the derivative of comoving distance dD_c/dz in Mpc.
    """
    return c / hubble_parameter(z, H0, Omega_m)

#cosmology parameter we are using are
H0 = 67.9 #km/sMpc
c = 3e5 #km/s
omega_m = 0.3065 #matter density
cosmo =  FlatLambdaCDM(H0, omega_m)

def get_ddL_bydz_factor(dL_Mpc):
    """
    only works for float value not for arrays
    """
    z_at_dL = z_at_value(cosmo.luminosity_distance, dL_Mpc*u.Mpc).value
    D_c = comoving_distance(z_at_dL, H0, omega_m)
    dD_c_dz = comoving_distance_derivative(z_at_dL, H0, omega_m)
    ddL_dz = D_c + (1 + z_at_dL) * dD_c_dz
    return  z_at_dL, ddL_dz

def get_dVdz_factor(dL_Mpc):
    """
    only works for float value not for arrays
    """
    z_at_dL, ddL_dz = get_ddL_bydz_factor(dL_Mpc)
    dV_dMpc_cube = 4.0 * np.pi * cosmo.differential_comoving_volume(z_at_dL)/ ddL_dz
    dV_dzGpc3 = dV_dMpc_cube.to(u.Gpc**3 / u.sr).value
    return dV_dzGpc3


def get_massed_indetector_frame(dLMpc, mass):
    zcosmo = z_at_value(cosmo.luminosity_distance, dLMpc*u.Mpc).value
    mdet = mass*(1.0 + zcosmo)
    return mdet

def plot_volume_factor(dL_grid):
    volume_factor1D = np.array([get_dVdz_factor(dL) for dL in dL_grid])
    plt.figure(figsize=(8, 6))
    plt.plot(dL_grid, volume_factor1D)
    plt.xlabel(r"$d_L\,[\mathrm{Mpc}]$")
    plt.ylabel(r"$\frac{dV_c}{dz}$")
    plt.title("Volume Factor")
    plt.savefig("onedvolumefactor.png")
    plt.show()
    return volume_factor1D

def plot_comoving_volume(XX, YY, volume_factor2D):
    plt.figure(figsize=(10, 8))
    mesh = plt.pcolormesh(XX, YY, volume_factor2D, norm=LogNorm())
    plt.colorbar(mesh, label=r"$\frac{dV_c}{dz}$")
    plt.contour(XX, YY, volume_factor2D, norm=LogNorm(), colors='k', linewidths=1.5)
    plt.xlabel(r"$m_1\,[M_\odot]$")
    plt.ylabel(r"$d_L\,[\mathrm{Mpc}]$")
    plt.semilogx()
    plt.title("Comoving Volume Factor")
    plt.savefig("comovingvolumefactor.png")
    plt.show()

def plot_detection_probability(XX, YY, pdet2Dfilter, levels):
    plt.figure(figsize=(10, 7))
    plt.contourf(XX, YY, pdet2Dfilter, levels=levels, cmap='viridis', norm=Normalize(vmax=1))
    plt.colorbar(label=r"$p_\mathrm{det}$")
    plt.contour(XX, YY, pdet2Dfilter, colors='white', linestyles='dashed', levels=levels)
    plt.xlabel(r"$m_{1,\mathrm{source}}\,[M_\odot]$")
    plt.ylabel(r"$d_L\,[\mathrm{Mpc}]$")
    plt.loglog()
    plt.tight_layout()
    plt.title(r"Detection Probability ($p_\mathrm{det}$)")
    plt.title(r'$p_\mathrm{det}, \,  q^{1.26}, \, \mathrm{with} \, max(0.1, p_\mathrm{det})$', fontsize=18)
    plt.savefig("pdetcontourplot.png")
    plt.show()

def special_plot_rate(meanxi1, meanxi2, XX, YY, pdet2Dnofilter, CI50):
    fig, axl = plt.subplots(1, 1, figsize=(8, 6))
    levels_pdet = [0.01, 0.03, 0.1]

    # Plot PDET contours
    pdet_contour = axl.contour(
        XX, YY, pdet2Dnofilter,
        colors=['orange'] * len(levels_pdet),
        levels=levels_pdet,
        linewidths=2,
        linestyles=['--'] * len(levels_pdet)
    )

    # Add labels to contours
    contour_label_positions = []
    for collection in pdet_contour.collections:
        paths = collection.get_paths()
        for path in paths:
            vertices = path.vertices
            midpoint_index = len(vertices) // 2
            contour_label_positions.append(vertices[midpoint_index])

    axl.clabel(
        pdet_contour,
        inline=True,
        inline_spacing=8,
        use_clabeltext=True,
        fontsize=16,
        fmt="%.2f",
        manual=contour_label_positions
    )

    # Determine contour levels for CI50
    max_density = np.max(CI50)
    max_exp = np.floor(np.log10(max_density))
    contourlevels = 10 ** (max_exp - np.arange(5))[::-1]

    # CI50 colormap
    p = axl.pcolormesh(
        XX, YY, CI50,
        cmap=plt.cm.Purples,
        norm=LogNorm(vmin=contourlevels[0], vmax=contourlevels[-1]),
        shading='auto'
    )
    cbar = plt.colorbar(p, ax=axl)
    cbar.set_label(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}V_c [\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-1}]$', fontsize=20)

    # CI50 contour lines
    axl.contour(
        XX, YY, CI50,
        colors='black',
        levels=contourlevels,
        linestyles='dashed',
        linewidths=1.5,
        norm=LogNorm()
    )

    # Add scatter points
    axl.scatter(meanxi1, meanxi2, marker="+", color="r", s=20)

    # Set axis labels and limits
    axl.set_ylabel(r'$d_L\,[\mathrm{Mpc}]$', fontsize=20)
    axl.set_xlabel(r'$m_\mathrm{1, source} \,[M_\odot]$', fontsize=20)
    axl.set_ylim(ymin=200, ymax=7000)
    axl.semilogx()

    # Save the plot
    fig.tight_layout()
    plt.savefig("Special_pdetcontourlines_on_combined_average_Rate1000Iteration.png")
    plt.close()




path_to_files = '/home/jsadiq/Research/J_Ana/cbc_pdet/cbc_pdet/'
fz = h5.File(path_to_files+'/Final_noncosmo_GWTC3_redshift_datafile.h5', 'r')
dz1 = fz['randdata']
f1 = h5.File(path_to_files+'Final_noncosmo_GWTC3_m1srcdatafile.h5', 'r')
d1 = f1['randdata']
f2 = h5.File(path_to_files+'Final_noncosmo_GWTC3_dL_datafile.h5', 'r')
d2 = f2['randdata']
medianlist1 = []
medianlist2 = []
for k in d1.keys():
    d_Lvalues = d2[k][...]
    m_values = d1[k][...]
    mdet_values = d1[k][...]*(1.0 + dz1[k][...])
    medianlist1.append(np.percentile(m_values, 50))
    medianlist2.append(np.percentile(d_Lvalues, 50))

f1.close()
f2.close()
fz.close()
meanxi1 = np.array(medianlist1)
meanxi2 = np.array(medianlist2)

#assuming we have this data 
frateh5 = h5.File('m1_dL_powerlaw_m2_analysis_KDEsdata_hyperparas_reweight_samples_pdet_cap_0_1.hdf5', 'r')

XX = frateh5['data_xx'][:]
YY = frateh5['data_yy'][:]
m1_grid = XX[:, 0]
dL_grid = YY[0, :]
print("m1grid: min, max, N ", min(m1_grid), max(m1_grid), len(m1_grid))
print("dLgrid: min, max, N ", min(dL_grid), max(dL_grid), len(dL_grid))

#we need volume_factor to get correct units for Rates
volume_factor1D = np.zeros(len(dL_grid))
for i, dLval in enumerate(dL_grid):
    volume_factor1D[i] = get_dVdz_factor(dLval)
#2Dvolume same as KDE shape
xx, volume_factor2D = np.meshgrid(m1_grid, volume_factor1D, indexing='ij')
plot_comoving_volume(XX, YY, volume_factor2D)
################################################################################
m2_min = 5.0 #minimum m2 in integration
beta = 1.26 #spectrial index for q  

pdet2Dnofilter = frateh5['pdet2D'][:]
pdet2D= np.maximum(pdet2Dnofilter, 0.1)
pdet2Dfilter = frateh5['pdet2Dwithcap'][:]
#levels = [0.001,0.01,0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0] for pdet2Dnofilter
levels = [0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0]
plot_detection_probability(XX_mesh, YY, pdet2Dfilter, levels=levels)


# Now we want average KDE/Rate(incomoving units)
Total_Iterations = 1000
discard = 100
Nbuffer = 100

iterkde_list = []
iter2Drate_list = []
iterbwxlist = []
iterbwylist = []
iteralplist = []
print(frateh5['iteration_0'])

for i in range(Total_Iterations + discard):
    group = frateh5[f'iteration_{i}']
    # Save the data in the group
    rwsamples = group['rwsamples'][...]
    shiftedalp = group['alpha'][...]
    bwx = group['bwx'][...]
    bwy = group['bwy'][...]
    current_kdeval = group['kde'][:]
    current_kdeval = current_kdeval.reshape(XX.shape)
    iterkde_list.append(current_kdeval)
    iterbwxlist.append(bwx)
    iterbwylist.append(bwy)
    iteralplist.append(shiftedalp)
    current_rateval = len(rwsamples)*current_kdeval/pdet2D
    iter2Drate_list.append(current_rateval)
iterstep = 1001
frateh5.close()

ratelists = iter2Drate_list[100:]
#apply comoving volume Jacobian factor
CI50 = np.percentile(ratelists, 50, axis=0)/volume_factor2D
special_plot_rate(meanxi1, meanxi2, XX, YY, pdet2Dnofilter, CI50)


#median and offset plots
iterstep = 1001
#median and offset plots
rate_lnm1dLmed = np.percentile(iter2Drate_list[discard:], 50., axis=0)
rate_lnm1dL_5 = np.percentile(iter2Drate_list[discard:], 5., axis=0)
rate_lnm1dL_95 = np.percentile(iter2Drate_list[discard:], 95., axis=0)

colormap = plt.cm.magma#rainbow
dLarray = np.array([300, 600 ,900, 1200, 1500, 1800, 2100, 2400, 2700, 3000])#, 3000, 3500, 4000, 4500, 5000])

zarray = z_at_value(cosmo.luminosity_distance, dLarray*u.Mpc).value
norm = Normalize(vmin=zarray.min(), vmax=zarray.max())
y_offset = 0 
for plottitle in ['offset', 'median']:
    fig, ax = plt.subplots(figsize=(10, 8))
    for ik, val in enumerate(dLarray):
        color = colormap(norm(zarray[ik]))
        closest_index = np.argmin(np.abs(YY - val))
        fixed_dL_value = YY.flat[closest_index]
        volume_factor = get_dVdz_factor(fixed_dL_value)
        indices = np.isclose(YY, fixed_dL_value)
        # Extract the slice of rate_lnm1dL for the specified dL
        rate_lnm1_slice50 = rate_lnm1dLmed[indices]
        rate_lnm1_slice5 = rate_lnm1dL_5[indices]
        rate_lnm1_slice95 = rate_lnm1dL_95[indices]
        #correcting units
        rate50 =  rate_lnm1_slice50/volume_factor
        rate05 =  rate_lnm1_slice5/volume_factor
        rate95 =  rate_lnm1_slice95/volume_factor
        # Extract the corresponding values of lnm1 from XX
        m1_values = XX[indices]
    
        ###%%for each slice separate plot
        #plt.figure(figsize=(8, 6))
        #plt.plot(m1_values, rate50,  linestyle='-', color='k', lw=2)
        #plt.plot(m1_values, rate05,  linestyle='--', color='r', lw=1.5)
        #plt.plot(m1_values, rate95,  linestyle='--', color='r', lw=1.5)
        #plt.xlabel(r'$m_{1,\, source}$')
        #plt.ylabel(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}V_c [\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-1}]$ ',fontsize=18)
        #plt.title(r'$d_L=${0}[Mpc]'.format(val))
        #plt.semilogy()
        #plt.ylim(ymin=1e-6)
        #plt.grid(True)
        #plt.tight_layout()
        #plt.savefig('OneD_rate_m1_slicedL{0:.1f}.png'.format(val))
        #plt.semilogx()
        #plt.tight_layout
        #plt.savefig('OneD_rate_m1_slicedL{0:.1f}LogXaxis_ComovingVolume_Units.png'.format(val))
        #plt.close()
        #print("done")
    
        # Masking for huge errors
        mask = (rate05 >= 1e-3 * rate50) & (rate95 <= 5e3 * rate50)
        # Use masked arrays to filter the invalid regions
        m1_masked = np.ma.masked_where(~mask, m1_values)  # Masked x values
        p50_masked = np.ma.masked_where(~mask, rate50)
        p5_masked = np.ma.masked_where(~mask, rate05)
        p95_masked = np.ma.masked_where(~mask, rate95)


        if  plottitle == 'median':
            ax.plot(m1_values, rate50, color=color,  lw=1.5, label=f'z={zarray[ik]:.1f}')
            ax.set_ylabel(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}V_c [\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-1}]$ ',fontsize=20)
            ax.set_ylim(ymin=1e-4)

        else:
            ax.plot(m1_masked, np.log10(p50_masked)+y_offset, color=color,  lw=1.5, label=f'z={zarray[ik]:.1f}')
            ax.fill_between(m1_masked, np.log10(p5_masked) + y_offset, np.log10(p95_masked) + y_offset, color=color, alpha=0.3)
            ax.set_ylabel(r'$\mathrm{log}10(\mathrm{d}\mathcal{R}/\mathrm{d}m_1\mathrm{d}V_c) [\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}\mathrm{M}_\odot^{-1}]$ + offset',fontsize=18)
            ax.set_ylim(ymin=-5, ymax=17)
            y_offset += 2
            
    from matplotlib import cm
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label='$z$')
    ax.set_xlabel(r'$m_\mathrm{1, source} \,[M_\odot]$', fontsize=20)
    ax.set_xlim(5, 80)
    if  plottitle == 'median':
        plt.semilogy()
    plt.tight_layout()
    plt.savefig(plottitle+'_rate_m1_at_slice_dLplot_with_redshift_colors.png')
    plt.semilogx()
    plt.tight_layout()
    plt.savefig(plottitle+'_rate_m1_at_slice_dLplot_with_redshift_colors_log_Xaxis.png')
    plt.show()

