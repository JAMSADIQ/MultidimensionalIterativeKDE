import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from scipy.integrate import quad
from matplotlib.colors import LogNorm, Normalize
from matplotlib import rcParams
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u

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

###################################################################
# Define cosmology
H0 = 67.9  # km/s/Mpc
omega_m = 0.3065
cosmo = FlatLambdaCDM(H0=H0, Om0=omega_m)
c = 299792458.0/1000.0  #km/s

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

#### For conversion of units dR/dm1dL to dR/dm1dVc = dR/dm1dL*(dL/dz)/(dVc/dz) 
#Based on Hogg 1996
def hubble_parameter(z, H0, Omega_m):
    """
    Calculate the Hubble parameter H(z) for a flat Lambda-CDM cosmology.
    https://ned.ipac.caltech.edu/level5/Hogg/Hogg4.html
    """
    Omega_Lambda = 1.0 - Omega_m
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)

def comoving_distance(z, H0, Omega_m):
    """
    Compute the comoving distance D_c(z) in Mpc.
    we can also use astropy
    astropy_val = cosmo.comoving_distance(z)
    """
    astropy_val = cosmo.comoving_distance(z)
    integrand = lambda z_prime: c / hubble_parameter(z_prime, H0, Omega_m)
    D_c, _ = quad(integrand, 0, z)
    return D_c


def comoving_distance_derivative(z, H0, Omega_m):
    """
    Compute the derivative of comoving distance dD_c/dz in Mpc.
    """
    return c / hubble_parameter(z, H0, Omega_m)


def precompute_cosmology_factors(z, dL_Mpc=None):
    """
    Precompute commonly used cosmology values to avoid redundant calculations.
    Returns a dictionary containing:
    - comoving_distance: D_c in Mpc
    - dDc_dz: d(D_c)/dz in Mpc
    - luminosity_distance: D_L in Mpc (if dL_Mpc provided)
    """
    factors = {}
    factors["comoving_distance"] = comoving_distance(z, H0, omega_m)
    factors["dDc_dz"] = comoving_distance_derivative(z, H0, omega_m)
    if dL_Mpc is not None:
        factors["luminosity_distance"] = dL_Mpc
    return factors

def get_dDL_dz_factor(z_at_dL, precomputed=None):
    """
    Compute d(D_L)/dz using precomputed cosmology factors if available.
    given redshift 
    from notes: dL = (1+z)*Dc
    """
    if precomputed is None:
        precomputed = precompute_cosmology_factors(z_at_dL)
    Dc = precomputed["comoving_distance"]
    dDc_dz = precomputed["dDc_dz"]
    return Dc + (1 + z_at_dL) * dDc_dz

def get_dVc_dz_factor(z_at_dL, precomputed=None):
    """
    Compute d(V_c)/dz in Gpc^3 units using precomputed cosmology factors.
    return dVc/dz in Gpc^3 units
    Vc = 4pi/3 Dc^3 (Dc = comoving disatnce)
    dVc/dz = 4pi Dc  dDc/dz
    """
    if precomputed is None:
        precomputed = precompute_cosmology_factors(z_at_dL)
    Dc = precomputed["comoving_distance"]
    dDc_dz = precomputed["dDc_dz"]
    dVcdz = 4 * np.pi * Dc**2 * dDc_dz
    return dVcdz / 1e9  # Convert to Gpc^3


def get_volume_factor(dLMpc):
    """
    Compute the volume factor for a given luminosity distance in Mpc.
    result is in Gpc^3-yr
    """
    z_at_dL = z_at_value(cosmo.luminosity_distance, dLMpc * u.Mpc).value
    precomputed = precompute_cosmology_factors(z_at_dL, dL_Mpc=dLMpc)
    ddL_dz = get_dDL_dz_factor(z_at_dL, precomputed)
    dVc_dz = get_dVc_dz_factor(z_at_dL, precomputed)
    factor_time_det_frame = 1.0 #+ z_at_dL
    return dVc_dz / ddL_dz / factor_time_det_frame

def get_dVcdz_from_comoving_volume(dLMpc):
    """
    only works for float value not for arrays
    """
    z_at_dL = z_at_value(cosmo.luminosity_distance, dLMpc*u.Mpc).value
    dV_dMpc_cube = 4.0 * np.pi * cosmo.differential_comoving_volume(z_at_dL)
    dVc_dzGpc3 = dV_dMpc_cube.to(u.Gpc**3 / u.sr).value
    return dVc_dzGpc3


#testing it
dLvals = np.linspace(100, 8000, 100)
Vf = np.zeros(len(dLvals))
for i in range(len(Vf)):
    Vf[i] = get_volume_factor(dLvals[i])

plt.plot(dLvals, Vf)
plt.xlabel(r"$d_L$[Mpc]")
plt.ylabel(r"$\mathrm{Volume factor Gpc}^3-\mathrm{yr}$")
plt.tight_layout()
plt.show()
m1vals = np.logspace(np.log10(5), np.log10(105), 150)
xx, yy = np.meshgrid(m1vals, dLvals, indexing='ij')
xx, Vf2D = np.meshgrid(m1vals, Vf, indexing='ij')
plt.contourf(xx, yy, Vf2D, cmap='viridis', norm=LogNorm())
plt.colorbar(label=r'$\mathrm{Volume factor Gpc3-yr}$')
plt.contour(xx, yy, Vf2D, colors='white', linestyles='dashed',  norm=LogNorm())
plt.xlabel(r'$m_{1, source} [M_\odot]$', fontsize=20)
plt.ylabel(r'$d_L [Mpc]$', fontsize=20)
plt.loglog()
plt.tight_layout()
plt.show()

