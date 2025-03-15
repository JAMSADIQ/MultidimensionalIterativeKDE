import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LogNorm, Normalize
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import matplotlib.patches
from matplotlib.patches import Rectangle
import glob
import deepdish as dd

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
#rcParams["grid.linestyle"] = ':'
rcParams["grid.alpha"] = 0.6


with h5py.File('MultidimensionalIterativeKDE/data/public/m1m2xieff_files/KDEgrid_based_finer_VT_data.h5', 'r') as f:
    VT = f['VT'][:]
    m1 = f['m1'][:]
    m2 = f['m2'][:]
    Xieff = f['Xieff'][:]

m2_index_10 = np.argmin(np.abs(m2 - 10))

# Extract VT at the fixed m2 value
VT_at_m2_10 = VT[:, m2_index_10, :]
# Create a meshgrid for m1 and Xieff
M1, Xieff_grid = np.meshgrid(m1, Xieff, indexing='ij')
# Create the plot
plt.figure(figsize=(8, 6))
# Colormesh plot
mesh = plt.pcolormesh(M1, Xieff_grid, VT_at_m2_10/1e9, shading='auto', cmap='viridis', norm=LogNorm())
plt.colorbar(mesh, label=r'$\mathrm{VT}\, [\mathrm{Gpc}^3 - \mathrm{yr}]$')
# Contour plot with white bold lines
contour = plt.contour(M1, Xieff_grid, VT_at_m2_10/1e9, levels=10, colors='white', linewidths=2, norm=LogNorm())
#plt.clabel(contour, inline=True, fontsize=10, fmt='%1.1f')  # Add labels to contour lines

# Add labels and title
plt.xlabel(r'$m_1$')
plt.ylabel(r'$\chi_\mathrm{eff}$')
plt.title('VT at m2 = 10')
plt.show()

# Find the index for Xieff = 0.0
Xieff_index_0 = np.argmin(np.abs(Xieff - 0.0))

# Extract VT at the fixed Xieff value
VT_at_Xieff_0 = VT[:, :, Xieff_index_0]

# Create a meshgrid for m1 and m2
M1, M2 = np.meshgrid(m1, m2, indexing='ij')
# Create a mask for m2 > m1
mask = M2 > M1

# Apply the mask to VT data (set values to NaN where m2 > m1)
VT_masked = np.where(mask, np.nan, VT_at_Xieff_0)
# Create the plot
plt.figure(figsize=(8, 6))

# Colormesh plot
#mesh = plt.pcolormesh(M1, M2, VT_at_Xieff_0/1e9, shading='auto', cmap='viridis', norm=LogNorm())
mesh = plt.pcolormesh(M1, M2, VT_masked/1e9, shading='auto', cmap='viridis', norm=LogNorm())
plt.colorbar(mesh, label=r'$\mathrm{VT}\, [\mathrm{Gpc}^3 - \mathrm{yr}]$')

# Contour plot with white bold lines
#contour = plt.contour(M1, M2, VT_at_Xieff_0/1e9, levels=10, colors='white', linewidths=2, norm=LogNorm())
contour = plt.contour(M1, M2, VT_masked/1e9, levels=10, colors='white', linewidths=2, norm=LogNorm())
#plt.clabel(contour, inline=True, fontsize=10, fmt='%1.1f')  # Add labels to contour lines

# Add labels and title
plt.xlabel(r'$m_1$')
plt.ylabel(r'$m_2$')
plt.title('VT at Xieff = 0.0')
plt.loglog()
plt.show()
