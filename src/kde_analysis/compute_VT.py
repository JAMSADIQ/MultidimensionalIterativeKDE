#script to compute VT 3D generic or slice at fix 
# to run use this 
#python compute_VT.py  --fix_m1 20  --output 2DVTm2Xieff_m1slice_at_20.csv
#python compute_VT.py  --fix_m2 35  --output 2DVTm1Xieff_m2slice_at_20.csv
#python compute_VT.py  --fix_Xieff 0.25  --output 2DVTm1m2_m1slice_at_0_25.csv


######In case we want to run for days 
# python compute_VT.py --m1points 150 --m2points 150 --Xipoints 100 --output Full3DVTm1m2Xieff.csv
import os
import csv
import matplotlib.pyplot as plt
import new_o123_class_found_inj_general as u_pdet
import h5py as h5
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Generate VT values based on m1, m2, and Xieff.")
parser.add_argument('--m1points', type=int, default=50, help='Number of points in m1 grid')
parser.add_argument('--m2points', type=int, default=50, help='Number of points in m2 grid')
parser.add_argument('--Xipoints', type=int, default=40, help='Number of points in Xieff grid')
parser.add_argument('--fix_m1', type=float, help='Fix m1 at this value and slice along m2 and Xieff')
parser.add_argument('--fix_m2', type=float, help='Fix m2 at this value and slice along m1 and Xieff')
parser.add_argument('--fix_Xieff', type=float, help='Fix Xieff at this value and slice along m1 and m2')
parser.add_argument('--output', type=str, required=True, help='Output file name')
args = parser.parse_args()

########### for VT computation #################################
run_fit = 'o3'
run_dataset = 'o3'
dmid_fun = 'Dmid_mchirp_fdmid_fspin' #'Dmid_mchirp_fdmid'
emax_fun = 'emax_exp'
alpha_vary = None
g = u_pdet.Found_injections(dmid_fun = 'Dmid_mchirp_fdmid_fspin', emax_fun='emax_exp', alpha_vary = None, ini_files = None, thr_far = 1, thr_snr = 10)
g.load_inj_set(run_dataset)
g.get_opt_params(run_fit)
g.set_shape_params()

#############Grids ###############
m1 = np.logspace(np.log10(5), np.log10(105), args.m1points)
m2 = np.logspace(np.log10(5), np.log10(105), args.m2points)
Xieff = np.linspace(-1, 1, args.Xipoints)


csv_file = args.output
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['m1', 'm2', 'Xieff', 'VT'])  # Write header row


# Initialize VT array
if args.fix_m1 is not None:
    # Fix m1 and for loop along m2 and Xieff
    fixed_m1 = args.fix_m1
    # Open the CSV file in append mode
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        for j, m2val in enumerate(m2):
            for k, Xival in enumerate(Xieff):
                VT = g.total_sensitive_volume(fixed_m1, m2val, chieff =Xival)
                writer.writerow([fixed_m1, m2val, Xival, VT])

elif args.fix_m2 is not None:
    # Fix m2 and loop along m1 and Xieff
    fixed_m2 = args.fix_m2
    # Open the CSV file in append mode
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        for i, m1val in enumerate(m1):
            for k, Xival in enumerate(Xieff):
                VT =   g.total_sensitive_volume(m1val, fixed_m2, chieff =Xival) 
                writer.writerow([m1val, fixed_m2, Xival, VT])

elif args.fix_Xieff is not None:
    # Fix Xieff and slice along m1 and m2
    fixed_Xieff = args.fix_Xieff
    # Open the CSV file in append mode
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        for i, m1val in enumerate(m1):
            for j, m2val in enumerate(m2):
                VT =  g.total_sensitive_volume(m1val, m2val, chieff =fixed_Xieff)
                writer.writerow([m1val, m2val, fixed_Xieff, VT])

else:
    # Open the CSV file in append mode
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
    # No fixed variable, compute full 3D VT
        for i, m1val in enumerate(m1):
            for j, m2val in enumerate(m2):
                for k, Xival in enumerate(Xieff):
                    VT = g.total_sensitive_volume(m1val, m2val, chieff =Xival)
                    writer.writerow([m1val, m2val, Xival, VT])

quit()
#read file and interplote get VT plot 
import pandas as pd
df = pd.read_csv(csv_file)
m1_vals = np.sort(df['m1'].unique())
m2_vals = np.sort(df['m2'].unique())
Xieff_vals = np.sort(df['Xieff'].unique())
VT_values = np.empty((len(m1_vals), len(m2_vals), len(Xieff_vals)))
# Fill the 3D array with VT values
for _, row in df.iterrows():
    i = np.where(m1_vals == row['m1'])[0][0]
    j = np.where(m2_vals == row['m2'])[0][0]
    k = np.where(Xieff_vals == row['Xieff'])[0][0]
    VT_values[i, j, k] = row['VT']/1e9  #Gpc^3


interp_VT = RegularGridInterpolator((m1_vals, m2_vals, Xieff_vals), VT)
m2_query = np.logspace(np.log10(5), np.log10(105), 150)  # More resolution
Xieff_query = np.linspace(-1, 1, 100)

# Meshgrid for interpolation
M2q, XIEFFq = np.meshgrid(m2_query, Xieff_query, indexing='ij')
query_points = np.array([np.full(M2q.size, args.fix_m1) ,M2q.ravel(), XIEFFq.ravel()]).T
VT_interpolated = interp_VT(query_points).reshape(M2q.shape)

fig, ax = plt.subplots(111, figsize=(12, 5), constrained_layout=True)
vmin, vmax = np.nanpercentile(VT_interpolated, [5, 95])  # Avoid extreme outliers
norm = colors.LogNorm(vmin=max(vmin, 1e-3), vmax=min(vmax, 1e3))  # Avoid 0 or negative values
c = ax.pcolormesh(M2q, XIEFFq, VT_interpolated, cmap='viridis', norm=norm, shading='auto')
ax.contour(M2q, XIEFFq, VT_interpolated, levels=10, colors='white', linewidths=0.5, norm=norm)

# Labels & Title
ax.set_title(f"VT Slice at m2 = {m2_fixed}")
ax.set_xlabel("m2")
ax.set_ylabel("Xieff")
# Colorbar
fig.colorbar(c, ax=ax, label="VT (Log Scale)")
plt.show()
