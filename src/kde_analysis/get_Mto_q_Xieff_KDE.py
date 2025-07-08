import numpy as np
import matplotlib.pyplot as plt
import popde.density_estimate as d
import popde.adaptive_kde as ad

#Mock data for mtot, q and xieff
# Set random seed for reproducibility (optional)
np.random.seed(42)
# Generate 100 samples with 3 values each
samples = []
for _ in range(100):
    # First value: peaks around 20, 60, and 150 (between 4 and 200)
    # We'll create a mixture of normal distributions centered at these points
    choice = np.random.choice([0, 1, 2], p=[0.4, 0.5, 0.1])  # weights for each peak
    if choice == 0:
        val1 = np.random.normal(20, 5)
    elif choice == 1:
        val1 = np.random.normal(60, 10)
    else:
        val1 = np.random.normal(150, 15)
    val1 = np.clip(val1, 4, 200).round(2)  # ensure within bounds and round

    # Second value: power law distribution between 1 and 10 (more values close to 1)
    val2 = (np.random.power(3) * 9 + 1) # 3 is the exponent, higher = more skewed to 1

    # Third value: uniform between -1 and 1
    val3 = np.random.uniform(-1, 1)

    samples.append([val1, val2, val3])

# Convert to numpy array (optional)
samples = np.array(samples)

### we need to automatize this?
Mmin, Mmax = 4, 200  #m1
qmin, qmax = 1, 10 #m2
Ximin, Ximax = -1, 1 #dL
Npoints = 150 #opts.Npoints

Mtot_grid = np.logspace(np.log10(Mmin), np.log10(Mmax), Npoints)
q_grid = np.logspace(qmin, qmax, Npoints)
Xieff_grid = np.linspace(Ximin, Ximax, 60) 

XX, YY, ZZ = np.meshgrid(Mtot_grid, q_grid, Xieff_grid, indexing='ij')
eval_samples = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
#for plot
MTOT, XIEFF = np.meshgrid(Mtot_grid, Xieff_grid, indexing='ij')

#train KDE use fixed bandwidtha and alpha
alpha=0.1
bwx= 0.2
bwy = 0.2
bwz = 0.2
train_kde = ad.AdaptiveBwKDE(samples,
                None,
                input_transf=('log', 'none', 'none'),
                stdize=True,
                rescale=[1/bwx, 1/bwy, 1/bwz],
                alpha=alpha
            )

# Evaluate the KDE on the evaluation samples
eval_kde3d = train_kde.evaluate_with_transf(eval_samples)
KDE_slice = eval_kde3d.reshape(XX.shape)
KDE2D = KDE_slice[:, 0, :]  # Shape will be (Npoints, 60)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
# 1. Colormesh plot
mesh = ax1.pcolormesh(MTOT, XIEFF, KDE2D, shading='auto')
fig.colorbar(mesh, ax=ax1, label='Density')
ax1.set_xscale('log')
ax1.set_xlabel('Mtot')
ax1.set_ylabel('Xieff')
ax1.set_title('Colormesh Plot')

# 2. Contour plot
contour = ax2.contourf(MTOT, XIEFF, KDE2D, levels=20, cmap='viridis')
fig.colorbar(contour, ax=ax2, label='Density')
ax2.set_xscale('log')
ax2.set_xlabel('Mtot')
ax2.set_ylabel('Xieff')
ax2.set_title('Contour Plot')

plt.tight_layout()
plt.show()

