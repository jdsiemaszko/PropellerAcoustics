import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
data_dir = "./Data/current/validation"
index_k = -1  # channel to plot

# ---------------------------------------------------------
# Find files
# ---------------------------------------------------------
pattern = os.path.join(data_dir, "green_on_surface_*_*_*.npz")
files = sorted(glob.glob(pattern))

if not files:
    raise FileNotFoundError(f"No files found matching {pattern}")

# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------
datasets = []

for f in files:
    fname = os.path.basename(f)

    m = re.match(
        r"green_on_surface_(\d+)_(\d+)_(\d+)\.npz",
        fname,
    )

    if m is None:
        continue

    mmax, Nq_prop, Nq_evan = map(int, m.groups())

    dat = np.load(f)

    datasets.append(
        {
            "mmax": mmax,
            "Nq_prop": Nq_prop,
            "Nq_evan": Nq_evan,
            "eps": dat["eps"],
            "L2": dat["L2"],
        }
    )

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))

# cmap = plt.cm.viridis
# colors = cmap(np.linspace(0, 1, len(datasets)))

colors = ['r', 'g', 'b', 'm', 'c', 'y', 'pink']
markers = ['^', 's', 'o', 'p', '*', '8', '.']
for color, marker, ds in zip(colors, markers, datasets):

    ax.plot(
        ds["eps"],
        ds["L2"][:, index_k],
        lw=2,
        color=color,
        # label=(
        #     f"m={ds['mmax']}, "
        #     f"Nq_prop={ds['Nq_prop']}, "
        #     f"Nq_evan={ds['Nq_evan']}"
        # ),
        marker=marker,
                label=(
            f"{ds['mmax']}/{ds['Nq_prop'] + ds['Nq_evan']}"
        ),
    )

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"$\Delta r / a$")
# ax.set_ylabel(fr"$L_2(k={index_k})$")
ax.set_ylabel(r"$\epsilon = \sqrt{1/N\sum_{n<N} (G_0(x_n|y) + G_s(x_n|y))} $")

ax.grid(True, which="both", alpha=0.3)

ax.legend(
    title="Discretization ($N_m/N_q$)",
    fontsize=8,
    title_fontsize=9,
)

plt.tight_layout()
plt.show()
fig.savefig('./Figures/error_convergence.pdf')