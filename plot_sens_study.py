from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

data_dir = Path("./Data/current/validation/sens_v2/")
B = 2
filenames = [
    "BASELINE.npz",
    "T10.npz",
    "FZ10.npz",
    "FPHI10.npz",
    "BLH10.npz",
    "LOADING_LE.npz",
    "LOADING_TE.npz",
    "LOADING_HALF.npz",
]

colors = ['k', 'r', 'g', 'b', 'm', 'c', 'y', 'pink']
markers = ["o", "s", "^", '*', 'p', '8', '.', 'x']
labels = [r"baseline", r"$t/c \uparrow 10\% $", r"$F_z \uparrow 10\%$", r"$F_\phi \uparrow 10\%$", r"$\hat{F} \uparrow 10\%$",
           r"point load at LE", r"point load at TE", r'point load at mid-chord']

fig, ax = plt.subplots(figsize=(6, 4))

for fname, color, marker, label in zip(
    filenames, colors, markers, labels
):
    
    data = np.load(data_dir / fname)

    m = data["m"]
    SPL = data["SPL"]

    if 'BASELINE' in fname:
        SPL_base = SPL
        continue

    ax.plot(
        m,
        SPL -  SPL_base,
        color=color,
        marker=marker,
        label=label,
        linewidth=1.5,
    )

ax.set_xlabel(r"$f/B/Omega$")
ax.set_xticks(m)
ax.set_ylabel("$\Delta SPL$ [dB]")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc='upper right')
fig.tight_layout()

plt.show()