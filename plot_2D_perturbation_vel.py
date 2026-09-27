import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# =============================================================================
# Load data
# =============================================================================

data = loadmat("perturbation_data.mat")

y = data["y"].squeeze()
a = float(data["a"].squeeze())
U = float(data["U"].squeeze())
alpha = float(data["alpha"].squeeze())
ka_list = data["ka_list"].squeeze()

W_pot = data["W_pot"].squeeze()
W = data["W"]


# =============================================================================
# Plot settings
# =============================================================================

marker_list = ["s", "s", "^", "d", "v"]

legend_list = [
    # r"$1\times\mathrm{BPF}$",
    r'Scattering Operator, $k\rightarrow 0$',
    r"$3\times\mathrm{BPF}$",
    r"$5\times\mathrm{BPF}$",
    r"$7\times\mathrm{BPF}$",
    r"$9\times\mathrm{BPF}$",
]


# =============================================================================
# Real component
# =============================================================================

fig_real, ax = plt.subplots(figsize=(4, 3))


# Potential solution -- no marker
ax.plot(
    y / a,
    np.real(W_pot / U),
    "k-",
    linewidth=2,
        # marker='x'
)

# Helmholtz solutions
for i in range(1):

    ax.plot(
        y / a,
        np.real(W[i, :] / U),
        # linewidth=1.4,
        marker=marker_list[i],
        # markersize=5,
        markevery=10,
        alpha=0.5,
        color='b',
        label=legend_list[i],
    )

ax.set_xlabel(r"$y/R$")
ax.set_ylabel(r"$u/|\mathbf{U}_\infty|$")
ax.minorticks_on()
ax.grid(which='major', axis='both', linestyle='-')
ax.grid(which='minor', linestyle='--', alpha=0.5)



fig_real.tight_layout()
fig_real.savefig(
    "perturbation_real.pdf",
)


# =============================================================================
# Imaginary component
# =============================================================================

fig_imag, ax = plt.subplots(figsize=(4, 3))



# Potential solution -- no marker
ax.plot(
    y / a,
    -np.imag(W_pot / U),
    "k-",
    linewidth=2,
    # marker='x'
)

# Helmholtz solutions
for i in range(1):

    ax.plot(
        y / a,
        -np.imag(W[i, :] / U),
        # linewidth=1.4,
        marker=marker_list[i],
        # markersize=5,
        alpha=0.5,
        color='b',

        markevery=20,
    )

ax.set_xlabel(r"$y/R$")
ax.set_ylabel(r"$v/|\mathbf{u}_0|$")
ax.minorticks_on()
ax.grid(which='major', axis='both', linestyle='-')
ax.grid(which='minor', linestyle='--', alpha=0.5)
ax.legend(
    ["Potential Flow"] + legend_list,
    loc="lower left",
    # frameon=False,
)
ax.set_ylim([-0.35, 0.2])
fig_imag.tight_layout()
fig_imag.savefig(
    "perturbation_imag.pdf",
)


plt.show()