"""
Plot the Green kernel |G| and its gradient |∇G| across the radial direction.
"""

import numpy as np
import matplotlib.pyplot as plt
from TailoredGreen.CylinderGreen import CylinderGreen
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# -------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------
L = 0.02
D = 0.02
RTIP = 0.1
RROOT = RTIP * 0.16

RREF = 0.75 * RTIP
PHIREF = 0.0
# THETAREF = np.pi / 2
THETAREF = 0.0
# THETAREF = np.pi 


m = np.array([1])
B = 2
RPM = 8000
c0 = 340

wavenumber = m * B * RPM / 60 * 2 * np.pi / c0


# -------------------------------------------------------------------------
# Observer
# -------------------------------------------------------------------------
observer = np.array([
    RREF,
    D / 2 * (1 + 1e-12) * np.cos(THETAREF),
    -L + D / 2 * (1 + 1e-12) * np.sin(THETAREF)
]).reshape(3, 1)


# -------------------------------------------------------------------------
# Source points
# -------------------------------------------------------------------------
r_source = np.linspace(
    -5 * D + RREF,
    5 * D + RREF,
    250
)

sources = np.vstack([
    r_source,
    np.ones_like(r_source) * PHIREF * RREF,
    np.zeros_like(r_source)
])


# -------------------------------------------------------------------------
# Green function object
# -------------------------------------------------------------------------
green = CylinderGreen(
    D / 2,
    axis=np.array([1, 0, 0]),
    origin=np.array([0, 0, -L]),
    radial=np.array([0, 1, 0]),
    dim=3,
    numerics={
        'nmax': 16,
        'Nq_prop': 32,
        'Nq_evan': 16 * 10,
        'eps_radius': 1e-24,
        'Nazim': 9,
        'Nax': 32,
        'RMAX': 20,
        'mode': 'uniform',
        'geom_factor': 1.025,
        'eps_eval': 1e-12
    }
)


# -------------------------------------------------------------------------
# Evaluate kernels
# -------------------------------------------------------------------------

# Green function
kernel = green.getGreenFunction(
    observer,
    sources,
    wavenumber
)

# Analytical gradient
gradient_kernel = green.getGradientGreenAnalytical(
    observer,
    sources,
    wavenumber
)


# -------------------------------------------------------------------------
# Extract magnitudes
# -------------------------------------------------------------------------
kernel_abs = np.abs(kernel[:, 0, :]).T.squeeze()

# gradient_kernel has shape:
#     (3, Nk, Nx, Ny)
#
# Take the vector norm first, then extract the relevant dimensions.
gradient_abs = np.linalg.norm(
    gradient_kernel,
    axis=0
)

gradient_abs = gradient_abs[:, 0, :].T.squeeze()


# -------------------------------------------------------------------------
# Normalize independently
# -------------------------------------------------------------------------
kernel_abs /= np.max(kernel_abs)
gradient_abs /= np.max(gradient_abs)


# -------------------------------------------------------------------------
# Radial coordinate
# -------------------------------------------------------------------------
x = r_source - RREF
x_plot = x * 2 / D


# -------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4, 3))

ax.plot(
    x_plot,
    kernel_abs,
    # label=r'Kernel $|G|$',
    color='r',
    label=r'$|K_1(r-r_0)|$',
    lw=2
)

ax.plot(
    x_plot,
    gradient_abs,
    # label=r'Gradient $|\nabla G|$',
    color='b',
    label=r'$|\boldsymbol{K}_2(r-r_0)|$',
    lw=2,
    ls='--'
)


# -------------------------------------------------------------------------
# Blade radial extent: RROOT -> RTIP
#
# x = (r - RREF) * 2/D
a = (RROOT - RREF) * 2 / D
x_root = -a/2
x_tip  = a/2

# Vertical lines indicat
#ing the two ends
ax.axvline(
    x_root,
    color='k',
    ls=':',
    lw=1
)

ax.axvline(
    x_tip,
    color='k',
    ls=':',
    lw=1
)


# Double-sided arrow
y_arrow = 0.03

ax.annotate(
    '',
    xy=(x_tip, y_arrow),
    xytext=(x_root, y_arrow),
    arrowprops=dict(
        arrowstyle='<->',
        color='k',
        lw=1.5
    )
)

ax.text(
    (x_root + x_tip) / 2,
    y_arrow,
    r'$r_{\mathrm{tip}}-r_{\mathrm{root}}$',
    ha='center',
    va='bottom'
)


# -------------------------------------------------------------------------
# Labels / formatting
# -------------------------------------------------------------------------
ax.set_xlabel(r'$(r-r_0)/R$')
# ax.set_ylabel(r'Normalized magnitude $K/K_{\mathrm{max}}$')
ax.set_ylabel(r'$|K/K_{\mathrm{max}}|$')


plt.minorticks_on()

ax.grid(which='major', axis='both', linestyle='-')
ax.grid(which='minor', linestyle='--', alpha=0.5)

ax.legend()
ax.grid(True, alpha=0.2)
ax.set_ylim(0, 1.05)
ax.set_xlim(-10, 10)

plt.tight_layout()
plt.show()

import os
folder_name = f'./Figures/'
fig.savefig(
    os.path.join(folder_name, f"kernels_{THETAREF*360/2/np.pi:.0f}.pdf"),
    dpi=300,
    bbox_inches="tight",
)