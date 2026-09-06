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
THETAREF = np.pi / 2
# THETAREF = 0.0


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
    -100 * D + RREF,
    100 * D + RREF,
    2500
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



# moments!
r0 = np.linspace(RROOT, RTIP, 100)

import numpy as np


def get_moments(k, r, r0):
    """
    Compute moments of k(r) about r0.

    The irrelevant portions of k are masked to zero while retaining
    the original r-grid, avoiding integration across disjoint domains.

    Parameters
    ----------
    k : array_like
        Function values on the radial grid r.
    r : array_like
        1-D radial grid.
    r0 : float or array_like
        Reference radius/radii.

    Returns
    -------
    m0, m1, m2 : ndarray or float
        Zeroth, first, and second moments over RROOT < r < RTIP.
    m0_total : float
        Zeroth moment over the full radial domain.
    m0_tail : ndarray or float
        Zeroth moment over the complement of RROOT < r < RTIP.
    """
    k = np.asarray(k)
    r = np.asarray(r)
    r0 = np.atleast_1d(r0)

    # Full-domain zeroth moment
    m0_total = np.trapezoid(k, r)

    # Masks on the original r-grid
    inner_mask = (r > RROOT) & (r < RTIP)
    outer_mask = ~inner_mask

    # Zero irrelevant portions of k.
    # Keep the original r-grid so that the integration domains
    # remain contiguous.
    k_inner = np.where(inner_mask, k, 0.0)
    k_outer = np.where(outer_mask, k, 0.0)

    # Distance from each r0
    dr = r[None, :] - r0[:, None]

    # Moments over RROOT < r < RTIP
    m0 = np.trapezoid(
        k_inner[None, :],
        r,
        axis=1,
    )

    m1 = np.trapezoid(
        k_inner[None, :] * dr,
        r,
        axis=1,
    )

    m2 = np.trapezoid(
        k_inner[None, :] * dr**2,
        r,
        axis=1,
    )

    # Zeroth moment over the tail
    m0_tail = np.trapezoid(
        k_outer[None, :],
        r,
        axis=1,
    )

    # Return scalars if r0 was scalar
    if np.ndim(r0) == 1 and r0.size == 1:
        return m0[0], m1[0], m2[0], m0_total, m0_tail[0]

    return m0, m1, m2, m0_total, m0_tail


m01, m11, m21, m0_total1, m0_tail1 = get_moments(kernel[0, 0, :], r_source, r0)
m02, m12, m22, m0_total2, m0_tail2 = get_moments(gradient_kernel[2, 0, 0, :], r_source, r0)

# -------------------------------------------------------------------------
# Radial coordinate
# -------------------------------------------------------------------------
# x = r_source - RREF
x = r0
x_plot = x * 2 / D


# -------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4, 3))

ax.plot(
    x_plot,
    # kernel_abs,
    m11/m01,
    # label=r'Kernel $|G|$',
    color='r',
    label=r'$|K_1(r-r_0)|$',
    lw=2
)

ax.plot(
    x_plot,
    m12/m02,
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
ax.set_ylabel(r'$K/K_{\mathrm{max}}$')


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
    os.path.join(folder_name, f"kernels.pdf"),
    dpi=300,
    bbox_inches="tight",
)