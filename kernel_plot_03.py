"""
Plot the Green kernel |G| and its gradient |∇G| across the radial direction.
"""

import numpy as np
import matplotlib.pyplot as plt
from TailoredGreen.CylinderGreen import CylinderGreen
from Constants.helpers import read_force_file, plot_3D_directivity, plot_3D_phase_directivity, plot_beam_azimuth, plot_rotation_arrow

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
r_inner, Fz, Fphi  = read_force_file('./Data/Zamponi2026/FS_ISAE_2_8000.txt')

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
    -5 * D + RREF,
    5 * D + RREF,
    250
)

sources = np.vstack([
    r_source,
    np.ones_like(r_source) * PHIREF * RREF,
    np.zeros_like(r_source)
])

rho0 = 1.2
B = 2
m = 1
Omega = 8000/60 * 2 * np.pi
t__c = 0.0803
c = np.ones_like(r_inner) * 0.025 
A = -1/2/np.pi * rho0 * m**2 * B ** 3 * Omega ** 2 * t__c * c ** 2
B = -1/2/np.pi * np.sqrt(Fz**2 + Fphi**2)

fig, ax = plt.subplots(figsize=(4, 3))

ax.plot(
    r_inner / RTIP,
    abs(A)/max(abs(A)),
    # label=r'Kernel $|G|$',
    color='r',
    label=r'$|A(r)|/|A_{\mathrm{max}}|$',
    lw=2
)

ax.plot(
    r_inner / RTIP,
    abs(B)/max(abs(B)),
    # label=r'Gradient $|\nabla G|$',
    color='b',
    label=r'$|\boldsymbol{B}(r)|/|\boldsymbol{B}_{\mathrm{max}}|$',
    lw=2,
    ls='--'
)



# -------------------------------------------------------------------------
# Labels / formatting
# -------------------------------------------------------------------------
ax.set_xlabel(r'$r/r_{\mathrm{tip}}$')
# ax.set_ylabel(r'Normalized magnitude $K/K_{\mathrm{max}}$')
# ax.set_ylabel(r'$K/K_{\mathrm{max}}$')


plt.minorticks_on()

ax.grid(which='major', axis='both', linestyle='-')
ax.grid(which='minor', linestyle='--', alpha=0.5)

ax.legend()
ax.grid(True, alpha=0.2)
ax.set_ylim(0, 1.05)
ax.set_xlim(RROOT/RTIP, 1)

plt.tight_layout()
plt.show()

import os
folder_name = f'./Figures/'
fig.savefig(
    os.path.join(folder_name, f"source_dist.pdf"),
    dpi=300,
    bbox_inches="tight",
)

