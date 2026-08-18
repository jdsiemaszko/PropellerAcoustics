"""
plot the kernel G(*|y) across the radial direction
"""
import numpy as np
import matplotlib.pyplot as plt
from TailoredGreen.CylinderGreen import CylinderGreen
L = 0.02
D = 0.02
RTIP = 0.1
RROOT = RTIP * 0.16

RREF = 0.75 * RTIP
PHIREF = 0.0
THETAREF = np.pi / 2
m=np.array([1, 2, 3, 4, 5])
# m=np.array([1])

B=2
RPM = 8000
c0 = 340
wavenumber = m * B * RPM / 60 * 2 * np.pi / c0

observer = np.array([RREF, D/2 * (1+1e-12) * np.cos(THETAREF), -L + D/2 * (1+1e-12) * np.sin(THETAREF)]).reshape(3,1)

# r_source = np.linspace(RROOT, RTIP, 100)
r_source = np.linspace(-5 * D + RREF, 5 * D + RREF, 100)

sources = np.vstack([
    r_source,
      np.ones_like(r_source) * PHIREF * RREF,
      np.zeros_like(r_source)
])

green = CylinderGreen(D/2, axis=np.array([1, 0, 0]), origin=np.array([0, 0, -L]), radial=np.array([0, 1, 0]), dim=3,
                      numerics = {
                    'nmax': 16,
                    'Nq_prop': 32,
                    'Nq_evan': 16*10,
                    'eps_radius' : 1e-24, # must be lower than eps_eval!
                    'Nazim' : 9, # discretization of the boundary in the azimuth
                    'Nax': 32, # in the axial direction
                    'RMAX': 20, # max radius!
                    'mode': 'uniform', # uniform or geometric, defines the spacing of the surface panels!
                    'geom_factor': 1.025, # geometric stretching factor, only used if mode is 'geometric'
                    'eps_eval' : 1e-12 # evaluation distance from the actual surface, as a fraction of cylinder radius!
                    # Note: the function is currently NOT checking if the panels are compact!
                    })

kernel = green.getGreenFunction(observer, sources, wavenumber) # shape Nk, Nx, Ny
# kernel = np.linalg.norm(green.getGradientGreenAnalytical(observer, sources, wavenumber), axis=0) # shape Nk, Nx, Ny)

kernel /= np.max(np.abs(kernel))

from scipy.optimize import curve_fit

# -------------------------------------------------------------------------
# Kernel data
# -------------------------------------------------------------------------
x = r_source - RREF
y = np.abs(kernel[:, 0, :]).T.squeeze()




# -------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------
fig, ax = plt.subplots()

x_plot = x * 2 / D

ax.plot(
    x_plot,
    y,
    label='Kernel $|G|$'
)

# -------------------------------------------------------------------------
# Blade radial extent: RROOT -> RTIP
#
# Convert to the same coordinate as the x-axis:
# x = (r - RREF) * 2/D
# -------------------------------------------------------------------------
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
y_arrow = 0.08 * ax.get_ylim()[1]

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
    r' $r_{\mathrm{tip}}-r_{\mathrm{root}}$',
    ha='center',
    va='bottom'
)

# Arrow showing 2 sigma core size
y_core = 0.18 * ax.get_ylim()[1]

# -------------------------------------------------------------------------
# Labels / formatting
# -------------------------------------------------------------------------
ax.set_xlabel(r'$(r-r_0)/R$')
ax.set_ylabel(r'$|G|$')

ax.legend()
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()




