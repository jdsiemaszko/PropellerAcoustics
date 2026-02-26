import h5py
from scipy.special import hankel2, jve, jv
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PotentialInteraction.beam_to_blade import BladeLoadings
from PotentialInteraction.blade_to_beam import BeamLoadings
from PotentialInteraction.placeholder import *
from Hanson.far_field import HansonModel
from Hanson.near_field import NearFieldHansonModel
import matplotlib.animation as animation
from Constants.helpers import p_to_SPL

import numpy as np

# ------------------------------------------------------------------
# Define shared single instances (arrays & scalars)
# ------------------------------------------------------------------

NSEG = 20
radius_m = np.linspace(0.016, 0.1, NSEG+1)
radius_c = (radius_m[1:] + radius_m[:-1]) / 2
dr = np.diff(radius_m)
twist_rad = np.ones(NSEG+1) * np.deg2rad(10)
chord_m = np.ones(NSEG+1) * 0.025
Uz0_mps = -np.interp(radius_c / 0.1, R_RT_UDW, UDW_EXACT) # interpolated from data
Tprime_Npm = np.interp(radius_c / 0.1, R_RT_EXACT, DT_EXACT) # per blade
Qprime_Npm = np.interp(radius_c / 0.1, R_RT_EXACT, DQ_EXACT) # per blade

T_TARGET = 1.075 # N
TORQUE_TARGET = 0.0134 # Nm

T_N = Tprime_Npm * dr

T_TOT = np.sum(T_N)

T_N *= T_TARGET / T_TOT
Tprime_Npm = T_N / dr # per unit span
alpha = TORQUE_TARGET / np.sum(Tprime_Npm * radius_c * dr)
Qprime_Npm = T_N / dr * alpha # such that sum(Qprime_Npm * r * dr) = TORQUE_TARGET 


B = 2
Dcylinder_m = 0.02
Lcylinder_m = 0.02
Omega_rads = 8000 / 60 * 2 * np.pi
rho_kgm3 = 1.2
c_mps = 340
kmax = 40
nb = 1

NPHI = 36
NTHETA = 18

axis = np.array([0.0, 0.0, 1.0])
origin = np.array([0.0, 0.0, 0.0])

# ------------------------------------------------------------------
# Class initializations using shared instances
# ------------------------------------------------------------------

blade_l = BladeLoadings(
    twist_rad=twist_rad,
    chord_m=chord_m,
    radius_m=radius_m,
    Uz0_mps=Uz0_mps,
    Tprime_Npm=Tprime_Npm,
    Qprime_Npm=Qprime_Npm,
    B=B,
    Dcylinder_m=Dcylinder_m,
    Lcylinder_m=Lcylinder_m,
    Omega_rads=Omega_rads,
    rho_kgm3=rho_kgm3,
    c_mps=c_mps,
    kmax=kmax,
    nb=nb
)

beam_l = BeamLoadings(
    twist_rad=twist_rad,
    chord_m=chord_m,
    radius_m=radius_m,
    Uz0_mps=Uz0_mps,
    Tprime_Npm=Tprime_Npm,
    Qprime_Npm=Qprime_Npm,
    B=B,
    Dcylinder_m=Dcylinder_m,
    Lcylinder_m=Lcylinder_m,
    Omega_rads=Omega_rads,
    rho_kgm3=rho_kgm3,
    c_mps=c_mps,
    kmax=kmax,
    nb=nb
)

han = HansonModel(
    axis=axis,
    origin=origin,
    twist_rad=twist_rad,
    chord_m=chord_m,
    radius_m=radius_m,
    B=B,
    Omega_rads=Omega_rads,
    rho_kgm3=rho_kgm3,
    c_mps=c_mps,
    nb=nb
)

# han_nf = NearFieldHansonModel(
#     axis=axis,
#     origin=origin,
#     twist_rad=twist_rad,
#     chord_m=chord_m,
#     radius_m=radius_m,
#     B=B,
#     Omega_rads=Omega_rads,
#     rho_kgm3=rho_kgm3,
#     c_mps=c_mps,
#     nb=nb
# )

BLH = blade_l.getBladeLoadingHarmonics() # shape 3, Nk, Nr
BeamLH = beam_l.getBeamLoadingHarmonics() # shape 3, Nk, Nr

# Nx = 100*2+1
# Ny = 100*2+1
# Nt = 4 * 20+1
# x, y =  np.linspace(-4*Dcylinder_m, 4*Dcylinder_m, Nx), np.linspace(-4*Dcylinder_m, 4*Dcylinder_m, Ny)
# X, Y = np.meshgrid(
#     x, y
# )
# positions = np.stack((X.flatten(), Y.flatten()))
# period = 2 * np.pi / B / OMEGA 
# time = np.linspace(-period/2, period/2, Nt)

# pressure, _ = beam_l._getBeamVortexPressure(time, overwrite_positions=positions) # shape Npoints, Nt, Nr

# pressure_to_plot = pressure[:, :, -1]
# # index = np.where(X**2+Y**2 < (Dcylinder_m/2)**2)
# # pressure_to_plot[index, :] = 0.0
# pressure_reshaped = pressure_to_plot.reshape(Nx,Ny,Nt)
# # --- Grid reconstruction ---

# # --- Define explicit contour range ---
# vmin = -100        # <-- SET THIS
# vmax =  100        # <-- SET THIS
# n_levels = 60

# # levels = np.linspace(vmin, vmax, n_levels)
# levels = np.array([-100, -75, -50, -25, 0, 25, 50, 75, 100])

# # --- Figure ---
# fig, ax = plt.subplots(figsize=(6, 5))

# contour = ax.contourf(
#     X, Y,
#     pressure_reshaped[:, :, 0],
#     levels=levels,
#     cmap='RdBu_r',
#     # extend='both'
#         extend="neither",

# )

# cbar = fig.colorbar(contour)
# cbar.set_label("Pressure")

# ax.set_xlabel("$x/D_c$")
# ax.set_ylabel("$y/D_c$")
# # ax.set_xlim(-0.05, 0.05)
# # ax.set_ylim(-0.05, 0.05)

# title = ax.set_title(f"Pressure field at $t/T = {time[0]/period:.3f}$")

# def update(frame):
#     global contour  # if inside function scope, otherwise use global

#     # Remove previous contour safely (modern way)
#     contour.remove()

#     contour = ax.contourf(
#         X/Dcylinder_m, Y/Dcylinder_m,
#         pressure_reshaped[:, :, frame],
#         levels=levels,
#         cmap="RdBu_r",
#         extend="neither",
#     )

#     title.set_text(f"$t/T = {time[frame]/period:.3f}$")

#     return []   # IMPORTANT: no collections needed when blit=False

# ani = animation.FuncAnimation(
#     fig,
#     update,
#     frames=Nt,
#     interval=50,
#     blit=False
# )

# # IMPORTANT: draw once before saving
# fig.canvas.draw()
# plt.tight_layout()

# ani.save("Figures/pressure_animation.gif", writer="pillow")
# plt.show()

han.plot3DdirectivityRotor(m=5, loadings=BLH, R=1.62, Nphi=NPHI, Ntheta=NTHETA, valmax=65, valmin=40)
plt.show()
han.plot3DdirectivityStator(m=5, loadings=BeamLH, R=1.62, Nphi=NPHI, Ntheta=NTHETA, valmax=65, valmin=40)
plt.show()
han.plot3DdirectivityTotal(m=1, loadings=BLH, loadings_2=BeamLH, R=1.62, Nphi=NPHI, Ntheta=NTHETA, valmax=65, valmin=40)
plt.show()
han.plot3DdirectivityTotal(m=4, loadings=BLH, loadings_2=BeamLH, R=1.62, Nphi=NPHI, Ntheta=NTHETA, valmax=65, valmin=40)
plt.show()
han.plot3DdirectivityTotal(m=5, loadings=BLH, loadings_2=BeamLH, R=1.62, Nphi=NPHI, Ntheta=NTHETA, valmax=65, valmin=40)
plt.show()





