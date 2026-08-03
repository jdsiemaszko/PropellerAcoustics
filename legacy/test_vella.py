import h5py
from scipy.special import hankel2, jve, jv
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PotentialInteraction.beam_to_blade import BladeLoadings
from PotentialInteraction.blade_to_beam import BeamLoadings
# from PotentialInteraction.placeholder import *
from Hanson.far_field import HansonModel
from Hanson.near_field import NearFieldHansonModel
import matplotlib.animation as animation
from Constants.helpers import p_to_SPL
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat
import scipy.io

import numpy as np

# ==========================================================
# Hanson Far-field formula for a fixed distortion case
# ==========================================================

# ----------------------------------------------------------
# Test Conditions
# ----------------------------------------------------------
Mx = 0.0
c0 = 340.0
V = Mx * c0
rho0 = 1.2
NBEAMS = 2 # one beam for a radial strut

# ----------------------------------------------------------
# Blade Characteristics
# ----------------------------------------------------------
B = 2
D = 0.2
rT = D / 2
rR = 0.017
FA = 0
MCA = 0
pitch = 10
alpha = (90 - pitch) * np.pi / 180

TREF = 1.075
TORQUEREF = 26.7/2/1000 # Nm


# ----------------------------------------------------------
# Radial discretization
# ----------------------------------------------------------
NR = 10
r0 = np.linspace(rR, rT, 10)

RPM = 8000
f0 = RPM / 60
c = 0.025
Omega = RPM * 2 * np.pi / 60
V0 = r0 * Omega

dr = np.abs(rT - rR) / len(r0)
A = c * (rT - rR)
dA = c * dr

# ----------------------------------------------------------
# Aerodynamic load (.mat file)
# ----------------------------------------------------------
mat_data = loadmat('Data/Vella2026/Fz_Fy_aero_pale_sans_bras_tours1a5.mat')

Fz_tour_bis = mat_data['Fz_tour_bis'].flatten()
Fy_tour_bis = mat_data['Fy_tour_bis'].flatten()
x_load = mat_data['x'].flatten()

r_rT = r0 / rT

# --- 1D interpolation (MATLAB interp1 equivalent)
dT = np.interp(r_rT, x_load[:-1], Fz_tour_bis)
dQ = np.interp(r_rT, x_load[:-1], Fy_tour_bis)

dT *= TREF / np.sum(dT)
alpha = TORQUEREF / np.sum(dQ * r0)
dQ *= alpha

# ----------------------------------------------------------
# Aerodynamic coefficients
# ----------------------------------------------------------
Cd = dQ / (0.5 * rho0 * V0**2 * dA)
Cl = dT / (0.5 * rho0 * V0**2 * dA)

# ----------------------------------------------------------
# Complementary Data
# ----------------------------------------------------------
MT = rT * Omega / c0
U = np.sqrt(V**2 + (Omega * r0)**2)
Mr = U / c0

# ----------------------------------------------------------
# Observer
# ----------------------------------------------------------
R = 1.62
theta = 0.0
theta = np.pi/2 - theta
phi = 90 * np.pi/180
phi = np.pi - phi

OmegaD = Omega / (1 - Mx * np.cos(theta))

t = np.arange(0, 16, 1/(51.2e3))

# ----------------------------------------------------------
# Induced velocity field
# ----------------------------------------------------------
mat_vel = loadmat('Data/Vella2026/profils_vitesse_induite_axiale_8000RPM_B2_D20.mat')

xq = mat_vel['xq'].flatten()
zz = mat_vel['zz'].flatten()
w_mean = mat_vel['w_mean']

D_bras = 0.02
g = 0.02
z = np.array([-g + D_bras/2, -g, -g - D_bras/2])
z_rT = z / rT

# --- 2D interpolation (MATLAB interp2 equivalent)
interp_func = RegularGridInterpolator(
    (xq, zz),
    w_mean,
    bounds_error=False,
    fill_value=None
)

# Build interpolation points
points = np.array([[rr, zz_i] for zz_i in z for rr in r_rT])

U_flow_interp = interp_func(points)
U_flow_interp = U_flow_interp.reshape(len(z), len(r_rT))

U_flow = -U_flow_interp[0, :]

# ------------------------------------------------------------------
# Class initializations using shared instances
# ------------------------------------------------------------------
ddr = r0[1] - r0[0]
r_outer = np.concatenate((r0-ddr/2, [r0[-1]+ddr/2]))
ddr = np.diff(r_outer)

blade_l = BladeLoadings(
    twist_rad=np.deg2rad(pitch)* np.ones(r_outer.shape),
    chord_m=c* np.ones(r_outer.shape),
    radius_m=r_outer,
    Uz0_mps=U_flow,
    Tprime_Npm=dT / dr,
    Qprime_Npm=dQ / dr,
    B=B,
    Dcylinder_m=D_bras,
    Lcylinder_m=g,
    Omega_rads=Omega,
    rho_kgm3=rho0,
    c_mps=c0,
    kmax=40,
    nb=NBEAMS
)

beam_l = BeamLoadings(
    twist_rad=np.deg2rad(pitch)* np.ones(r_outer.shape),
    chord_m=c* np.ones(r_outer.shape),
    radius_m=r_outer,
    Uz0_mps=U_flow,
    Tprime_Npm= dT / dr,
    Qprime_Npm= dQ / dr,
    B=B,
    Dcylinder_m=D_bras,
    Lcylinder_m=g,
    Omega_rads=Omega,
    rho_kgm3=rho0,
    c_mps=c0,
    kmax=40,
    nb=NBEAMS
)

han = HansonModel(
radius_m=r_outer, # blade radius stations [m] of size Nr + 1
axis=np.array([0, 0, 1]), origin=np.array([0, 0, 0]), radial=np.array([1, 0, 0]), # coordinate system (not needed here)
B=2, # number of blades
Omega_rads=Omega, # rotation speed [rad/s]
rho_kgm3=rho0, # fluid density [kg/m^3]
c_mps= c0, # speed of sound [m/s]
nb=NBEAMS # number of beams (irrelevant)
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

NPHI = 36
NTHETA=18
VMIN = 10
VMAX = 65
han.plot3DdirectivityRotor(m=5, loadings=BLH, R=1.62, Nphi=NPHI, Ntheta=NTHETA, valmax=VMAX, valmin=VMIN)
plt.show()
han.plot3DdirectivityStator(m=5, loadings=BeamLH, R=1.62, Nphi=NPHI, Ntheta=NTHETA, valmax=VMAX, valmin=VMIN)
plt.show()
han.plot3DdirectivityTotal(m=5, loadings=BLH, loadings_2=BeamLH, R=1.62, Nphi=NPHI, Ntheta=NTHETA, valmax=VMAX, valmin=VMIN,
                           chord=c * np.ones_like(r0), t_c = 0.122 * np.ones_like(r0))
plt.show()

han.plot3DdirectivityTotal(m=1, loadings=BLH, loadings_2=BeamLH, R=1.62, Nphi=NPHI, Ntheta=NTHETA, valmax=VMAX, valmin=VMIN,
                           chord=c * np.ones_like(r0), t_c = 0.122 * np.ones_like(r0))
plt.show()
han.plot3DdirectivityTotal(m=4, loadings=BLH, loadings_2=BeamLH, R=1.62, Nphi=NPHI, Ntheta=NTHETA, valmax=VMAX, valmin=VMIN,
                           chord=c * np.ones_like(r0), t_c = 0.122 * np.ones_like(r0))
plt.show()

# han.plot2DdirectivityTotal(m=5, loadings=BLH, loadings_2=BeamLH, R=1.62, plane='xy')
# plt.show()

# han.plotDirectivityContour(m=5, loadings=BLH, loadings_2=BeamLH, R=1.62, Ntheta=NTHETA, valmax=VMAX, valmin=VMIN, mode='total')
# plt.show()

# han.plotDirectivityContour(m=5, loadings = BeamLH,R=1.62, Ntheta=NTHETA, valmax=VMAX, valmin=VMIN, mode='stator')
# plt.show()

han.plotDirectivityContour(m=5, loadings = BLH,R=1.62, Ntheta=NTHETA, valmax=VMAX, valmin=VMIN, mode='rotor')
plt.show()





