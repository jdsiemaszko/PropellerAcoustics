import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from Hanson.far_field import HansonModel
from PotentialInteraction.beam_to_blade import BladeLoadings

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

# ----------------------------------------------------------
# Radial discretization
# ----------------------------------------------------------
r0 = np.linspace(rR, rT, 20)

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

# replicate MATLAB edge correction
dT[-2:] = Fz_tour_bis[-1]
dQ[-2:] = Fy_tour_bis[-1]

dT *= TREF / np.sum(dT)

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
theta = 0 * np.pi/180
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


data = scipy.io.loadmat('Data/Vella2026/Fblade.mat')
Fblade = data['Lpus'] 
Fblade_x = -Fblade * np.sin(alpha)
Fblade_phi = Fblade * np.cos(alpha)


# MY CODE
ddr = r0[1] - r0[0]
r_outer = np.concatenate((r0-ddr/2, [r0[-1]+ddr/2]))
ddr = np.diff(r_outer)

blade_l = BladeLoadings(
    twist_rad=np.deg2rad(pitch)* np.ones(r_outer.shape),
    chord_m=c* np.ones(r_outer.shape),
    radius_m=r_outer,
    Uz0_mps=U_flow,
    Tprime_Npm= np.zeros(r_outer.shape[0] - 1),
    # dT / dr,
    Qprime_Npm= np.zeros(r_outer.shape[0] - 1),
    #   dQ / dr,
    B=B,
    Dcylinder_m=D_bras,
    Lcylinder_m=g,
    Omega_rads=Omega,
    rho_kgm3=rho0,
    c_mps=c0,
    kmax=40,
    nb=1
)

F_blade_k = blade_l.getBladeLoadingHarmonics()
k_local = np.arange(1, 41, 1)

# F_blade_k[:, 1:, :] *= (-1)**k_local[None, :, None]

nr = 15

period = 2 * np.pi / B / Omega


fig, ax = plt.subplots()

ax.scatter(k_local, np.imag(Fblade_x[:, nr]), color='b',
        #    linestyle='dashed',
             marker='s')
ax.scatter(k_local, -np.imag(Fblade_phi[:, nr]), color='r',
        #    linestyle='dashed',
             marker='s') # opposite sign
ax.plot(k_local, np.imag(F_blade_k[1, 1:, nr]), color='b', label='$F_x$'
        # , marker='^'
        )
ax.plot(k_local, np.imag(F_blade_k[2, 1:, nr]), color='r', label='$F_\phi$'
        # , marker='^'
        )


print(np.max(np.abs(F_blade_k[1, 1:, :] - Fblade_x[:, :])))
print(np.max(np.abs(F_blade_k[2, 1:, :] + Fblade_phi[:, :])))


ax.set_xlabel('$k$')
ax.set_ylabel('$F$ [N/m]')
ax.set_xlim(0, 20)
ax.legend()
ax.grid()
plt.tight_layout()
plt.show()

# fig, ax = plt.subplots()

# ax.plot(k_local, np.abs(F_blade_k[1, 1:, nr] - np.conjugate(Fblade_x[:, nr])), color='b', label='$F_x$'
#         # , marker='^'
#         )
# ax.plot(k_local, np.abs(F_blade_k[2, 1:, nr]+ np.conjugate(Fblade_phi[:, nr])), color='r', label='$F_\phi$'
#         # , marker='^'
#         )

# ax.set_xlabel('$k$')
# ax.set_ylabel('$\Delta F$ [N/m]')
# ax.set_xlim(0, 20)
# ax.legend()
# ax.grid()
# plt.tight_layout()
# plt.show()


# Initialize Module
NSEG = len(r0)
hm = HansonModel(twist_rad = np.deg2rad(10 * np.ones(NSEG+1)), # blade twist array [rad] of size Nr+1 (segment edges)
                chord_m = c * np.ones(NSEG+1), # blade chord array [m] of size Nr+1
                radius_m=r_outer, # blade radius stations [m] of size Nr + 1
                axis=np.array([0, 0, 1]), origin=np.array([0, 0, 0]), radial=np.array([1, 0, 0]), # coordinate system (not needed here)
                B=2, # number of blades
                Omega_rads=Omega, # rotation speed [rad/s]
                rho_kgm3=rho0, # fluid density [kg/m^3]
                c_mps= c0, # speed of sound [m/s]
                nb = 1 # number of beams (irrelevant)
                )
hm.dr = np.ones(NSEG) # overwrite cell size to pass loadings as per-unit-span

# 1) Plot directivity (3D)
# fig = plt.figure(figsize=(7, 7))
# ax1 = fig.add_subplot(111, projection="3d")
# hm.plot3Ddirectivity(
#     fig=fig,
#     ax=ax1,
#     m=4, # harmonic to plot
#     R=R, # observation radius
#     Nphi=36*2, # plotting params
#     Ntheta=18*2,
#     valmin=10,
#     valmax=65,
#     # title='far-field',
#     mode='rotor', # 'rotor' or 'stator'
#     loadings=F_blade_k # blade loading harmonics
# )
# plt.tight_layout()
# plt.show()

# 2) Plot directivity (2D contour)
# not yet implemented!

data = scipy.io.loadmat('Data/Vella2026/SPL0.mat')
SPL_REF = data['SPL_REF'][:, 0] 

data = scipy.io.loadmat('Data/Vella2026/BPF.mat')
BPF_REF = data['BPF'][0] / Omega / B * 2 * np.pi


# 3) Plot spectrum at a point
fig, ax = plt.subplots()
hm.plotPressureSpectrum(fig=fig, ax=ax, 
                        x = np.array([R * np.cos(theta) * np.cos(phi), R * np.cos(theta) * np.sin(phi), R*np.sin(theta)]).T, # position to plot the spectrum at
                        m = np.arange(1, 11, 1), # modes to compute
                        loadings=F_blade_k * dr, # blade loading harmonics
                        plot_kwargs={'color': 'r', 'marker':'x', 'markersize':10}
                            )
ax.plot(BPF_REF, SPL_REF, color='k', linestyle='dashed', marker='^')
plt.tight_layout()
plt.show()
