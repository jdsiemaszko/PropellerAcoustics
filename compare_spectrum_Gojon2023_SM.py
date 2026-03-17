import h5py
# from scipy.special import hankel2, jve, jv
import numpy as np
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PotentialInteraction.beam_to_blade import BladeLoadings
# from PotentialInteraction.blade_to_beam import BeamLoadings
from TailoredGreen.CylinderGreen import CylinderGreen
from TailoredGreen.TailoredGreen import TailoredGreen
from SourceMode.SourceMode import SourceModeArray
from Constants.helpers import p_to_SPL, plot_BPF_peaks, spl_from_autopower
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat

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
TORQUEREF = 26.7/B/1000 # Nm


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
# dT[-2:] = Fz_tour_bis[-1]
# dQ[-2:] = Fy_tour_bis[-1]

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

OmegaD = Omega

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
Nk = 40

twist = np.deg2rad(pitch)* np.ones(r_outer.shape)
chord = c* np.ones(r_outer.shape)
blade_l = BladeLoadings(
    twist_rad= twist,
    chord_m= chord,
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
    kmax=Nk,
    nb=1
)

# CYLINDER GREEN'S FUNCTION
caxis = np.array([1.0, 0.0, 0.0])
D_prop = 0.2
D = 20 / 1000
L = 20 / 1000
corigin = np.array([0.0, 0.0, -L])

cg = CylinderGreen(radius=D/2, axis=caxis, origin=corigin, dim=3, 
                           numerics={
                        'mmax': 32, # mind that increasing this increases the chance of overflows, ***should*** be handled by the safe Bessel functions, but beware
                        'Nq_prop': 128,
                        'Nq_evan': 128,
                        'eps_k' : 1e-24,
                    }) # cylinder

# SOURCE MODE MODULE
axis_prop = np.array([0.0, 0.0, 1.0]) # z-direction propeller...
origin_prop = np.array([0.0, 0.0, 0.0]) # ... at z=0
NDIPOLES = 36
sourceArray = SourceModeArray(BLH=blade_l.getBladeLoadingMagnitude(), # loading per unit span
                        B = B,
                        Omega=Omega, gamma =twist,
                        axis=axis_prop, origin=origin_prop,
                        radius=r_outer,
                        green = cg,
                        numerics={'Ndipoles' : NDIPOLES},
                        c = c0
                        )

# _____________ PLOTTING & RESULTS _______________

# 1) plot geometry

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")
sourceArray.plotSelf(fig, ax)
# ax.set_box_aspect([1, 1, 1])
ax.set_aspect('equal')
ax.set_axis_off()
plt.show()
plt.close()

# PARSE EXPERIMENTAL

ind_theta = 6
ind_phi = 9
datadir = './Experimental/dataverse_files'
casefile = 'ISAE_2_D20_L20'

def load_h5(filename):
    return h5py.File(filename, "r")
with load_h5(f"{datadir}/{casefile}_autopower.h5") as f:
    g = f["ISAE_2_D20_L20"]
    freq = np.array(g["frequency_Hz"])
    ap = g["Autopower"]

    phi_exp = np.array(g["phi_deg"])[0][ind_phi] # azimuth
    theta_exp = np.array(g["theta_deg"])[0][ind_theta] # polar
    radius = g["radius_m"][0][0] # float

    BPF = B * RPM / 60
    data = ap[f"Autopower_RPM_{RPM}_Pa2"][:, ind_theta, ind_phi] # (freq, polar, azimuth), (aziuth=0 = > beam axis, azimuth=9 => 90 deg)
theta = 90 - theta_exp
phi = 180 - phi_exp
print(f'Theta_exp = {theta_exp} deg, Phi_exp = {phi_exp} deg')
print(f'Theta = {theta} deg, Phi = {phi} deg')
x_cart = np.array([
    radius * np.cos(np.deg2rad(phi)) * np.sin(np.deg2rad(theta)),
    radius * np.sin(np.deg2rad(phi)) * np.sin(np.deg2rad(theta)),
    radius * np.cos(np.deg2rad(theta)),
]).reshape((3, 1))

Nr = len(r0)
steady_loading = np.zeros((3, Nk+1, Nr), dtype=np.complex_)
steady_loading[1, 0, :] = dT / dr
steady_loading[2, 0, :] = dQ / dr

unsteady_loading = np.zeros((3, Nk+1, Nr), dtype=np.complex_)
unsteady_loading[:, 1:, :] = blade_l.getBladeLoadingHarmonics()[:, 1:, :]


ms = np.arange(1, 26, 1) # harmonics to extract
pmB_model_rotor = sourceArray.getPressure(x_cart, ms, 
                                    #    blade_l.getBladeLoadingHarmonics()
                                    steady_loading
                                       )[0][0]

ptmB_model_rotor = han.getThicknessNoiseRotor(x_cart, ms, c * np.ones_like(r0), 0.122 * np.ones_like(r0))[0][0] # NACA0012
pmB_model_beam = han.getPressureStator(x_cart, ms*B, # mind the different input harmonics!
                                        beam_l.getBeamLoadingHarmonics())[0][0]
pmB_model_total = pSmB_model_rotor + pUSmB_model_rotor + ptmB_model_rotor + pmB_model_beam
# p_rms_total = np.sqrt(np.abs(pSmB_model_rotor + pUSmB_model_rotor + ptmB_model_rotor)**2 + np.abs(pmB_model_beam)**2) # assuming incoherent

SPL_rotor_S = p_to_SPL(pSmB_model_rotor)
SPL_rotor_US = p_to_SPL(pUSmB_model_rotor)

SPL_rotor_thickness = p_to_SPL(ptmB_model_rotor)

SPL_beam = p_to_SPL(pmB_model_beam)

SPL_total = p_to_SPL(pmB_model_total)
# SPL_total = p_to_SPL(p_rms_total) # same computation

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(freq[0] / BPF, spl_from_autopower(data), label=f"Experimental", color='k', alpha=0.75)
fig, ax = plot_BPF_peaks(fig, ax, freq[0] / BPF, spl_from_autopower(data), N0=1, N1= 25, range=0.01, 
                         plot_kwargs={
                             'color':'k',
                             'linestyle':'dashed',
                             'alpha':0.75
                         })
ax.plot(ms, SPL_rotor_S, label=f"Model (rotor, steady loading)", color='r', marker='^')
ax.plot(ms, SPL_rotor_US, label=f"Model (rotor, unsteady loading)", color='g', marker='^')
ax.plot(ms, SPL_rotor_thickness, label=f"Model (rotor, thickness)", color='b', marker='+')
ax.plot(ms, SPL_beam, label=f"Model (beam, loading)", color='m', marker='o')
ax.plot(ms, SPL_total, label=f"Model (total)", color='k', marker='s')

ax.legend()
ax.set_xlabel("$f^+ = f/B/\Omega$ (Hz)")
ax.set_ylabel("SPL (dB)")
ax.set_xscale('log')
# ax.set_yscale('log')

ax.grid(visible=True, which='major', color='k', linestyle='-')
ax.grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.5)


plt.xlim(0.1, 100)
plt.ylim(0, 70)
plt.tight_layout()
plt.show()


