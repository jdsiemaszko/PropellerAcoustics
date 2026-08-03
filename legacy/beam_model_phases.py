"""
check phase of all terms relevant in the beam PIN model: from upwash harmonics on the blades to the beam loading
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PotentialInteraction.blade_to_beam import BeamLoadings
from PotentialInteraction.beam_to_blade import BladeLoadings
import scipy.io
from Constants.helpers import periodic_sum_interpolated
from Hanson.far_field import HansonModel
from Constants.helpers import p_to_SPL

# INPUTS +++++++++++++++++++++++++++++++++++++++++++++++++++++++=
# Parameters
rho = 1.2
B = 2
RPM = 8000
Omega = RPM * 2 * np.pi / 60
D = 0.2
r_tip = D / 2
r_root = 0.016
r = np.linspace(r_root, r_tip, 20)
dr = r[1] - r[0]
r_outer = np.concatenate((r-dr/2, [r[-1]+dr/2]))
c = 0.025
D_bras = 0.02
g = 0.02  # + for RD, - for RU

R = 1.62
theta = 0 * np.pi/180
phi = 90 * np.pi/180
phi = np.pi - phi

# Read flow data
uy_m01R = pd.read_csv('Data/Vella2026/uy_m01R.csv', header=None).to_numpy()

Omega_num = 6000 * 2 * np.pi / 60
r_tip_num = 0.125

# Scale uy data
uy_m01R[:, 1] = uy_m01R[:, 1] / (Omega_num * r_tip_num) * (Omega * r_tip)

# Interpolate flow along blade radius
U_flow_m01R = np.zeros(len(r))
for j in range(len(r)):
    r_rT = r[j] / r_tip
    idx_m01R = np.where(uy_m01R[:, 0] <= r_rT)[0]
    U_flow_m01R[j] = uy_m01R[idx_m01R[-1], 1]

# Induced velocity
Vi = np.abs(U_flow_m01R)
Ur = np.sqrt((Omega * r)**2 + Vi**2)
OmegaR = Omega * r

f0 = RPM / 60
dr = np.abs(r_tip - r_root) / len(r)
A = c * (r_tip - r_root)
dA = c * dr

# Read thrust and torque data
dataT = pd.read_csv('Data/Vella2026/thrust_B2_D20_NACA0012.csv', header=None, sep='\t').to_numpy()
dataQ = pd.read_csv('Data/Vella2026/torque_B2_D20_NACA0012.csv', header=None, sep='\t').to_numpy()

T = dataT[4, 1] / 2  # One blade [N]
Q = dataQ[4, 1] / 2 / 1000  # One blade [N.m] (data in N.mm)

# Read distributed load data
data_3 = pd.read_csv('Data/Vella2026/donnees_load_distrib_TJ.csv', header=None, sep='\t').to_numpy()
r_bis = data_3[:, 0] * r_tip
T_bis = data_3[:, 1]

r0 = np.linspace(r_root + 0.01, r_tip, 20)
distrib_T = np.zeros(len(r))
idx_r1 = np.zeros(len(r), dtype=int)

for j in range(len(r)):
    idx_r1[j] = np.max(np.where(r_bis <= r0[j])[0])
    distrib_T[j] = T_bis[idx_r1[j]] * dr

T_tot = np.sum(distrib_T)
dT = T * distrib_T / T_tot
dQ = Q * distrib_T / T_tot

# Lift and drag coefficients
Cd = dQ / (0.5 * rho * r * Ur**2 * dA)
Cl = dT / (0.5 * rho * Ur**2 * dA)

# Lift per unit span
Lift = 0.5 * rho * Cl * dA * Ur**2 / dr

dr = np.diff(r_outer)

# PARSE PARTIALS
data = scipy.io.loadmat('Data/Vella2026/time.mat')
time = data['time_shifted'][0] 

data = scipy.io.loadmat('Data/Vella2026/stations.mat')
r_pos = data['r'][0] 

data = scipy.io.loadmat('Data/Vella2026/Ft_x.mat')
Fx = data['Ft_x'] 

data = scipy.io.loadmat('Data/Vella2026/Ft_phi.mat')
Fphi = data['Ft_phi'] 

data = scipy.io.loadmat('Data/Vella2026/Fk_x.mat')
Fx_k = data['Fk_x'] 

data = scipy.io.loadmat('Data/Vella2026/Fk_phi.mat')
Fphi_k = data['Fk_phi'] 

# MY CODE +++++++++++++++++++++++++++++++++++++++++++++++++++++++=


beam_l = BeamLoadings(
    twist_rad=np.deg2rad(10)* np.ones(r_outer.shape),
    chord_m=c* np.ones(r_outer.shape),
    radius_m=r_outer,
    Uz0_mps=-U_flow_m01R,
    Tprime_Npm=dT / dr,
    Qprime_Npm=dQ / dr,
    B=B,
    Dcylinder_m=D_bras,
    Lcylinder_m=g,
    Omega_rads=Omega,
    rho_kgm3=rho,
    c_mps=340,
    kmax=40,
    nb=1
)

blade_l = BladeLoadings(
    twist_rad=np.deg2rad(10)* np.ones(r_outer.shape),
    chord_m=c* np.ones(r_outer.shape),
    radius_m=r_outer,
    Uz0_mps=-U_flow_m01R,
    Tprime_Npm=dT / dr,
    Qprime_Npm=dQ / dr,
    B=B,
    Dcylinder_m=D_bras,
    Lcylinder_m=g,
    Omega_rads=Omega,
    rho_kgm3=rho,
    c_mps=340,
    kmax=40,
    nb=1
)

BLH = blade_l.getBladeLoadingHarmonics() # 3, Nk, Nr
time = np.linspace(-np.pi/Omega, np.pi/Omega, 1000)
w, _ = blade_l.getUpwashInTime(time) # Nt, Nr
Lblade, _ = blade_l.getLoadingInTime(time) # 3, Nt, Nr
Gamma, _ = beam_l.getGammaInTime(time, BLH=BLH) # Nt, Nr
Gamma_steady, _ = beam_l.getGammaInTime(time, BLH=None) # Nt, Nr
Lbeam, _ = beam_l.getLoadingInTime(time, BLH=BLH) # 3, Nt, Nr
Lbeam_steady, _ = beam_l.getLoadingInTime(time, BLH=None) # 3, Nt, Nr

period = 2 * np.pi / Omega

# choose radial station index
ir = -5  # change as needed

fig, ax = plt.subplots(figsize=(8, 5))

def rescale(arr):
    """Rescale array to same order of magnitude (divide by max abs)."""
    m = np.max(np.abs(arr))
    # mean = np.mean(arr)
    # arr_rescaled = (arr - mean) / m + mean
    return arr / m if m != 0 else arr

# --- scalar fields ---
ax.plot(time/period, rescale(w[:, ir]), label='$w$')
ax.plot(time/period, rescale(Gamma[:, ir]), label='$\Gamma$ (Unsteady)')
ax.plot(time/period, rescale(Gamma_steady[:, ir]), label='$\Gamma$ (Steady)')

# --- vector fields (loop over 3 components) ---
for i in range(2):
    if i == 0:
        string = '(axial)'
    elif i==1:
        string = '(tangential)'
    ax.plot(time/period, rescale(Lblade[i+1, :, ir]), linestyle='--', label=f'$F_{{blade}}$ {string}')
    ax.plot(time/period, rescale(Lbeam[i+1, :, ir]), linestyle='-.', label=f'$F_{{beam, unsteady}}$ {string}')
    ax.plot(time/period, rescale(Lbeam_steady[i+1, :, ir]), linestyle=':', label=f'$F_{{beam, steady}}$ {string}')

# formatting
ax.set_xlabel('$t/T$')
ax.set_ylabel('Value (normalized)')
# ax.set_title(f'Signals at radial station ir={ir}')
ax.legend(ncol=1, fontsize=8)
ax.grid(True)

plt.tight_layout()
plt.show()
