import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PotentialInteraction.blade_to_beam import BeamLoadings
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

# loadings = beam_l.getBeamLoadingHarmonics()
# pass

loadings_in_time = beam_l._getBeamVortexLoads(time)

fig, ax = plt.subplots()

nr = 15

print(r_pos[15], r[15])

period = 2 * np.pi / B / Omega
ax.plot(time / period, Fx[nr, :], color='b',linestyle='dashed')
ax.plot(time / period, -Fphi[nr, :], color='r',linestyle='dashed') # opposite sign 
ax.plot(time / period, loadings_in_time[1, :, nr], color='b', label='$F_x$')
ax.plot(time / period, loadings_in_time[2, :, nr], color='r', label='$F_\phi$')

ax.set_xlabel('$t/T$')
ax.set_ylabel('$F$ [N/m]')
ax.legend()
ax.grid()
plt.tight_layout()
plt.show()

points_per_period = 20
k_local = np.arange(1, 41, 1)
T_periodic = np.linspace(-period/2, period/2, points_per_period * int(np.max(k_local)), endpoint=False) # Np
T_periodic, F_beam = periodic_sum_interpolated(loadings_in_time, period=period, time=time, kind='cubic', t_new=T_periodic)
dt = T_periodic[1] - T_periodic[0]
Np = T_periodic.shape[0] # should be equal to points_per_period * max(k_local)!

# shape (3, Nk, Nr)

F_beam_k = 1/period * np.sum(F_beam[:, None, :, :] * np.exp(1j *
            k_local[None, :, None, None] * 2 * np.pi / period * 
            T_periodic[None, None, :, None]) * dt, axis=2) # our convention for fourier transform: minus in the exp




fig, ax = plt.subplots()

ax.plot(k_local, np.abs(Fx_k[nr, :])*B, color='b',linestyle='dashed', marker='s')
ax.plot(k_local, np.abs(Fphi_k[nr, :])*B, color='r',linestyle='dashed', marker='s') # opposite sign 
ax.plot(k_local, np.abs(F_beam_k[1, :, nr]), color='b', label='$F_x$', marker='^')
ax.plot(k_local, np.abs(F_beam_k[2, :, nr]), color='r', label='$F_\phi$', marker='^')

ax.set_xlabel('$k$')
ax.set_ylabel('$F$ [N/m]')
ax.set_xlim(0, 20)
ax.legend()
ax.grid()
plt.tight_layout()
plt.show()

# Initialize Module
NSEG = len(r0)
hm = HansonModel(twist_rad = np.deg2rad(10 * np.ones(NSEG+1)), # blade twist array [rad] of size Nr+1 (segment edges)
                chord_m = c * np.ones(NSEG+1), # blade chord array [m] of size Nr+1
                radius_m=r_outer, # blade radius stations [m] of size Nr + 1
                axis=np.array([0, 0, 1]), origin=np.array([0, 0, 0]), radial=np.array([1, 0, 0]), # coordinate system (not needed here)
                B=2, # number of blades
                Omega_rads=Omega, # rotation speed [rad/s]
                rho_kgm3=rho, # fluid density [kg/m^3]
                c_mps= 340, # speed of sound [m/s]
                nb = 1 # number of beams (irrelevant)
                )
# hm.dr = np.ones(NSEG) # overwrite cell size to pass loadings in Newtons


# data = scipy.io.loadmat('Data/Vella2026/PREF27_BEAM.mat')
# P_REF = data['PREF27'][:, 0] * B
data = scipy.io.loadmat('Data/Vella2026/PREF19_BEAM.mat')
P_REF = data['PREF19'][:, 0] * B
SPL_REF = p_to_SPL(P_REF)

data = scipy.io.loadmat('Data/Vella2026/BPF_BEAM.mat')
BPF_REF = data['BPF'][0] / Omega / B * 2 * np.pi

p, _ = hm.getPressureStator(
        x = np.array([R * np.cos(theta) * np.cos(phi), R * np.cos(theta) * np.sin(phi), R*np.sin(theta)]).T, # position to plot the spectrum at
        m = np.arange(1, 21, 1)*B, # modes to compute, only mutiples of BPF!
        Fbeam=beam_l.getBeamLoadingHarmonics())

# 3) Plot spectrum at a point
fig, ax = plt.subplots()
ax.plot(BPF_REF, p_to_SPL(p.reshape((20,))), color='r', marker='x', markersize=10)
ax.plot(BPF_REF, SPL_REF, color='k', linestyle='dashed', marker='^')
ax.set_xlabel('$f^+$')
ax.set_ylabel('$SPL$')
ax.grid()
plt.tight_layout()
plt.show()




fig, ax = plt.subplots()
ax.plot(np.arange(1, 11, 1), np.angle(p.reshape((10,))), color='r', marker='x', markersize=10)
ax.plot(BPF_REF, np.angle(P_REF), color='k', linestyle='dashed', marker='^')


plt.tight_layout()
plt.show()
