from PotentialInteraction.PIN import PotentialInteraction
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PotentialInteraction.beam_to_blade import NACA0012_T10_PIN, BladeLoadings
from PotentialInteraction.blade_to_beam import BeamLoadings
from Constants.helpers import read_force_file, p_to_SPL, spl_from_autopower, plot_BPF_peaks
from Hanson.far_field import HansonModel
import h5py

r_inner, Fz, Fphi = read_force_file('./Data/Zamponi2026/FS_ISAE_2_8000.txt')

dr = np.diff(r_inner)[0]
r_outer = np.hstack([r_inner-dr/2, r_inner[-1]+dr/2])
NRADIALSEGMENTS = np.shape(r_outer)[0]
NHARMONICS = 40
ms = np.arange(1, 16, 1)


pin = PotentialInteraction(
    twist_rad= np.deg2rad(10) * np.ones(NRADIALSEGMENTS),
    chord_m = 0.025 * np.ones(NRADIALSEGMENTS),
    radius_m=r_outer,
    t_c = np.ones_like(r_outer) * 0.12,
    # Uz0_mps=U_flow,
    Fzprime_Npm=Fz,
    Fphiprime_Npm=Fphi,
    B=2,
    Dcylinder_m=0.02,
    Lcylinder_m=0.02,
    Omega_rads=8000/60*2*np.pi,
    rho_kgm3=1.2,
    c_mps=340.0,
    kmax=NHARMONICS,
    nb=1,
    numerics={'Nphi': 720, 'Nthetab': 36*2, 'include_vortex_sources':True, 'include_thickness_sources':True}
)
theta = np.linspace(0, np.pi, 100001)
# chord_stations_outer = np.linspace(-1, 1, 101)
chord_stations_outer = -np.cos(theta)
chord_stations = (chord_stations_outer[1:] + chord_stations_outer[:-1]) / 2
# pin.plotDownwashInRotorPlane()
# plt.show()

# pin.plotStrutLoading3D()
# plt.show()
pin.plotBladeLoadingPerUnitArea(m=1, chord_stations = chord_stations)
plt.show()


FbladeSears = pin.getBladeLoadingHarmonics()

FbladeAmiet_dist, FbladeAmiet, _, _, _ = pin.getBladeLoadingHarmonicsAmiet(chord_stations=chord_stations)

k = pin.k

fig, ax = plt.subplots()

ax.plot(k, np.abs(FbladeAmiet[1, :, 30]), marker='s', color='r', label='Amiet')
ax.plot(k, np.abs(FbladeSears[1, :, 30]), marker='^', color='b', label='Sears')

ax.plot(k, np.abs(FbladeAmiet[2, :, 30]), marker='s', color='r', linestyle='dashed')
ax.plot(k, np.abs(FbladeSears[2, :, 30]), marker='^', color='b', linestyle='dashed')
ax.set_xlabel('k')
ax.set_ylabel('$|F^z_{beam}|$ [N/m]')
ax.legend()
ax.grid()
plt.title('Beam Loadings')
plt.show()


fig, ax = plt.subplots()

ax.plot(k, np.real(FbladeAmiet[1, :, 30]), marker='s', color='r', label='Amiet')
ax.plot(k, np.real(FbladeSears[1, :, 30]), marker='^', color='b', label='Sears')

ax.plot(k, np.real(FbladeAmiet[2, :, 30]), marker='s', color='r', linestyle='dashed')
ax.plot(k, np.real(FbladeSears[2, :, 30]), marker='^', color='b', linestyle='dashed')
ax.set_xlabel('k')
ax.set_ylabel('$|F^z_{beam}|$ [N/m]')
ax.legend()
ax.grid()
plt.title('Beam Loadings')
plt.show()

fig, ax = plt.subplots()

ax.plot(k, np.imag(FbladeAmiet[1, :, 30]), marker='s', color='r', label='Amiet')
ax.plot(k, np.imag(FbladeSears[1, :, 30]), marker='^', color='b', label='Sears')

ax.plot(k, np.imag(FbladeAmiet[2, :, 30]), marker='s', color='r', linestyle='dashed')
ax.plot(k, np.imag(FbladeSears[2, :, 30]), marker='^', color='b', linestyle='dashed')
ax.set_xlabel('k')
ax.set_ylabel('$|F^z_{beam}|$ [N/m]')
ax.legend()
ax.grid()
plt.title('Beam Loadings')
plt.show()

fig, ax = plt.subplots()

ax.plot(k, np.rad2deg(np.angle(FbladeAmiet[1, :, 30])), marker='s', color='r', label='Amiet')
ax.plot(k, np.rad2deg(np.angle(FbladeSears[1, :, 30])), marker='^', color='b', label='Sears')

ax.plot(k, np.rad2deg(np.angle(FbladeAmiet[2, :, 30])), marker='s', color='r', linestyle='dashed')
ax.plot(k, np.rad2deg(np.angle(FbladeSears[2, :, 30])), marker='^', color='b', linestyle='dashed')

ax.set_xlabel('k')
ax.set_ylabel('$Arg(F^z_{beam})$ [deg]')
ax.legend()
ax.grid()
plt.title('Beam Loadings')
plt.show()



