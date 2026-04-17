from PotentialInteraction.PIN import PotentialInteraction
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PotentialInteraction.beam_to_blade import NACA0012_T10_PIN, BladeLoadings

def read_force_file(filepath):
    data = np.loadtxt(filepath, skiprows=1)

    r = data[:, 0]
    Fx = data[:, 1]
    Fz = data[:, 2]

    return r, Fx, Fz

r_inner, Fz, Fphi = read_force_file('./Data/Zamponi2026/FS_ISAE_2_8000.txt')

dr = np.diff(r_inner)[0]
r_outer = np.hstack([r_inner-dr/2, r_inner[-1]+dr/2])
NRADIALSEGMENTS = np.shape(r_outer)[0]
NHARMONICS = 40


pin = PotentialInteraction(
    twist_rad= np.deg2rad(10) * np.ones(NRADIALSEGMENTS),
    chord_m = 0.025 * np.ones(NRADIALSEGMENTS),
    radius_m=r_outer,
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
    nb=1
)

blade_l = BladeLoadings(
    twist_rad= np.deg2rad(10) * np.ones(NRADIALSEGMENTS),
    chord_m = 0.025 * np.ones(NRADIALSEGMENTS),
    radius_m=r_outer,
    Uz0_mps=np.sqrt(Fz/  4 / np.pi / pin.rho / pin.seg_radius),
    Tprime_Npm=Fz,
    Qprime_Npm=Fphi,
    B=2,
    Dcylinder_m=0.02,
    Lcylinder_m=0.02,
    Omega_rads=8000/60*2*np.pi,
    rho_kgm3=1.2,
    c_mps=340.0,
    kmax=NHARMONICS,
    nb=1
)


pin.plotDownwashInRotorPlane()
plt.show()


Fblade = pin.getBladeLoadingHarmonics()
Fblade_old = blade_l.getBladeLoadingHarmonics()
k = pin.k
fig, ax = plt.subplots()

ax.plot(k, np.abs(Fblade[1, :, 30]), marker='s', color='r', label='PIN')
ax.plot(k, np.abs(Fblade_old[1, :, 30]), marker='^', color='b', label='Old')
ax.set_xlabel('k')
ax.set_ylabel('$|F^z_{blade}|$')
ax.legend()
plt.show()

