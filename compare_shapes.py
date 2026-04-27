from PotentialInteraction.ConformalPIN import HypotrochoidalPIN
import numpy as np
from Constants.helpers import read_force_file, p_to_SPL, spl_from_autopower, plot_BPF_peaks
import matplotlib.pyplot as plt

r_inner, Fz, Fphi = read_force_file('./Data/Zamponi2026/FS_ISAE_2_8000.txt')

dr = np.diff(r_inner)[0]
r_outer = np.hstack([r_inner-dr/2, r_inner[-1]+dr/2])
NRADIALSEGMENTS = np.shape(r_outer)[0]
NHARMONICS = 40
ms = np.arange(1, 16, 1)

pin_triangle = HypotrochoidalPIN(
    Nsides=3, theta0=np.deg2rad(210), rho_corner=0.5,
    twist_rad= np.deg2rad(10) * np.ones(NRADIALSEGMENTS),
    chord_m = 0.025 * np.ones(NRADIALSEGMENTS),
    radius_m=r_outer,
    # Uz0_mps=U_flow,
    Fzprime_Npm=Fz,
    Fphiprime_Npm=Fphi,
    B=2,
    Dcylinder_m=0.01,
    Lcylinder_m=0.02,
    Omega_rads=8000/60*2*np.pi,
    rho_kgm3=1.2,
    c_mps=340.0,
    kmax=NHARMONICS,
    nb=1,
    numerics={'Nphi': 720, 'Nthetab': 36*4}
)

pin_square = HypotrochoidalPIN(
    Nsides=4, theta0=np.deg2rad(45), rho_corner=0.3,
    twist_rad= np.deg2rad(10) * np.ones(NRADIALSEGMENTS),
    chord_m = 0.025 * np.ones(NRADIALSEGMENTS),
    radius_m=r_outer,
    # Uz0_mps=U_flow,
    Fzprime_Npm=Fz,
    Fphiprime_Npm=Fphi,
    B=2,
    Dcylinder_m=0.01 * 1.163 * 0.94,
    Lcylinder_m=0.02,
    Omega_rads=8000/60*2*np.pi,
    rho_kgm3=1.2,
    c_mps=340.0,
    kmax=NHARMONICS,
    nb=1,
    numerics={'Nphi': 720, 'Nthetab': 36*4}
)

pin_circle = HypotrochoidalPIN(
    Nsides=100, theta0=np.deg2rad(0), rho_corner=0.01,
    twist_rad= np.deg2rad(10) * np.ones(NRADIALSEGMENTS),
    chord_m = 0.025 * np.ones(NRADIALSEGMENTS),
    radius_m=r_outer,
    # Uz0_mps=U_flow,
    Fzprime_Npm=Fz,
    Fphiprime_Npm=Fphi,
    B=2,
    Dcylinder_m=0.01,
    Lcylinder_m=0.02,
    Omega_rads=8000/60*2*np.pi,
    rho_kgm3=1.2,
    c_mps=340.0,
    kmax=NHARMONICS,
    nb=1,
    numerics={'Nphi': 720, 'Nthetab': 36*4}
)

fig, ax = pin_triangle.plotZ()
fig, ax = pin_square.plotZ(fig, ax)
fig, ax = pin_circle.plotZ(fig, ax)
plt.show()

pin_triangle.plotDownwashInRotorPlane()
plt.show()

pin_square.plotDownwashInRotorPlane()
plt.show()

pin_circle.plotDownwashInRotorPlane()
plt.show()