from PotentialInteraction.ConformalPIN import HypotrochoidalPIN
import numpy as np
from Constants.helpers import read_force_file, p_to_SPL, spl_from_autopower, plot_BPF_peaks
import matplotlib.pyplot as plt
from Constants.data_assim import getGojonData

r_inner, Fz, Fphi = read_force_file('./Data/Zamponi2026/FS_ISAE_2_8000.txt')

dr = np.diff(r_inner)[0]
r_outer = np.hstack([r_inner-dr/2, r_inner[-1]+dr/2])
NRADIALSEGMENTS = np.shape(r_outer)[0]
NHARMONICS = 40
ms = np.arange(1, 16, 1)

pin = HypotrochoidalPIN(
    Nsides=3, theta0=np.deg2rad(0), rho_corner=0.5,
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
    numerics={'Nphi': 180, 'Nthetab': 360}
)

# fig, ax = pin.plotZ()
fig, ax = pin.plotMap(center = 0.012 + 0.00j, radii = np.linspace(0.02/100, 0.02 * 10, 200))
plt.show()

# pin.plotDownwashInRotorPlane()
# plt.show()

# pin.plotStrutLoading3D()
# plt.show()

# inverse transform OKAY
# fig, ax = pin.plotZ()
# LS = [0.01]
# for L in LS:
#     z = pin.phi[None, :] * pin.seg_radius[:, None] + 1j * L
#     zeta = pin.getZeta(z)
#     ax.plot(np.real(z[30, :]), np.imag(z[30, :]), label=f'L={L:.4f}')
#     ax.legend()
# plt.show()

# fig, ax = pin.plotZeta()
# for L in LS:
#     z = pin.phi[None, :] * pin.seg_radius[:, None] + 1j * L
#     zeta = pin.getZeta(z)
#     ax.plot(np.real(zeta[30, :]), np.imag(zeta[30, :]), label=f'L={L:.4f}')
#     ax.legend()
# plt.show()

