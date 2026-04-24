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

pin = HypotrochoidalPIN(
    Nsides=3, theta0=np.deg2rad(30), rho_corner=0.65,
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
    nb=1,
    numerics={'Nphi': 180, 'Nthetab': 36}
)

# pin.plotCrossSection()
# pin.plotMap()
# plt.show()

# pin.plotDownwashInRotorPlane()
# plt.show()

# inverse transform OKAY
fig, ax = pin.plotZ()

for L in [0.02, 0.01, 0.0075, 0.005]:
    z = pin.phi[None, :] * pin.seg_radius[:, None] + 1j * L
    zeta = pin.getZeta(z)
    ax.plot(np.real(z[30, :]), np.imag(z[30, :]), label=f'L={L:.4f}')
    ax.legend()
plt.show()

fig, ax = pin.plotZeta()
for L in [0.02, 0.01, 0.0075, 0.005]:
    z = pin.phi[None, :] * pin.seg_radius[:, None] + 1j * L
    zeta = pin.getZeta(z)
    ax.plot(np.real(zeta[30, :]), np.imag(zeta[30, :]), label=f'L={L:.4f}')
    ax.legend()
plt.show()
