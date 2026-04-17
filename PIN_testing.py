from PotentialInteraction.PIN import PotentialInteraction
from DataPost.Vella2026 import dT, dr, dQ
import numpy as np
import matplotlib.pyplot as plt

NRADIALSEGMENTS = 20
NHARMONICS = 40
pin = PotentialInteraction(
    twist_rad= np.deg2rad(10) * np.ones(NRADIALSEGMENTS + 1),
    chord_m = 0.025 * np.ones(NRADIALSEGMENTS + 1),
    radius_m=np.linspace(0.016, 0.1, NRADIALSEGMENTS + 1),
    # Uz0_mps=U_flow,
    Tprime_Npm=dT / dr,
    Qprime_Npm=dQ / dr,
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
