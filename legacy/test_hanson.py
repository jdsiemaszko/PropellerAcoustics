import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from Hanson.far_field import HansonModel
from Hanson.near_field import NearFieldHansonModel

NSEG = 20 # number of radial prop segments
kmaxx = 64
ROBS = 1

rad = np.linspace(0.016, 0.1, NSEG+1, endpoint=True)
loadings = np.zeros((kmaxx, NSEG), dtype=np.complex128)
loadings[1:5, :] = 1.0 + 0j # arbitrary loading

hm = HansonModel(twist_rad = np.deg2rad(10 * np.ones(NSEG+1)), chord_m = 0.025 * np.ones(NSEG+1),
                 radius_m=rad,
                 loadings_Npm=loadings,
                axis=np.array([0, 0, 1]), origin=np.array([0, 0, 0]), radial=np.array([1, 0, 0]),
                  B=2, Omega_rads=8000/60 * 2 * np.pi, rho_kgm3=1.2, c_mps= 340., nb = 1)

hmnf = NearFieldHansonModel(twist_rad = np.deg2rad(10 * np.ones(NSEG+1)), chord_m = 0.025 * np.ones(NSEG+1),
                 radius_m=rad,
                 loadings_Npm=loadings,
                axis=np.array([0, 0, 1]), origin=np.array([0, 0, 0]), radial=np.array([1, 0, 0]),
                  B=2, Omega_rads=8000/60 * 2 * np.pi, rho_kgm3=1.2, c_mps= 340., nb = 1,
                  N_points = 64)

fig = plt.figure(figsize=(14, 7))

# Left subplot
ax1 = fig.add_subplot(121, projection="3d")
hm.plot3Ddirectivity(
    fig=fig,
    ax=ax1,
    m=np.array([1.0]),
    R=ROBS,
    Nphi=36*2,
    Ntheta=18*2,
    valmin=40,
    valmax=65,
    title='far-field'
)

# Right subplot
ax2 = fig.add_subplot(122, projection="3d")
hmnf.plot3Ddirectivity(
    fig=fig,
    ax=ax2,
    m=np.array([1.0]),
    R=ROBS,
    Nphi=36*2,
    Ntheta=18*2,
    valmin=40,
    valmax=65,
    title='near-field'

)

plt.tight_layout()
plt.show()