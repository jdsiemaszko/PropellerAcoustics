from PotentialInteraction.PIN import PotentialInteraction
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PotentialInteraction.beam_to_blade import NACA0012_T10_PIN, BladeLoadings
from PotentialInteraction.blade_to_beam import BeamLoadings
from Constants.helpers import read_force_file, p_to_SPL, spl_from_autopower, plot_BPF_peaks
from Constants.data_assim import getGojonData
from Hanson.far_field import HansonModel
import h5py

r_inner, Fz, Fphi = read_force_file('./Data/Zamponi2026/FS_ISAE_2_8000.txt')

dr = np.diff(r_inner)[0]
r_outer = np.hstack([r_inner-dr/2, r_inner[-1]+dr/2])
NRADIALSEGMENTS = np.shape(r_outer)[0]
NHARMONICS = 40
ms = np.arange(1, 16, 1)

B = 2
RPM = 8000
Omega = RPM / 60 * 2 * np.pi
D = 0.02
L = 0.02

pin = PotentialInteraction(
    twist_rad= np.deg2rad(10) * np.ones(NRADIALSEGMENTS),
    chord_m = 0.025 * np.ones(NRADIALSEGMENTS),
    radius_m=r_outer,
    t_c = np.ones_like(r_outer) * 0.12,
    # Uz0_mps=U_flow,
    Fzprime_Npm=Fz,
    Fphiprime_Npm=Fphi,
    B=B,
    Dcylinder_m=D,
    Lcylinder_m=L,
    Omega_rads=Omega,
    rho_kgm3=1.2,
    c_mps=340.0,
    kmax=NHARMONICS,
    nb=1,
    numerics={'Nphi': 720, 'Nthetab': 36*2, 'include_vortex_sources':True, 'include_thickness_sources':False}
)

hanson = HansonModel(
    radius_m=r_outer,
    axis= np.array([0.0, 0.0, 1.0]), # z-direction propeller...
    origin= np.array([0.0, 0.0, 0.0]), # ... at z=0
    B=B,
    nb=1,
    Omega_rads= RPM
)

# pin.plotDownwashInRotorPlane()
# plt.show()

for m in [1]:
    pin.plotSurfacePressureContour(m = m * B)
plt.show()


pin.plotStrutLoading2D(0.8)
plt.show()

pin._numerics['include_vortex_sources'] = True
pin._numerics['include_thickness_sources'] = True
pin.plotStrutLoadingHarmonics2D(0.8)
pin._numerics['include_vortex_sources'] = True
pin._numerics['include_thickness_sources'] = False
pin.plotStrutLoadingHarmonics2D(0.8)
plt.show()

pin._numerics['include_vortex_sources'] = True
pin._numerics['include_thickness_sources'] = True
Fbeam_total = pin.getStrutLoadingHarmonics()

pin._numerics['include_vortex_sources'] = True
pin._numerics['include_thickness_sources'] = False
Fbeam_loading = pin.getStrutLoadingHarmonics()

pin._numerics['include_vortex_sources'] = False
pin._numerics['include_thickness_sources'] = True
Fbeam_thickness = pin.getStrutLoadingHarmonics()


ind_theta = 6       # -60 to 60 in 10
ind_phi = 9          # 0 to 350 in 10
datadir = './Experimental/dataverse_files'
data, BPF, freq, x_cart_data, theta_data, phi_data, theta_exp, phi_exp, casefile = getGojonData(datadir, D, L, shape='D', B=B, RPM=RPM)

x_cart = x_cart_data[:, ind_theta, ind_phi].reshape((3, 1))
data = data[:, ind_theta, ind_phi]

p_beam_tot, _ = hanson.getPressureStator(x_cart, ms * B, Fstator=Fbeam_total)
p_beam_th, _ = hanson.getPressureStator(x_cart, ms * B, Fstator=Fbeam_thickness)
p_beam_l, _ = hanson.getPressureStator(x_cart, ms * B, Fstator=Fbeam_loading)



