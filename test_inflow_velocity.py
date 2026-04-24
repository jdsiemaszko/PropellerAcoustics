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

hanson = HansonModel(
    radius_m=r_outer,
    axis= np.array([0.0, 0.0, 1.0]), # z-direction propeller...
    origin= np.array([0.0, 0.0, 0.0]), # ... at z=0
    B=2,
    nb=1,
    Omega_rads= 8000 / 60 * 2 * np.pi
)


# pin.plotDownwashInRotorPlane()
# plt.show()

# pin.plotStrutLoading3D()
# plt.show()




ind_theta = 6       # -60 to 60 in 10
ind_phi = 9          # 0 to 350 in 10
datadir = './Experimental/dataverse_files'
casefile = f'ISAE_2_D{int(1000*0.02)}_L{int(1000*0.02)}'
B=2
RPM = 8000
def load_h5(filename):
    return h5py.File(filename, "r")

with load_h5(f"{datadir}/{casefile}_autopower.h5") as f:
    g = f[casefile]
    freq = np.array(g["frequency_Hz"])
    ap = g["Autopower"]

    phi_exp = np.array(g["phi_deg"])[0][ind_phi] # azimuth
    theta_exp = np.array(g["theta_deg"])[0][ind_theta] # polar
    radius = g["radius_m"][0][0] # float

    BPF = B * RPM / 60
    data = ap[f"Autopower_RPM_{RPM}_Pa2"][:, ind_theta, ind_phi] # (freq, polar, azimuth), (aziuth=0 = > beam axis, azimuth=9 => 90 deg)
    
theta = 90 - theta_exp
phi = 180 - phi_exp
print(f'Theta_exp = {theta_exp} deg, Phi_exp = {phi_exp} deg')
print(f'Theta = {theta} deg, Phi = {phi} deg')
x_cart = np.array([
    radius * np.cos(np.deg2rad(phi)) * np.sin(np.deg2rad(theta)),
    radius * np.sin(np.deg2rad(phi)) * np.sin(np.deg2rad(theta)),
    radius * np.cos(np.deg2rad(theta)),
]).reshape((3, 1))


fig, ax = plt.subplots(figsize=(7, 4))
velocity_ref = pin.Ui
for inflow_factor in [0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]:
    print(f'consider inflow scaling factor of {inflow_factor}')
    pin.Ui = velocity_ref * inflow_factor

    vmag = np.max(np.abs(pin.Ui))
    Fbeam = pin.getStrutLoadingHarmonics()
    p_beam, _ = hanson.getPressureStator(x_cart, ms * B, Fstator=Fbeam)

    ax.plot(ms, p_to_SPL(p_beam)[0], label=f"$|v|={vmag:.2f}$", marker='^')

ax.legend(ncol=2)
ax.set_xlabel("$f^+ = f/B/\Omega$ (Hz)")
ax.set_ylabel("SPL (dB)")
ax.set_xscale('log')

ax.grid(visible=True, which='major', color='k', linestyle='-')
ax.grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.5)
ax.set_title(f'Theta = {theta} deg, Phi = {phi} deg')
plt.xlim(0.1, 100)
plt.ylim(0, 70)
plt.tight_layout()
plt.show()


