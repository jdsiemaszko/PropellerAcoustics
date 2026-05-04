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
    numerics={'Nphi': 720, 'Nthetab': 36*2}
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

beam_l = BeamLoadings(
    twist_rad= np.deg2rad(10) * np.ones(NRADIALSEGMENTS),
    chord_m = 0.025 * np.ones(NRADIALSEGMENTS),
    radius_m=r_outer,
    Uz0_mps=np.sqrt(Fz/  4 / np.pi / pin.rho / pin.seg_radius),
    # Uz0_mps = np.zeros_like(Fz), # no inflow!
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

hanson = HansonModel(
    radius_m=r_outer,
    axis= np.array([0.0, 0.0, 1.0]), # z-direction propeller...
    origin= np.array([0.0, 0.0, 0.0]), # ... at z=0
    B=2,
    nb=1,
    Omega_rads= 8000 / 60 * 2 * np.pi
)


pin.plotDownwashInRotorPlane()
plt.show()

pin.plotStrutLoading3D()
plt.show()


Fblade = pin.getBladeLoadingHarmonics()
Fblade_old = blade_l.getBladeLoadingHarmonics()

Fbeam = pin.getStrutLoadingHarmonics()
Fbeam_old = beam_l.getBeamLoadingHarmonics(BLH=Fblade_old)
Fbeam_old_steady = beam_l.getBeamLoadingHarmonics(BLH=None)



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


p_rotor_loading, _ = hanson.getPressureRotor(x_cart, ms, Fblade=Fblade)
p_rotor_thickness, _ = hanson.getThicknessNoiseRotor(x_cart, ms, chord=0.025 * np.ones(NRADIALSEGMENTS-1), thickness_to_chord=0.12 * np.ones(NRADIALSEGMENTS-1))
p_beam, _ = hanson.getPressureStator(x_cart, ms * B, Fstator=Fbeam)



fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(ms, p_to_SPL(p_rotor_loading)[0], label=f"Blade Loading", color='r', marker='^')
ax.plot(ms, p_to_SPL(p_rotor_thickness)[0], label=f"Blade Thickness", color='b', marker='^')
ax.plot(ms, p_to_SPL(p_rotor_thickness+p_rotor_loading)[0], label=f"Rotor Total", color='y', marker='^')
ax.plot(ms, p_to_SPL(p_beam)[0], label=f"Beam Loading", color='m', marker='^')
ax.plot(ms, p_to_SPL(p_beam + p_rotor_thickness + p_rotor_loading)[0], label=f"Total", color='k', marker='^')
ax.plot(freq[0] / BPF, spl_from_autopower(data), label=f"Experimental", color='k', alpha=0.75)
fig, ax = plot_BPF_peaks(fig, ax, freq[0] / BPF, spl_from_autopower(data), N0=1, N1= 25, range=0.01, 
                         plot_kwargs={
                             'color':'k',
                             'linestyle':'dashed',
                             'alpha':0.75
                         })

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



k = pin.k

fig, ax = plt.subplots()

ax.plot(k, np.abs(Fblade[1, :, 30]), marker='s', color='r', label='PIN')
ax.plot(k, np.abs(Fblade_old[1, :, 30]), marker='^', color='b', label='Old')

ax.plot(k, np.abs(Fblade[2, :, 30]), marker='s', color='r', linestyle='dashed')
ax.plot(k, np.abs(Fblade_old[2, :, 30]), marker='^', color='b', linestyle='dashed')
ax.set_xlabel('k')
ax.set_ylabel('$|F^z_{blade}|$')
ax.legend()
ax.grid()
plt.title('Blade Loadings')
plt.show()

fig, ax = plt.subplots()

ax.plot(k, np.rad2deg(np.angle(Fblade[1, :, 30])), marker='s', color='r', label='PIN')
ax.plot(k, np.rad2deg(np.angle(Fblade_old[1, :, 30])), marker='^', color='b', label='Old')

ax.plot(k, np.rad2deg(np.angle(Fblade[2, :, 30])), marker='s', color='r', linestyle='dashed')
ax.plot(k, np.rad2deg(np.angle(Fblade_old[2, :, 30])), marker='^', color='b', linestyle='dashed')
ax.set_xlabel('k')
ax.set_ylabel('$Arg(F^z_{blade})$ [deg]')
ax.legend()
ax.grid()
plt.title('Blade Loadings')
plt.show()


fig, ax = plt.subplots()

ax.plot(k, np.abs(Fbeam[1, :, 30]), marker='s', color='r', label='PIN')
ax.plot(k, np.abs(Fbeam_old[1, :, 30]), marker='^', color='b', label='Old')

ax.plot(k, np.abs(Fbeam[2, :, 30]), marker='s', color='r', linestyle='dashed')
ax.plot(k, np.abs(Fbeam_old[2, :, 30]), marker='^', color='b', linestyle='dashed')
ax.set_xlabel('k')
ax.set_ylabel('$|F^z_{beam}|$ [N/m]')
ax.legend()
ax.grid()
plt.title('Beam Loadings')
plt.show()

fig, ax = plt.subplots()

ax.plot(k, np.rad2deg(np.angle(Fbeam[1, :, 30])), marker='s', color='r', label='PIN')
ax.plot(k, np.rad2deg(np.angle(Fbeam_old[1, :, 30])), marker='^', color='b', label='Old')

ax.plot(k, np.rad2deg(np.angle(Fbeam[2, :, 30])), marker='s', color='r', linestyle='dashed')
ax.plot(k, np.rad2deg(np.angle(Fbeam_old[2, :, 30])), marker='^', color='b', linestyle='dashed')

ax.set_xlabel('k')
ax.set_ylabel('$Arg(F^z_{beam})$ [deg]')
ax.legend()
ax.grid()
plt.title('Beam Loadings')
plt.show()



