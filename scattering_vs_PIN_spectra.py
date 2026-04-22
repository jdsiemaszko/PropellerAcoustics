from scattering_vs_PIN import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

SUFFIX = '_D360_HR'

ind_theta = 7       # -60 to 60 in 10
ind_phi = 0          # 0 to 350 in 10
datadir = './Experimental/dataverse_files'
casefile = f'ISAE_2_D{int(1000*D_bras)}_L{int(1000*g)}'

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

Nr = len(r_inner)



ms = np.arange(1, 11, 1) # harmonics to extract



pSmB_model_rotor = han.getPressureRotor(x_cart, ms, 
                                    #    blade_l.getBladeLoadingHarmonics()
                                    BLH_S
                                       )[0][0]

pUSmB_model_rotor = han.getPressureRotor(x_cart, ms, 
                                    #    blade_l.getBladeLoadingHarmonics()
                                    BLH_US
                                       )[0][0]

ptmB_model_rotor = han.getThicknessNoiseRotor(x_cart, ms, c * np.ones_like(r_inner), 0.0809 * np.ones_like(r_inner))[0][0] # NACA0012
# BL  =  beam_l.getBeamLoadingHarmonics(BLH = BLH)
BL = PIN.getStrutLoadingHarmonics()
pmB_model_beam = han.getPressureStator(x_cart, ms*B, BL)[0][0]

pmB_model_rotor_total = pSmB_model_rotor + pUSmB_model_rotor + ptmB_model_rotor
# pmB_model_total = pSmB_model_rotor + pUSmB_model_rotor + ptmB_model_rotor + pmB_model_beam # assuming coherent
pmB_model_total = np.sqrt(np.abs(pmB_model_rotor_total)**2 + np.abs(pmB_model_beam)**2) # assuming incoherent




# SUFFIX = ''

# save gradients on the surface (run once per m)

# for index, sm in enumerate(sourceArray.children):
#     #TODO: reuse this for directivities
#     print(f'pre-computing surface gradients: {index+1}')
#     source_positions = sm.dipole_positions
#     gradG_surface = sm.green.getGreenGradAtSurface(source_positions, ms*B * Omega / c0) # shape (3, Nm, Nz, Ny)
#     np.save(f'./Data/current/NACA0012_rotor/gradG_surface_sm_{index}_{MODE}{SUFFIX}.npy', gradG_surface)

# save gradients in the far-field (run once per observer and m)
for index, sm in enumerate(sourceArray.children):

    gradG_surface = np.load(f'./Data/current/NACA0012_rotor/gradG_surface_sm_{index}_{MODE}{SUFFIX}.npy') # shape (3, Nm, Nz, Ny)
    print(f'pre-computing far-field gradients {index+1}')

    gradG = sm.getScatteringGreenGradient(x_cart, ms*B * Omega / c0, gradG_surface) # shape (3, Nm, Nx, Ny)
    np.save(f'./Data/current/NACA0012_rotor/gradG_sm_{index}_{MODE}_{ind_theta}_{ind_phi}.npy', gradG)



# extract and rearrange

gradG_arr = np.zeros((Nr, 3, ms.shape[0], x_cart.shape[1], NDIPOLES), dtype=np.complex128)
for index, sm in enumerate(sourceArray.children):
    gradG_arr[index] = np.load(f'./Data/current/NACA0012_rotor/gradG_sm_{index}_{MODE}_{ind_theta}_{ind_phi}.npy')


p_scattered = sourceArray.getScatteredPressure(x_cart, ms, gradG=gradG_arr)[0]
np.save(f'./Data/current/NACA0012_rotor/p_s_spectrum_{MODE}_{ind_theta}_{ind_phi}.npy', p_scattered)

p_scattered = np.load(f'./Data/current/NACA0012_rotor/p_s_spectrum_{MODE}_{ind_theta}_{ind_phi}.npy')
# p_scattered_US = np.load(f'./Data/current/NACA0012_rotor/p_s_spectrum_UNSTEADY_{MODE}_{ind_theta}_{ind_phi}.npy')

SPL_rotor_S = p_to_SPL(pSmB_model_rotor)
SPL_rotor_US = p_to_SPL(pUSmB_model_rotor)
SPL_rotor_total = p_to_SPL(pmB_model_rotor_total)
SPL_rotor_thickness = p_to_SPL(ptmB_model_rotor)
SPL_beam = p_to_SPL(pmB_model_beam)

SPL_scattered = p_to_SPL(p_scattered)
# SPL_scattered_US = p_to_SPL(p_scattered_US)




SPL_total = p_to_SPL(pmB_model_total)
# SPL_total = p_to_SPL(p_rms_total) # same computation

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(freq[0] / BPF, spl_from_autopower(data), label=f"Experimental, total", color='k', alpha=0.75)
# fig, ax = plot_BPF_peaks(fig, ax, freq[0] / BPF, spl_from_autopower(data), N0=1, N1= 25, range=0.01, 
#                          plot_kwargs={
#                              'color':'k',
#                              'linestyle':'dashed',
#                              'alpha':0.75
#                          })
# ax.plot(ms, SPL_rotor_S, label=f"Model (rotor, steady loading)", color='r', marker='^')
# ax.plot(ms, SPL_rotor_US, label=f"Model (rotor, unsteady loading)", color='g', marker='^')
# ax.plot(ms, SPL_rotor_thickness, label=f"Model (rotor, thickness)", color='b', marker='+')
# ax.plot(ms, SPL_rotor_total, label=f"Model (rotor, total)", color='y', marker='*', linestyle='dashed')
# ax.plot(ms, SPL_beam, label=f"Model (beam, loading)", color='m', marker='o')
# ax.plot(ms, SPL_total, label=f"Model (total)", color='k', marker='s', linestyle='dashed')
ax.plot(ms, SPL_beam, label=f"Beam PIN", color='r', marker='o')
ax.plot(ms, SPL_scattered, label=f"Blade Scattering", color='b', marker='s')
# ax.plot(ms, SPL_scattered_US, label=f"Blade Scattering (unsteady)", color='g', marker='^')

import pandas as pd
filename1 = './Data/Vella2026/SPL_theta0phi90.csv'
data1 = pd.read_csv(filename1)
spl_vella = data1[' y']
fs_vella = data1['x']

filename2 = './Data/Zamponi2026/SPL_theta0phi90.csv'
data2 = pd.read_csv(filename2)
spl_zamponi = data2[' y']
fs_zamponi = data2['x']

ax.plot(fs_vella/BPF, spl_vella, label=f"Vella et al. (2026)", color='m', marker='^')
ax.plot(np.round(fs_zamponi), spl_zamponi, label=f"Zamponi et al.(2026)", color='c', marker='o')
ax.plot(np.round(fs_zamponi), spl_zamponi + 10 * np.log10(2), label=f"Zamponi et al.(2026) (corrected)", color='g', marker='p')


ax.legend()
ax.set_xlabel("$f^+ = f/B/\Omega$ (Hz)")
ax.set_ylabel("SPL (dB)")
ax.set_xscale('log')
# ax.set_yscale('log')

ax.grid(visible=True, which='major', color='k', linestyle='-')
ax.grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.5)
ax.set_title(f'Theta = {theta} deg, Phi = {phi} deg')
plt.xlim(0.1, 100)
plt.ylim(0, 70)
plt.tight_layout()
plt.show()
