"""
combine results from thickness and loading noise scattered on the beam
compare to direct beam noise evaluation
suggest model for thickness noise as PIN -> sources/sinks :)
the last point seems significant at 2-3 x BPF:
-- at 1xBPF direct noise dominant?
-- at 2-3xBPF thickness noise/ scattering of thickness noise dominant?
-- at 4+xBPF scattering of loading noise dominant
"""

# open G and gradG on surface. Compute resulting green terms, compute loading and thickness noise (direct + scattered)
# open exp data, evaluate all at exp observer positions
# compare spectra at points, directivities, ...
# profit?

from scattering_vs_PIN import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


ind_theta = 2       # -60 to 60 in 10
ind_phi = 9          # 0 to 350 in 10
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

Nr = len(r0)



ms = np.arange(1, 11, 1) # harmonics to extract



# pSmB_model_rotor = han.getPressureRotor(x_cart, ms, 
#                                     #    blade_l.getBladeLoadingHarmonics()
#                                     BLH_S
#                                        )[0][0]

# pUSmB_model_rotor = han.getPressureRotor(x_cart, ms, 
#                                     #    blade_l.getBladeLoadingHarmonics()
#                                     BLH_US
#                                        )[0][0]
# pLmB_model_rotor = pSmB_model_rotor + pUSmB_model_rotor
# SPL_rotor_S = p_to_SPL(pSmB_model_rotor)
# SPL_rotor_US = p_to_SPL(pUSmB_model_rotor)
pLmB_model_rotor = han.getPressureRotor(x_cart, ms, 
                                    #    blade_l.getBladeLoadingHarmonics()
                                    BLH
                                       )[0][0]

ptmB_model_rotor = han.getThicknessNoiseRotor(x_cart, ms, c * np.ones_like(r0), 0.122 * np.ones_like(r0))[0][0] # NACA0012
BL  =  beam_l.getBeamLoadingHarmonics(BLH=BLH)
pmB_model_beam = han.getPressureStator(x_cart, ms*B, BL)[0][0]


pmB_model_rotor_total = pLmB_model_rotor + ptmB_model_rotor
pmB_model_rotor_loading = pLmB_model_rotor
pmB_model_total = pLmB_model_rotor + ptmB_model_rotor + pmB_model_beam # assuming coherent
# pmB_model_total = np.sqrt(np.abs(pmB_model_rotor_total)**2 + np.abs(pmB_model_beam)**2) # assuming incoherent




SUFFIX = '_HIGHRES'
# SUFFIX = '

# -------------------------------- SCATTERED LOADING NOISE ------------------------------------------
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
p_direct = sourceArray.getDirectPressure(x_cart, ms)[0]

# np.save(f'./Data/current/NACA0012_rotor/p_s_spectrum_{MODE}_{ind_theta}_{ind_phi}.npy', p_scattered)
# p_scattered = np.load(f'./Data/current/NACA0012_rotor/p_s_spectrum_{MODE}_{ind_theta}_{ind_phi}.npy')


# -------------------------------- SCATTERED Thickness NOISE ------------------------------------------

# save gradients in the far-field (run once per observer and m)
for index, sm in enumerate(sourceArray.children):

    G_surface = np.load(f'./Data/current/NACA0012_rotor/G_surface_sm_{index}_{MODE}{SUFFIX}.npy') # shape (Nm, Nz, Ny)
    print(f'pre-computing far-field gradients {index+1}')

    G = sm.getScatteringGreen(x_cart, ms*B * Omega / c0, G_surface) # shape (Nm, Nx, Ny)
    np.save(f'./Data/current/NACA0012_rotor/G_sm_{index}_{MODE}_{ind_theta}_{ind_phi}.npy', G)


G_arr = np.zeros((Nr, ms.shape[0], x_cart.shape[1], NDIPOLES), dtype=np.complex128)
for index, sm in enumerate(sourceArray.children):
    G_arr[index] = np.load(f'./Data/current/NACA0012_rotor/G_sm_{index}_{MODE}_{ind_theta}_{ind_phi}.npy')


p_scattered_thickness = sourceArray.getThicknessPressureScattered(x_cart, ms, G=G_arr)[0]
p_direct_thickness = sourceArray.getThicknessPressureDirect(x_cart, ms)[0]

# np.save(f'./Data/current/NACA0012_rotor/p_s_spectrum_thickness_{MODE}_{ind_theta}_{ind_phi}.npy', p_scattered_thickness)
# p_scattered_thickness = np.load(f'./Data/current/NACA0012_rotor/p_s_spectrum_thickness_{MODE}_{ind_theta}_{ind_phi}.npy')

p_total_scattering = p_direct + p_scattered + p_direct_thickness + p_scattered_thickness
p_rotor_total = p_direct + p_direct_thickness
p_total_minus_scattered_thickness = p_total_scattering - p_scattered_thickness

# SPLS PIN MODEL
SPL_rotor_loading = p_to_SPL(pmB_model_rotor_loading)

SPL_rotor_total = p_to_SPL(pmB_model_rotor_total)
SPL_rotor_thickness = p_to_SPL(ptmB_model_rotor)
SPL_beam = p_to_SPL(pmB_model_beam)
SPL_total_PIN = p_to_SPL(pmB_model_total)

# SPLS TOTAL SCATTERING
SPL_direct  = p_to_SPL(p_direct)
SPL_scattered = p_to_SPL(p_scattered)
SPL_direct_thickness = p_to_SPL(p_direct_thickness)
SPL_scattered_thickness = p_to_SPL(p_scattered_thickness)
SPL_SM_rotor_total = p_to_SPL(p_rotor_total)
SPL_total_scattering = p_to_SPL(p_total_scattering)
SPL_total_scattering_minus_scattered_thickness = p_to_SPL(p_total_minus_scattered_thickness)

# SPL_total = p_to_SPL(p_rms_total) # same computation

fig, ax = plt.subplots(figsize=(7, 4))

# ax.plot(ms, SPL_rotor_S, label=f"Steady Loading Noise (Hanson)", color='r', marker='^')
# ax.plot(ms, SPL_rotor_US, label=f"Unsteady Loading Noise (Hanson)", color='g', marker='^')





ax.plot(ms, SPL_rotor_loading, label=f"Loading Noise (Hanson)", color='r', marker='^')
ax.plot(ms, SPL_rotor_thickness, label=f"Thickness Noise (Hanson)", color='b', marker='^')
ax.plot(ms, SPL_rotor_total, label=f"Rotor Total (Hanson)", color='y', marker='^')
ax.plot(ms, SPL_beam, label=f"Beam Loading (PIN)", color='m', marker='^')
ax.plot(ms, SPL_total_PIN, label=f"Hanson+PIN", color='k', marker='^')
ax.plot(freq[0] / BPF, spl_from_autopower(data), label=f"Experimental, total", color='k', alpha=0.75)
fig, ax = plot_BPF_peaks(fig, ax, freq[0] / BPF, spl_from_autopower(data), N0=1, N1= 25, range=0.01, 
                         plot_kwargs={
                             'color':'k',
                             'linestyle':'dashed',
                             'alpha':0.75
                         })

ax.plot(ms, SPL_direct, label=f"Loading Noise (SM)", color='r', marker='s', linestyle='dashed')
ax.plot(ms, SPL_direct_thickness, label=f"Thickness Noise (SM)", color='b', marker='s', linestyle='dashed')
ax.plot(ms, SPL_SM_rotor_total, label=f"Rotor Total (SM)", color='y', marker='s', linestyle='dashed')
ax.plot(ms, SPL_scattered, label=f"Scattered Loading Noise", color='m', marker='s', linestyle='dashed')
ax.plot(ms, SPL_total_scattering, label=f"Direct+Scattering", color='k', marker='s', linestyle='dashed')

ax.plot(ms, SPL_scattered_thickness, label=f"Scattered Thickness Noise", color='c', marker='s', linestyle='dashed')

ax.plot(ms, SPL_total_scattering_minus_scattered_thickness, label=f"Direct+Scattering (Minus Thickness)", color='g', marker='s', linestyle='dashed')


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
