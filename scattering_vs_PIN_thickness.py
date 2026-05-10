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
from Constants.data_assim import getGojonData

SUFFIX = '_D360_HR'
# SUFFIX = '_HIGHRES'

ind_theta = 6       # -60 to 60 in 10
ind_phi = 9          # 0 to 350 in 10
datadir = './Experimental/dataverse_files'
data, BPF, freq, x_cart_data, theta_data, phi_data, theta_exp, phi_exp, casefile = getGojonData(datadir, D_bras, g, shape='D', B=2, RPM=8000)
data = data[:, ind_theta, ind_phi]
x_cart = x_cart_data[:, ind_theta, ind_phi].reshape((3, 1))
x_cart = x_cart_data[:, ind_theta, ind_phi].reshape((3, 1))
theta = theta_data[ind_theta]
phi = phi_data[ind_phi]

Nr = len(r_inner)
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

ptmB_model_rotor = han.getThicknessNoiseRotor(x_cart, ms, c * np.ones_like(r_inner), 0.082 * np.ones_like(r_inner))[0][0] # NACA0012


# BL  =  beam_l.getBeamLoadingHarmonics(BLH=BLH)
PIN._numerics['include_vortex_sources'] = False
PIN._numerics['include_thickness_sources'] = True
             
BL = PIN.getStrutLoadingHarmonics()
pmB_model_beam = han.getPressureStator(x_cart, ms*B, BL)[0][0]


pmB_model_rotor_total = pLmB_model_rotor + ptmB_model_rotor
pmB_model_rotor_loading = pLmB_model_rotor
pmB_model_total = pLmB_model_rotor + ptmB_model_rotor + pmB_model_beam # assuming coherent
# pmB_model_total = np.sqrt(np.abs(pmB_model_rotor_total)**2 + np.abs(pmB_model_beam)**2) # assuming incoherent


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

sourceArray.numerics['CompactnessCorrection'] = True
p_scattered_thickness_corr = sourceArray.getThicknessPressureScattered(x_cart, ms, G=G_arr)[0]
p_direct_thickness_corr = sourceArray.getThicknessPressureDirect(x_cart, ms)[0]

sourceArray.numerics['CompactnessCorrection'] = False
p_scattered_thickness_uncorr = sourceArray.getThicknessPressureScattered(x_cart, ms, G=G_arr)[0]
p_direct_thickness_uncorr = sourceArray.getThicknessPressureDirect(x_cart, ms)[0]

# np.save(f'./Data/current/NACA0012_rotor/p_s_spectrum_thickness_{MODE}_{ind_theta}_{ind_phi}.npy', p_scattered_thickness)
# p_scattered_thickness = np.load(f'./Data/current/NACA0012_rotor/p_s_spectrum_thickness_{MODE}_{ind_theta}_{ind_phi}.npy')

# SPLS PIN MODEL
SPL_rotor_thickness = p_to_SPL(ptmB_model_rotor)
SPL_beam = p_to_SPL(pmB_model_beam)

# SPLS TOTAL SCATTERING
SPL_direct_thickness = p_to_SPL(p_direct_thickness_corr)
SPL_scattered_thickness = p_to_SPL(p_scattered_thickness_corr)

# SPLS TOTAL SCATTERING
SPL_direct_thickness_unc = p_to_SPL(p_direct_thickness_uncorr)
SPL_scattered_thickness_unc = p_to_SPL(p_scattered_thickness_uncorr)

# SPL_total = p_to_SPL(p_rms_total) # same computation

fig, ax = plt.subplots(figsize=(7, 4))

# ax.plot(ms, SPL_rotor_S, label=f"Steady Loading Noise (Hanson)", color='r', marker='^')
# ax.plot(ms, SPL_rotor_US, label=f"Unsteady Loading Noise (Hanson)", color='g', marker='^')





ax.plot(ms, SPL_rotor_thickness, label=f"Thickness Noise (Hanson)", color='b', marker='^')
ax.plot(ms, SPL_beam, label=f"Beam Loading due to Blade Thickness", color='r', marker='^')
# ax.plot(freq[0] / BPF, spl_from_autopower(data), label=f"Experimental, total", color='k', alpha=0.75)
# fig, ax = plot_BPF_peaks(fig, ax, freq[0] / BPF, spl_from_autopower(data), N0=1, N1= 25, range=0.01, 
#                          plot_kwargs={
#                              'color':'k',
#                              'linestyle':'dashed',
#                              'alpha':0.75
#                          })

# ax.plot(ms, SPL_direct_thickness, label=f"Thickness Noise (SM, corrected)", color='b', marker='s', linestyle='dashed')
# ax.plot(ms, SPL_scattered_thickness, label=f"Scattered Thickness Noise, corrected", color='r', marker='s', linestyle='dashed')
# ax.plot(ms, SPL_direct_thickness_unc, label=f"Thickness Noise (SM, uncorrected)", color='b', marker='s', linestyle='dotted')
# ax.plot(ms, SPL_scattered_thickness_unc, label=f"Scattered Thickness Noise (uncorrected)", color='r', marker='s', linestyle='dotted')
# ax.plot(ms, SPL_direct_thickness, label=f"Thickness Noise (SM, corrected)", color='b', marker='s', linestyle='dashed')
# ax.plot(ms, SPL_scattered_thickness, label=f"Scattered Thickness Noise, corrected", color='r', marker='s', linestyle='dashed')
ax.plot(ms, SPL_direct_thickness_unc, label=f"Thickness Noise (SM)", color='b', marker='s', linestyle='dotted')
ax.plot(ms, SPL_scattered_thickness_unc, label=f"Scattered Thickness Noise", color='r', marker='s', linestyle='dotted')
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