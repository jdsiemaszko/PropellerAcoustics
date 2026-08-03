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

sourceArray.numerics['CompactnessCorrection'] = True
# sourceArray.numerics['CompactnessCorrection'] = False


ind_theta = 6       # -60 to 60 in 10
ind_phi = 9          # 0 to 350 in 10
datadir = './Experimental/dataverse_files'

data, BPF, freq, x_cart_data, theta_data, phi_data, theta_exp, phi_exp, casefile = getGojonData(datadir, D_bras, g, shape='D', B=2, RPM=8000)

x_cart = x_cart_data[:, ind_theta, ind_phi].reshape((3, 1))
data = data[:, ind_theta, ind_phi]
theta = theta_data[ind_theta]
phi = phi_data[ind_phi]

Nr = len(r_inner)
ms = np.arange(1, 11, 1) # harmonics to extract



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

fig, ax = plt.subplots(figsize=(7, 4))

for bol, linestyle in zip([True, False], ['dashed', 'solid']):

    sourceArray.numerics['CompactnessCorrection'] = bol

    # todo: compute for both with and without corrections
    p_scattered = sourceArray.getScatteredPressure(x_cart, ms, gradG=gradG_arr)[0]
    p_direct = sourceArray.getDirectPressure(x_cart, ms)[0]
    p_scattered_thickness = sourceArray.getThicknessPressureScattered(x_cart, ms, G=G_arr)[0]
    p_direct_thickness = sourceArray.getThicknessPressureDirect(x_cart, ms)[0]

    ax.plot(ms, p_to_SPL(p_direct), label=f"blade loading, direct" if bol else None, color='r', marker='s', linestyle=linestyle)
    ax.plot(ms, p_to_SPL(p_direct_thickness), label=f"blade thickness, direct" if bol else None, color='b', marker='s', linestyle=linestyle)
    ax.plot(ms, p_to_SPL(p_scattered), label=f"blade loading, scattered" if bol else None, color='m', marker='s', linestyle=linestyle)
    ax.plot(ms, p_to_SPL(p_scattered_thickness), label=f"blade thickness, scattered" if bol else None, color='c', marker='s', linestyle=linestyle)

# ax.plot(ms, SPL_total_scattering_minus_scattered_thickness, label=f"Direct+Scattering (Minus Thickness)", color='g', marker='s', linestyle='dashed')


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
