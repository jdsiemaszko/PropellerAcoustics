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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# BEGINNING OF HEADER
MODE = 'half'
from Constants.helpers import read_force_file, plot_3D_directivity, plot_3D_phase_directivity, p_to_SPL, spl_from_autopower, plot_BPF_peaks
from Constants.data_assim import getGojonData
# vary configuration
from SourceMode.Configurations_NACA0012 import m_surface

# from SourceMode.Configurations_NACA0012 import D20L20W00_D360 as sourceArray # pick configuration
# SUFFIX = '_D360_HR'

# from SourceMode.Configurations_NACA0012 import D20L20W20_D180 as sourceArray # pick configuration
# SUFFIX = '_D20L20W20_D180'

from SourceMode.Configurations_NACA0012 import D20L20W00_D180, D20L20W10_D180, D20L20W20_D180, D20L20W40_D180, D20L20W60_D180, D20L20W80_D180, D20L20W100_D180
CASES = [D20L20W00_D180, D20L20W00_D180, D20L20W10_D180, D20L20W20_D180, D20L20W40_D180, D20L20W60_D180, D20L20W80_D180, D20L20W100_D180]
SUFFIXES = ['_D180_MR','_D180_MR', '_D20L20W10_D180', '_D20L20W20_D180', '_D20L20W40_D180', '_D20L20W60_D180', '_D20L20W80_D180', '_D20L20W100_D180']
COLORS = ['k', 'r', 'b', 'g', 'm', 'c', 'y', 'tab:pink']
LABELS = [f'W={mm:.0f}mm' for mm in [0, 0, 10, 20, 40, 60, 80, 100]]
LABELS[0] += ' (incl. unsteady loading)'

ind_theta = 0      # -60 to 60 in 10
ind_phi = 7          # 0 to 350 in 10
datadir = './Experimental/dataverse_files'
# casefile = f'ISAE_2_D{int(1000*D_bras)}_L{int(1000*g)}'

data, BPF, freq, x_cart_data, theta_data, phi_data, theta_exp, phi_exp, casefile = getGojonData(datadir, 0.02, 0.02, shape='D', B=2, RPM=8000)
data = data[:, ind_theta, ind_phi]
x_cart = x_cart_data[:, ind_theta, ind_phi].reshape((3, 1))
theta = theta_data[ind_theta]
phi = phi_data[ind_phi]

ms = np.arange(1, 11, 1) # harmonics to extract

fig, ax = plt.subplots(figsize=(7, 4))

for INDEX, (sourceArray, SUFFIX, COLOR, LABEL) in enumerate(zip(
    CASES, SUFFIXES, COLORS, LABELS
)):
    print(f'processing case {SUFFIX}')
    sourceArray.numerics['CompactnessCorrection'] = True

    NDIPOLES = sourceArray.Ndipoles

    r_inner, Fz, Fphi  = read_force_file('./Data/Zamponi2026/FS_ISAE_2_8000.txt') # reuse the radial stations from data

    BLH, _, _, _ = sourceArray.getLoading(Fz, Fphi, steady_only=True if INDEX > 0 else False) # compute loading on the fly, return PIN for reuse

    PIN = sourceArray.PIN
    D_bras = sourceArray.green.radius * 2
    g = -1 * sourceArray.green.origin[2]
    B = sourceArray.B
    c = sourceArray.chord[0]
    Omega = sourceArray.Omega
    c0 = sourceArray.SoS
    han = sourceArray.getHanson()

    Nr = len(r_inner)


    # -------------------------------- SCATTERED LOADING NOISE ------------------------------------------
    # save gradients in the far-field (run once per observer and m)
    for index, sm in enumerate(sourceArray.children):

        gradG_surface = np.load(f'./Data/current/NACA0012_rotor/gradG_surface_sm_{index}_{MODE}{SUFFIX}.npy') # shape (3, Nm, Nz, Ny)
        print(f'pre-computing far-field gradients {index+1}')

        gradG = sm.getScatteringGreenGradient(x_cart, ms*B * Omega / c0, gradG_surface) # shape (3, Nm, Nx, Ny)
        np.save(f'./Data/current/NACA0012_rotor/gradG_sm_{index}_{MODE}_{ind_theta}_{ind_phi}{SUFFIX}.npy', gradG)

    # extract and rearrange

    gradG_arr = np.zeros((Nr, 3, ms.shape[0], x_cart.shape[1], NDIPOLES), dtype=np.complex128)
    for index, sm in enumerate(sourceArray.children):
        gradG_arr[index] = np.load(f'./Data/current/NACA0012_rotor/gradG_sm_{index}_{MODE}_{ind_theta}_{ind_phi}{SUFFIX}.npy')

    p_scattered = sourceArray.getScatteredPressure(x_cart, ms, gradG=gradG_arr)[0]
    p_direct = sourceArray.getDirectPressure(x_cart, ms)[0]

    # -------------------------------- SCATTERED Thickness NOISE ------------------------------------------
    # save gradients in the far-field (run once per observer and m)
    for index, sm in enumerate(sourceArray.children):

        G_surface = np.load(f'./Data/current/NACA0012_rotor/G_surface_sm_{index}_{MODE}{SUFFIX}.npy') # shape (Nm, Nz, Ny)
        print(f'pre-computing far-field gradients {index+1}')

        G = sm.getScatteringGreen(x_cart, ms*B * Omega / c0, G_surface) # shape (Nm, Nx, Ny)
        np.save(f'./Data/current/NACA0012_rotor/G_sm_{index}_{MODE}_{ind_theta}_{ind_phi}{SUFFIX}.npy', G)

    G_arr = np.zeros((Nr, ms.shape[0], x_cart.shape[1], NDIPOLES), dtype=np.complex128)
    for index, sm in enumerate(sourceArray.children):
        G_arr[index] = np.load(f'./Data/current/NACA0012_rotor/G_sm_{index}_{MODE}_{ind_theta}_{ind_phi}{SUFFIX}.npy')

    p_scattered_thickness = sourceArray.getThicknessPressureScattered(x_cart, ms, G=G_arr)[0]
    p_direct_thickness = sourceArray.getThicknessPressureDirect(x_cart, ms)[0]

    p_total_scattering = p_direct + p_scattered + p_direct_thickness + p_scattered_thickness

    # SPLS TOTAL SCATTERING
    SPL_total_scattering = p_to_SPL(p_total_scattering)

    ax.plot(ms, SPL_total_scattering, label=LABEL, color=COLOR, marker='s', linestyle='dashed')

ax.plot(freq[0] / BPF, spl_from_autopower(data), label=f"Experimental, W=0mm", color='k', alpha=0.75)
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
