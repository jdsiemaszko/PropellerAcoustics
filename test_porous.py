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
from matplotlib.lines import Line2D

# BEGINNING OF HEADER
FILE='TOTAL'
MODE = 'half'
from Constants.helpers import read_force_file, plot_3D_directivity, plot_3D_phase_directivity, p_to_SPL, spl_from_autopower, plot_BPF_peaks
from Constants.data_assim import getGojonData
# vary configuration
from SourceMode.Configurations_NACA0012 import m_surface, D20L20W00_D180
from SourceMode.Configuration_Porous_NACA0012 import D20L20_porous


fig, ax = plt.subplots(figsize=(12, 5))

for sourceArray, SUFFIX,  shape, COLOR, LABEL in zip(
    [D20L20W00_D180, D20L20_porous, D20L20_porous, D20L20_porous, D20L20_porous, D20L20_porous], # rigid, porous
    ['_D180_MR', 'D20L20_POROUS_40mm_w_tape', 'D20L20_POROUS_30mm_w_tape','D20L20_POROUS_20mm_w_tape','D20L20_POROUS_10mm_w_tape', 'D20L20_POROUS_RIGID_v2'],
    ['D', 'D', 'D', 'D', 'D', 'D'],
    ['r', 'g', 'b', 'c', 'm', 'k'],
    ['Rigid', '40mm w. tape', '30mm w. tape', '20mm w. tape', '10mm w. tape', r'$|Z|\rightarrow \infty$ (control)']
):
    sourceArray.numerics['CompactnessCorrection'] = True
    # sourceArray.numerics['CompactnessCorrection'] = False
    NDIPOLES = sourceArray.Nsources

    r_inner, Fz, Fphi  = read_force_file('./Data/Zamponi2026/FS_ISAE_2_8000.txt') # reuse the radial stations from data
    Omega_ref = 8000/60*2*np.pi

    Fz *= (5.96 / 6.4) # rescale thrust and torque to roughly match the data from Montoya-Ospina et al. 2026
    Fphi *= (4.04 / 4.0)

    D_bras = sourceArray.green.radius * 2
    g = -1 * sourceArray.green.origin[2]
    BLH, BLH_S, BLH_US, _ = sourceArray.getLoading(Fz, Fphi, D_bras, g, steady_only=False) # compute loading on the fly, return PIN for reuse
    PIN = sourceArray.PIN

    # g= 0.02
    B = sourceArray.B
    c = sourceArray.chord
    Omega = sourceArray.Omega
    if shape == 'PARROT':
        Omega *= -1

    c0 = sourceArray.SoS
    han = sourceArray.getHanson()


    # Experimental data from Gojon et al. 2023

    ind_theta = 6+5     # theta_exp from 60 to -60 in 10, pick theta=140, theta_exp = -50
    ind_phi = 27          # phi_exp 0 to 350 in 10, pick phi = 270, phi_exp = 270

    datadir = './Experimental/dataverse_files'
    data, BPF, freq, x_cart_data, theta_data, phi_data, theta_exp, phi_exp, casefile = getGojonData(datadir, D_bras, g, shape=shape, B=2, 
                                                                                                    RPM=int(Omega * 60/2/np.pi)
                                                                                                    )
    data = data[:, ind_theta, ind_phi]
    x_cart = x_cart_data[:, ind_theta, ind_phi].reshape((3, 1))
    theta = theta_data[ind_theta]
    phi = phi_data[ind_phi]

    print(theta, phi)

    # extract the data from Montoya-Ospina et al. 2026
    data_porous = np.loadtxt('./Data/Santiago2026/Spectra_theta_140deg.csv', delimiter=';', skiprows=1)

    f_porous = data_porous[:, 0]
    SPL_porous = data_porous[:, 28]

    Nr = len(r_inner)
    ms = np.arange(1, 11, 1) # harmonics to extract


    # PIN MODEL
    pLSmB_model_rotor = han.getPressureRotor(x_cart, ms, 
                                        BLH_S
                                        )[0][0]

    pLUSmB_model_rotor = han.getPressureRotor(x_cart, ms, 
                                        BLH_US
                                        )[0][0]

    ptmB_model_rotor = han.getThicknessNoiseRotor(x_cart, ms, sourceArray.seg_chord, 0.082 * np.ones_like(r_inner))[0][0] # NACA0012


    PIN._numerics['include_vortex_sources'] = True
    PIN._numerics['include_thickness_sources'] = False
    BL = PIN.getStrutLoadingHarmonics()
    pmB_model_beam_loading = han.getPressureStator(x_cart, ms*B, BL)[0][0]

    PIN._numerics['include_vortex_sources'] = False
    PIN._numerics['include_thickness_sources'] = True
    BL = PIN.getStrutLoadingHarmonics()
    pmB_model_beam_thickness = han.getPressureStator(x_cart, ms*B, BL)[0][0]

    PIN._numerics['include_vortex_sources'] = True
    PIN._numerics['include_thickness_sources'] = True
    BL = PIN.getStrutLoadingHarmonics()
    pmB_model_beam_total = han.getPressureStator(x_cart, ms*B, BL)[0][0]


    pmB_model_rotor_total = pLSmB_model_rotor + pLUSmB_model_rotor + ptmB_model_rotor
    pmB_model_rotor_loading = pLSmB_model_rotor + pLUSmB_model_rotor
    pmB_model_total = pLSmB_model_rotor + pLUSmB_model_rotor + ptmB_model_rotor + pmB_model_beam_total # assuming coherent
    # pmB_model_total = np.sqrt(np.abs(pmB_model_rotor_total)**2 + np.abs(pmB_model_beam)**2) # assuming incoherent


    # SOURCE-MODE MODEL
    Nchildren = len(sourceArray.children)

    # -------------------------------- SCATTERED LOADING NOISE ------------------------------------------
    # save gradients in the far-field (run once per observer and m)
    # NOTE: this assumes precompute_gradients.py has been run to save the surface gradiens a priori
    for index, sm in enumerate(sourceArray.children):

        gradG_surface = np.load(f'./Data/current/NACA0012_rotor/gradG_surface_sm_{index}_{MODE}{SUFFIX}.npy') # shape (3, Nm, Nz, Ny)
        print(f'pre-computing far-field gradients {index+1}')

        gradG = sm.getScatteringGreenGradient(x_cart, ms*B * np.abs(Omega)  / c0, gradG_surface) # shape (3, Nm, Nx, Ny)
        np.save(f'./Data/current/NACA0012_rotor/gradG_sm_{index}_{MODE}_{ind_theta}_{ind_phi}_{FILE}{SUFFIX}.npy', gradG)

    # extract and rearrange
    gradG_arr = np.zeros((Nchildren, 3, ms.shape[0], x_cart.shape[1], NDIPOLES), dtype=np.complex128)
    for index, sm in enumerate(sourceArray.children):
        gradG_arr[index] = np.load(f'./Data/current/NACA0012_rotor/gradG_sm_{index}_{MODE}_{ind_theta}_{ind_phi}_{FILE}{SUFFIX}.npy')

    # p_direct_s = sourceArray.getDirectPressure(x_cart, ms, BLH=np.transpose(BLH_S, axes=[2, 0, 1]))[0]
    # p_direct_us = sourceArray.getDirectPressure(x_cart, ms, BLH=np.transpose(BLH_US, axes=[2, 0, 1]))[0]

    sourceArray.updateBLH(BLH_S)
    p_direct_s = sourceArray.getDirectPressure(x_cart, ms)[0]
    p_scattered_s = sourceArray.getScatteredPressure(x_cart, ms, gradG=gradG_arr)[0]

    sourceArray.updateBLH(BLH_US)
    p_direct_us = sourceArray.getDirectPressure(x_cart, ms)[0]
    p_scattered_us = sourceArray.getScatteredPressure(x_cart, ms, gradG=gradG_arr)[0]

    # -------------------------------- SCATTERED Thickness NOISE ------------------------------------------
    # save gradients in the far-field (run once per observer and m)
    # NOTE: this assumes precompute_gradients.py has been run to save the surface G a priori
    for index, sm in enumerate(sourceArray.children):

        G_surface = np.load(f'./Data/current/NACA0012_rotor/G_surface_sm_{index}_{MODE}{SUFFIX}.npy') # shape (Nm, Nz, Ny)
        print(f'pre-computing far-field gradients {index+1}')

        G = sm.getScatteringGreen(x_cart, ms*B * np.abs(Omega)  / c0, G_surface) # shape (Nm, Nx, Ny)
        np.save(f'./Data/current/NACA0012_rotor/G_sm_{index}_{MODE}_{ind_theta}_{ind_phi}_{FILE}{SUFFIX}.npy', G)


    G_arr = np.zeros((Nchildren, ms.shape[0], x_cart.shape[1], NDIPOLES), dtype=np.complex128)
    for index, sm in enumerate(sourceArray.children):
        G_arr[index] = np.load(f'./Data/current/NACA0012_rotor/G_sm_{index}_{MODE}_{ind_theta}_{ind_phi}_{FILE}{SUFFIX}.npy')


    p_scattered_thickness = sourceArray.getThicknessPressureScattered(x_cart, ms, G=G_arr)[0]
    p_direct_thickness = sourceArray.getThicknessPressureDirect(x_cart, ms)[0]

    for element in [p_direct_s, p_direct_us, p_scattered_s, p_scattered_us, p_direct_thickness, p_scattered_thickness]:
        element[np.isnan(element)] = 0.0

    p_total_scattering = p_direct_s + p_direct_us + p_scattered_s + p_scattered_us + p_direct_thickness + p_scattered_thickness
    p_rotor_total = p_direct_s + p_direct_us + p_direct_thickness
    p_total_minus_scattered_thickness = p_total_scattering - p_scattered_thickness

    # SPLS PIN MODEL
    SPL_rotor_loading = p_to_SPL(pmB_model_rotor_loading)
    SPL_rotor_S = p_to_SPL(pLSmB_model_rotor)
    SPL_rotor_US = p_to_SPL(pLUSmB_model_rotor)

    SPL_rotor_total = p_to_SPL(pmB_model_rotor_total)
    SPL_rotor_thickness = p_to_SPL(ptmB_model_rotor)
    SPL_beam_loading = p_to_SPL(pmB_model_beam_loading)
    SPL_beam_thickness = p_to_SPL(pmB_model_beam_thickness)

    SPL_total_PIN = p_to_SPL(pmB_model_total)

    # SPLS SCATTERING
    SPL_direct_s = p_to_SPL(p_direct_s)
    SPL_direct_us = p_to_SPL(p_direct_us)
    SPL_scattered_s = p_to_SPL(p_scattered_s)
    SPL_scattered_us = p_to_SPL(p_scattered_us)
    SPL_scattered = p_to_SPL(p_scattered_s + p_scattered_us)
    SPL_direct_thickness = p_to_SPL(p_direct_thickness)
    SPL_scattered_thickness = p_to_SPL(p_scattered_thickness)
    SPL_SM_rotor_total = p_to_SPL(p_rotor_total)
    SPL_total_scattering = p_to_SPL(p_total_scattering)
    SPL_total_scattering_minus_scattered_thickness = p_to_SPL(p_total_minus_scattered_thickness)



    # --- plotting ---
    # ax.plot(ms, SPL_rotor_S, color='r', marker='^')
    # ax.plot(ms, SPL_rotor_US, color='g', marker='^')
    # ax.plot(ms, SPL_rotor_thickness, color='b', marker='^')
    # ax.plot(ms, SPL_rotor_total, color='y', marker='^')
    # ax.plot(ms, SPL_beam_loading, color='m', marker='^')
    # ax.plot(ms, SPL_beam_thickness, color='c', marker='^')

    # if 'POROUS' not in SUFFIX: # PIN only works for a rigid beam
    #     ax.plot(ms, SPL_total_PIN, color=COLOR, marker='^')

    # ax.plot(ms, SPL_direct_s, color='r', marker='s', linestyle='--')
    # ax.plot(ms, SPL_direct_us, color='g', marker='s', linestyle='--')
    # ax.plot(ms, SPL_direct_thickness, color='b', marker='s', linestyle='--')
    # ax.plot(ms, SPL_SM_rotor_total, color='y', marker='s', linestyle='--')
    # ax.plot(ms, SPL_scattered, color='m', marker='s', linestyle='--')
    # ax.plot(ms, SPL_scattered_thickness, color='c', marker='s', linestyle='--')
    ax.plot(ms, SPL_total_scattering, color=COLOR, marker='s', linestyle='--',
            #  label=LABEL
             )

ax.plot(freq[0]/BPF,
        spl_from_autopower(data),
        color='r',
        linewidth=1)

fig, ax = plot_BPF_peaks(fig, ax, freq[0] / BPF, spl_from_autopower(data), N0=1, N1= 25, range=0.01, 
                         plot_kwargs={
                             'color':'r',
                             'linestyle':'dashed',
                             'alpha':1.0,
                             'linewidth': 1
                         })

ax.plot(f_porous/BPF,
        SPL_porous,
        color='g',
        linewidth=1)

fig, ax = plot_BPF_peaks(fig, ax, f_porous / BPF, SPL_porous, N0=1, N1= 25, range=0.01, 
                         plot_kwargs={
                             'color':'g',
                             'linestyle':'dashed',
                             'alpha':1.0,
                             'linewidth': 1
                         })

component_handles = [
    Line2D([0], [0], color='r', lw=2, label='Steady Rotor Loading'),
    Line2D([0], [0], color='g', lw=2, label='Unsteady Rotor Loading'),
    Line2D([0], [0], color='b', lw=2, label='Rotor Thickness'),
    Line2D([0], [0], color='y', lw=2, label='Rotor Total'),
    Line2D([0], [0], color='m', lw=2, label='Strut Noise due to Rotor Loading'),
    Line2D([0], [0], color='c', lw=2, label='Strut Noise due to Rotor Thickness'),
    Line2D([0], [0], color='k', lw=2, label='Total'),
]

model_handles = [
    # Line2D([0], [0], color='k', marker='^', linestyle='-',
    #        label='PIN'),
    Line2D([0], [0], color='k', marker='s', linestyle='--',
           label='Source-Modes'),
    Line2D([0], [0], color='0.3', lw=3,
           label='Experiment'),
]

case_handles = [
    Line2D([0], [0], color='r',
           label='Rigid'),
    Line2D([0], [0], color='g', 
           label='Porous'),
]

case_handles_SM = [
    Line2D([0], [0], color='r',
        label='Rigid', marker='s'),
    Line2D([0], [0], color='g', 
           label='Porous, h=40mm', marker='s'),
    Line2D([0], [0], color='b', 
           label='Porous, h=30mm', marker='s'),
               Line2D([0], [0], color='c', 
           label='Porous, h=20mm', marker='s'),
               Line2D([0], [0], color='m', 
           label='Porous, h=10mm', marker='s'),
    Line2D([0], [0], color='k', 
        label=r'Porous, $Z\rightarrow \infty$', marker='s'),
]

leg1 = ax.legend(handles=component_handles,
                 title='Noise Component',
                 loc='upper left')

leg2 = ax.legend(handles=model_handles,
                #  title='Model',
                 loc='lower left')
leg3 = ax.legend(handles=case_handles, loc='upper left')
leg4 = ax.legend(handles=case_handles_SM, loc='upper right')

# ax.add_artist(leg1)
ax.add_artist(leg2)
ax.add_artist(leg3)
ax.add_artist(leg4)



# ax.plot(ms, SPL_total_scattering_minus_scattered_thickness, label=f"Direct+Scattering (Minus Thickness)", color='g', marker='s', linestyle='dashed')


# ax.legend(ncol=2, loc='upper left', fontsize=8)
ax.set_xlabel("$f^+ = f/B/\Omega$ (Hz)")
ax.set_ylabel("SPL (dB)")
ax.set_xscale('log')

ax.grid(visible=True, which='major', color='k', linestyle='-')
ax.grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.5)
# ax.set_title(f'Theta = {theta} deg, Phi = {phi} deg')
# plt.xlim(0.03333, 100)
plt.xlim(0.1, 100)

plt.ylim(0, 75)
plt.tight_layout()
plt.show()
