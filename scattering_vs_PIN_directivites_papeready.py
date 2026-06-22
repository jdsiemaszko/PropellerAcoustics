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

# from scattering_vs_PIN import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from Constants.data_assim import getGojonData, getHarmonicsFromData
from PotentialInteraction.PIN import PotentialInteraction
from Constants.helpers import read_force_file, plot_3D_directivity, plot_3D_phase_directivity, plot_beam_azimuth, plot_rotation_arrow
MODE = 'half'
FILE = 'TOTAL_DIR'
Ntheta = 18
Nphi = 36


# BEGINNING OF HEADER
# vary configuration
from SourceMode.Configurations_NACA0012 import m_surface

# from SourceMode.Configurations_NACA0012 import D20L20W00_D360 as sourceArray # pick configuration
# SUFFIX = '_D360_HR'

# from SourceMode.Configurations_NACA0012 import D20L20W20_D180 as sourceArray # pick configuration
# SUFFIX = '_D20L20W20_D180'

# from SourceMode.Configurations_NACA0012 import D20L20W00_D180 as sourceArray # pick configuration
# SUFFIX = '_D180_MR'
# shape='D'

# from SourceMode.Configurations_NACA0012 import D10L20W00_D180 as sourceArray # pick configuration
# SUFFIX = '_D10L20_D180'
# shape='D'

# from SourceMode.Configurations_NACA0012 import D15L20W00_D180 as sourceArray # pick configuration
# SUFFIX = 'D15L20_D180'
# shape='D'

from SourceMode.Configurations_NACA0012 import PARROT_D20L20W00_D180 as sourceArray # pick configuration
SUFFIX = 'PARROT_D20L20_D180'
shape = 'PARROT'

# from SourceMode.Configurations_NACA0012 import PARROT_D20L21W00_D180 as sourceArray # pick configuration
# SUFFIX = 'PARROT_D20L20_D180_v2'
# shape = 'PARROT'

# from SourceMode.Configurations_NACA0012 import PARROT_D20L20W00_D36_1_10 as sourceArray # pick configuration
# SUFFIX = 'PARROT_D20L20_D36_1_10'
# shape = 'PARROT'

# from SourceMode.Configurations_NACA0012 import PARROT_D20L20W00_D36_5_10 as sourceArray # pick configuration
# SUFFIX = 'PARROT_D20L20_D36_5_10'
# shape = 'PARROT'

# from SourceMode.Configurations_NACA0012 import D20L20W00_D180_6000RPM as sourceArray
# SUFFIX = 'D20L20_D180_6000RPM'
# shape='D'

sourceArray.numerics['CompactnessCorrection'] = True

NDIPOLES = sourceArray.Nsources
mss = np.array([1, 2, 3, 4, 5])

for ms in mss:
    sourceArray.numerics['CompactnessCorrection'] = True

    print(f'plotting mode {ms}')
    ms = np.array([ms])

    r_inner, Fz, Fphi  = read_force_file('./Data/Zamponi2026/FS_ISAE_2_8000.txt') # reuse the radial stations from data
    Omega_ref = 8000/60*2*np.pi
    if shape == "PARROT":
        rt, t =  np.loadtxt('./Data/Parrot2024/thrust_Npm.csv', skiprows=1, delimiter=',').T # radius/r1, thrust in Npm
        rq, q =  np.loadtxt('./Data/Parrot2024/torque_Nmpm.csv', skiprows=1, delimiter=',').T # radius/r1, torque in Nmpm

        # q[18:] /= 1.125

        r_inner = sourceArray.seg_radius
        r1 = sourceArray.r1
        Fz = np.interp(r_inner/r1, rt, t) # same radial array
        Q = np.interp(r_inner/r1, rq, q) 
        Fphi = Q / r_inner

        TTARGET = 2.15 / sourceArray.B # Newtons
        QTARGET = 25 / 1000 / sourceArray.B # Newton-radian-meters
        Fz *= TTARGET / np.trapezoid(Fz, r_inner)  # rescale to target
        Fphi *= QTARGET / np.trapezoid(Fphi * r_inner, r_inner) # rescale to target

    elif shape == 'D' and sourceArray.Omega != Omega_ref:
        print(f'rescaling the loading from {Omega_ref} to {sourceArray.Omega} rad/s')
        Fz *= (sourceArray.Omega/Omega_ref)**2
        Fphi *= (sourceArray.Omega/Omega_ref)**2

    D_bras = sourceArray.green.radius * 2
    g = -1 * sourceArray.green.origin[2]
    BLH, BLH_S, BLH_US, _ = sourceArray.getLoading(Fz, Fphi, D_bras, g, steady_only=False) # compute loading on the fly, return PIN for reuse
    PIN = sourceArray.PIN
    # g=0.02
    B = sourceArray.B
    c = sourceArray.chord
    Omega = sourceArray.Omega
    if shape == 'PARROT':
        Omega *= -1

    c0 = sourceArray.SoS
    han = sourceArray.getHanson()
    # END OF HEADER

    sourceArray.updateBLH(BLH)


    datadir = './Experimental/dataverse_files'
    # casefile = f'ISAE_2_D{int(1000*D_bras)}_L{int(1000*g)}'

    data, BPF, freq, x_cart_data, theta_data, phi_data, theta_exp, phi_exp, casefile = getGojonData(datadir, D_bras, g, shape=shape, B=sourceArray.B, RPM=int(Omega * 60/2/np.pi))

    data_modal, ms_data = getHarmonicsFromData(data, freq.T, BPF)
    data = data_modal[np.where(ms[0] == ms_data)]

    peq_data = np.sqrt(np.array(data) / 2) # "equivalent" mode amplitude from the data

    theta_m_data, phi_m_data = np.meshgrid(theta_data, phi_data, indexing='ij')
    theta_arr_data = theta_m_data.ravel()
    phi_arr_data = phi_m_data.ravel()          

                                

    # angular coordinates
    theta = np.linspace(0.0, np.pi, Ntheta, endpoint=True)
    phi   = np.linspace(0.0, 2.0 * np.pi, Nphi, endpoint=True)

    # 2D mesh
    theta_m, phi_m = np.meshgrid(theta, phi, indexing='ij')
    # shapes: (Ntheta, Nphi)

    # flatten
    R = np.max(np.linalg.norm(x_cart_data, axis=0))
    R_arr     = np.full(theta_m.size, R)
    theta_arr = theta_m.ravel()
    phi_arr   = phi_m.ravel()

    X = R_arr * np.sin(theta_arr) * np.cos(phi_arr)
    Y = R_arr * np.sin(theta_arr) * np.sin(phi_arr)
    Z = R_arr * np.cos(theta_arr)

    x_cart = np.array([X, Y, Z])


    PIN._numerics['only_linear'] = True 
    PIN._numerics['only_nonlinear'] = False

    PIN._numerics['include_vortex_sources'] = True
    PIN._numerics['include_thickness_sources'] = False
    beam_loading = PIN.getStrutLoadingHarmonics() 

    p_beam_loading, _ = han.getPressureStator(x_cart, ms*B, beam_loading) # mind the indexing change for m
    p_blade_loading, _ = han.getPressureRotor(x_cart, ms, BLH) 

    p_blade_thickness, _ = han.getThicknessNoiseRotor(x_cart, ms, sourceArray.seg_chord, 0.082 * np.ones_like(r_inner)) # NACA0012

    PIN._numerics['include_vortex_sources'] = False
    PIN._numerics['include_thickness_sources'] = True
    beam_loading = PIN.getStrutLoadingHarmonics() 

    p_beam_thickness, _ = han.getPressureStator(x_cart, ms * B, beam_loading) # loading beam noise due to blade thickness, not to be confused with beam thickness noise, which is zero since the beam is stationary


    PIN._numerics['include_vortex_sources'] = True
    PIN._numerics['include_thickness_sources'] = True
    beam_loading = PIN.getStrutLoadingHarmonics()



    # Non-linear only!
    PIN._numerics['only_linear'] = False 
    PIN._numerics['only_nonlinear'] = True
    PIN._numerics['include_vortex_sources'] = True
    PIN._numerics['include_thickness_sources'] = True
    beam_loading_nonlinear_only = PIN.getStrutLoadingHarmonics() 



    ##### -------------------------------- SCATTERED LOADING NOISE ------------------------------------------
    #### save gradients in the far-field (run once per observer and m)
    # for index, sm in enumerate(sourceArray.children):

    #     gradG_surface = np.load(f'./Data/current/NACA0012_rotor/gradG_surface_sm_{index}_{MODE}{SUFFIX}.npy') # shape (3, Nm, Nz, Ny)
    #     print(f'pre-computing far-field gradients {index+1}')

    #     gradG = sm.getScatteringGreenGradient(x_cart, m_surface * B * np.abs(Omega)  / c0, gradG_surface) # shape (3, Nm, Nx, Ny)
    #     np.save(f'./Data/current/NACA0012_rotor/gradG_sm_{index}_{MODE}_{FILE}{SUFFIX}.npy', gradG)


    # extract and rearrange
    gradG_arr = np.zeros((len(sourceArray.children), 3, ms.shape[0], x_cart.shape[1], NDIPOLES), dtype=np.complex128)
    ind_m = np.where(m_surface == ms[0])[0][0]
    for index, sm in enumerate(sourceArray.children):
        gradG_arr[index] = np.load(f'./Data/current/NACA0012_rotor/gradG_sm_{index}_{MODE}_{FILE}{SUFFIX}.npy')[:, ind_m, :, :].reshape(3, ms.shape[0], x_cart.shape[1], NDIPOLES)



    # total scattered loading
    sourceArray.updateBLH(BLH_S)
    p_direct_s = sourceArray.getDirectPressure(x_cart, ms)
    p_scattered_s = sourceArray.getScatteredPressure(x_cart, ms, gradG=gradG_arr)

    sourceArray.updateBLH(BLH_US)
    p_direct_us = sourceArray.getDirectPressure(x_cart, ms)
    p_scattered_us = sourceArray.getScatteredPressure(x_cart, ms, gradG=gradG_arr)

    # only steady loading is scattered
    # BLH_S = np.zeros_like(BLH)
    # BLH_S[:, 0, :] = BLH[:, 0, :]
    # p_scattered_loading = sourceArray.getScatteredPressure(x_cart, ms, gradG=gradG_arr, BLH=np.transpose(BLH_S, axes=[2, 0, 1]))



    # p_scattered = np.load(f'./Data/current/NACA0012_rotor/p_scattered_{MODE}_m{int(m)}_{casename}{SUFFIX}.npy')
    # p_direct_blade = np.load(f'./Data/current/NACA0012_rotor/p_direct_{MODE}_m{int(m)}_{casename}{SUFFIX}.npy')




    #### -------------------------------- SCATTERED Thickness NOISE ------------------------------------------

    ### save gradients in the far-field (run once per observer and m)
    # for index, sm in enumerate(sourceArray.children):
    #     G_surface = np.load(f'./Data/current/NACA0012_rotor/G_surface_sm_{index}_{MODE}{SUFFIX}.npy') # shape (Nm, Nz, Ny)
    #     print(f'pre-computing far-field G {index+1}')

    #     G = sm.getScatteringGreen(x_cart, m_surface * B * np.abs(Omega) / c0, G_surface) # shape (Nm, Nx, Ny)
    #     np.save(f'./Data/current/NACA0012_rotor/G_sm_{index}_{MODE}_{FILE}{SUFFIX}.npy', G)

    Nr = sourceArray.seg_radius.shape[0]

    # G_arr = np.zeros((Nr, m_surface.shape[0], x_cart.shape[1], NDIPOLES), dtype=np.complex128)

    G_arr = np.zeros((len(sourceArray.children), ms.shape[0], x_cart.shape[1], NDIPOLES), dtype=np.complex128) # pick only the ones we neeed
    ind_m = np.where(m_surface == ms[0])[0][0]
    for index, sm in enumerate(sourceArray.children):
        G_arr[index] = np.load(f'./Data/current/NACA0012_rotor/G_sm_{index}_{MODE}_{FILE}{SUFFIX}.npy')[ind_m, :, :] # extract only the m we need for plotting!


    p_beam_total_linear , _ = han.getPressureStator(x_cart, ms*B, beam_loading)
    p_beam_nonlinear, _ = han.getPressureStator(x_cart, ms*B, beam_loading_nonlinear_only)

    p_scattered_thickness = sourceArray.getThicknessPressureScattered(x_cart, ms, G=G_arr)
    p_direct_thickness = sourceArray.getThicknessPressureDirect(x_cart, ms)


    # scattering total

    for element in [p_direct_s, p_direct_us, p_scattered_s, p_scattered_us, p_direct_thickness, p_scattered_thickness]:
        element[np.isnan(element)] = 0.0
    p_total_scattering = p_direct_s + p_direct_us + p_scattered_s + p_scattered_us + p_direct_thickness + p_scattered_thickness

    # pin total
    p_total_pin = p_blade_loading + p_blade_thickness + p_beam_total_linear + p_beam_nonlinear
    p_total_pin_loading = p_blade_loading + p_blade_thickness + p_beam_loading


    # uncorrected scattering solutions
    sourceArray.numerics['CompactnessCorrection'] = False

    sourceArray.updateBLH(BLH_S)
    C_p_direct_s = sourceArray.getDirectPressure(x_cart, ms)
    C_p_scattered_s = sourceArray.getScatteredPressure(x_cart, ms, gradG=gradG_arr)

    sourceArray.updateBLH(BLH_US)
    C_p_direct_us = sourceArray.getDirectPressure(x_cart, ms)
    C_p_scattered_us = sourceArray.getScatteredPressure(x_cart, ms, gradG=gradG_arr)

    C_p_scattered_loading = C_p_scattered_s +  C_p_scattered_us

    C_p_scattered_thickness = sourceArray.getThicknessPressureScattered(x_cart, ms, G=G_arr)
    C_p_direct_thickness = sourceArray.getThicknessPressureDirect(x_cart, ms)

    # shift to relative phase w.r.t. mic one at phi=0
    # ind_theta_ref = 0
    # ind_phi_ref = np.where(phi==0)[0][0]
    # ind_combined = ind_theta_ref * phi.shape[0] + ind_phi_ref

    # p_total_scattering *= np.exp(-1j * np.angle(p_total_scattering[ind_combined, :]))
    # p_total_pin  *= np.exp(-1j * np.angle(p_total_pin [ind_combined, :]))
    # p_total_pin_loading *= np.exp(-1j * np.angle(p_total_pin_loading[ind_combined, :]))




    # + p_beam_total 
    # + p_beam_loading + p_beam_thickness

    import os
    folder_name = f'./Figures/{SUFFIX}_{ms[0]}'
    os.makedirs(folder_name, exist_ok=True)

    components = [
        {
            "name": "pin_total_loading_only",
            "title": "PIN Model (vortex only)",
            "data": p_total_pin_loading[:, 0],
            "theta": theta_m,
            "phi": phi_m,
        },

        {
            "name": "scattering_total",
            "title": "Scattering Model",
            "data": p_total_scattering[:, 0],
            "theta": theta_m,
            "phi": phi_m,
        },
        {
            "name": "experiment",
            "title": "Experiment",
            "data": peq_data[0, :, :],
            "theta": np.deg2rad(theta_m_data),
            "phi": np.deg2rad(phi_m_data),
        },
        {
            "name": "pin_total",
            "title": "PIN Model (incl. blade thickness)",
            "data": p_total_pin[:, 0],
            "theta": theta_m,
            "phi": phi_m,
        },

        # parts of the scattering model
            {
            "name": "direct_thickness",
            "title": r"$\langle G_0 Q\rangle_\Omega$",
            "data": p_direct_thickness[:, 0],
            "theta": theta_m,
            "phi": phi_m,
        },
        {
            "name": "scattered_thickness",
            "title": r"$\langle G_s Q\rangle_\Omega$",
            "data": p_scattered_thickness[:, 0],
            "theta": theta_m,
            "phi": phi_m,
        },
        {
            "name": "direct_loading_steady",
            "title": r"$-\langle \boldsymbol{\nabla} G_0 \circ \boldsymbol{F}_0\rangle_\Omega$",
            "data": p_direct_s[:, 0],
            "theta": theta_m,
            "phi": phi_m,
        },
        {
            "name": "scattered_loading_steady",
            "title": r"$-\langle \boldsymbol{\nabla} G_s \circ \boldsymbol{F}_0\rangle_\Omega$",
            "data": p_scattered_s[:, 0],
            "theta": theta_m,
            "phi": phi_m,
        },
        {
            "name": "direct_loading_unsteady",
            "title": r"$-\sum_{k>0}\langle \boldsymbol{\nabla} G_0 \circ \boldsymbol{F}_k\rangle_\Omega$",
            "data": p_direct_us[:, 0],
            "theta": theta_m,
            "phi": phi_m,
        },
        {
            "name": "scattered_loading_unsteady",
            "title": r"$-\sum_{k>0}\langle \boldsymbol{\nabla} G_s \circ \boldsymbol{F}_k\rangle_\Omega$",
            "data": p_scattered_us[:, 0],
            "theta": theta_m,
            "phi": phi_m,
        },

                {
            "name": "scattered_loading",
            "title": r"$-\sum_{k>0}\langle \boldsymbol{\nabla} G_s \circ \boldsymbol{F}_k\rangle_\Omega$",
            "data": p_scattered_us[:, 0] + p_scattered_s[:, 0],
            "theta": theta_m,
            "phi": phi_m,
        },



        {       "name": "compact_scattered_loading_steady",
            "title": r"$-\langle \boldsymbol{\nabla} G_s \circ \boldsymbol{F}_0\rangle_\Omega$",
            "data": C_p_scattered_s[:, 0],
            "theta": theta_m,
            "phi": phi_m,
        },
        {
            "name": "compact_scattered_loading_unsteady",
            "title": r"$-\sum_{k>0}\langle \boldsymbol{\nabla} G_s \circ \boldsymbol{F}_k\rangle_\Omega$",
            "data": C_p_scattered_us[:, 0],
            "theta": theta_m,
            "phi": phi_m,
        },

                {
            "name": "compact_scattered_loading",
            "title": r"$-\sum_{k>0}\langle \boldsymbol{\nabla} G_s \circ \boldsymbol{F}_k\rangle_\Omega$",
            "data": C_p_scattered_us[:, 0] + C_p_scattered_s[:, 0],
            "theta": theta_m,
            "phi": phi_m,
        },
                {
            "name": "compact_scattered_thickness",
            "title": r"$\langle G_s Q\rangle_\Omega$",
            "data": C_p_scattered_thickness[:, 0],
            "theta": theta_m,
            "phi": phi_m,
        },

            {
            "name": "pin_nonlinear",
            "title": r"1/2rhov^2",
            "data": p_beam_nonlinear[:, 0],
            "theta": theta_m,
            "phi": phi_m,
        },

                    {
            "name": "pin_beam_loading_only",
            "title": r"1/2rhov^2",
            "data": p_beam_loading[:, 0],
            "theta": theta_m,
            "phi": phi_m,
        },

                        {
        "name": "pin_beam_thickness_only",
            "title": r"1/2rhov^2",
            "data": p_beam_thickness[:, 0],
            "theta": theta_m,
            "phi": phi_m,
        },
    ]

    VMIN, VMAX = 10, 65
    harmonic = int(ms[0] * B)
    R0 = 1.2
    R1 = R0 * 1.2

    for comp in components:

        # ==========================================================
        # SPL directivity
        # ==========================================================
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        fig, ax, SPL_mappable = plot_3D_directivity(
            comp["data"],
            comp["theta"],
            comp["phi"],
            fig=fig,
            ax=ax,
            valmin=VMIN,
            valmax=VMAX,
        )
        ax.set_xlim(-R0, R0)
        ax.set_ylim(-R0, R0)
        ax.set_zlim(-R0, R0)

        plot_beam_azimuth(R0, fig, ax)
        plot_rotation_arrow(R1, PHI_EXTENT=[20, 90], fig=fig, ax=ax)

        fig.savefig(
            os.path.join(
                folder_name,
                f"directivity_spl_{comp['name']}.pdf",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        # ==========================================================
        # Phase directivity
        # ==========================================================
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        fig, ax, phase_mappable = plot_3D_phase_directivity(
            comp["data"],
            comp["theta"],
            comp["phi"],
            fig=fig,
            ax=ax,
            valmin=VMIN,
            valmax=VMAX,
        )
        ax.set_xlim(-R0, R0)
        ax.set_ylim(-R0, R0)
        ax.set_zlim(-R0, R0)

        plot_beam_azimuth(R0, fig, ax)
        plot_rotation_arrow(R1, PHI_EXTENT=[20, 90], fig=fig, ax=ax)

        fig.savefig(
            os.path.join(
                folder_name,
                f"directivity_phase_{comp['name']}.pdf",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # ==========================================
    # SPL colorbar
    # ==========================================
    fig = plt.figure(figsize=(1.5, 5))
    cax = fig.add_axes([0.35, 0.05, 0.3, 0.9])

    cbar = fig.colorbar(
        SPL_mappable,
        cax=cax,
    )

    cbar.set_label("SPL [dB]")

    fig.savefig(
        os.path.join(folder_name, "colorbar_spl.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ==========================================
    # Phase colorbar
    # ==========================================
    fig = plt.figure(figsize=(1.5, 5))
    cax = fig.add_axes([0.35, 0.05, 0.3, 0.9])

    cbar = fig.colorbar(
        phase_mappable,
        cax=cax,
    )

    cbar.set_label("Phase [rad]")

    fig.savefig(
        os.path.join(folder_name, "colorbar_phase.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)