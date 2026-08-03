import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PotentialInteraction.blade_to_beam import BeamLoadings
from PotentialInteraction.beam_to_blade import BladeLoadings
from Hanson.far_field import HansonModel
from Constants.helpers import p_to_SPL, plot_BPF_peaks, spl_from_autopower, plot_directivity_contour, plot_3D_directivity, plot_3D_phase_directivity

import numpy as np

# from DataPost.Vella2026 import *
from DataPost.eVTOLUTION import *


# ------------------------------------------------------------------
# Class initializations using shared instances
# ------------------------------------------------------------------
MODE = 'half'
Nk = 40

# nr = -3
# nks = [0, 1, 2, 3, 4, 5]
# colors = plt.cm.viridis(np.linspace(0, 1, len(nks)))  # assign a color per k
# FACTORS = np.array([1, 2, 4, 8, 16, 32, 64, 128])

# # store results for plotting
# magnitudes_dynamic = {k: [] for k in nks}
# magnitudes_vortex = {k: [] for k in nks}


# for FACTOR in FACTORS:
#     beam_l = BeamLoadings(
#         twist_rad=twist,
#         chord_m=chord,
#         radius_m=r_outer,
#         Uz0_mps=U_flow * np.sqrt(FACTOR), # downwash scaled according to momentum thry
#         Tprime_Npm=dT / dr * FACTOR,
#         Qprime_Npm=dQ / dr * FACTOR,
#         B=B,
#         Dcylinder_m=D_bras,
#         Lcylinder_m=g,
#         Omega_rads=Omega,
#         rho_kgm3=rho0,
#         c_mps=c0,
#         kmax=Nk,
#         nb=NBEAMS
#     )

#     beam_loading_dynamic = beam_l.getBeamLoadingHarmonicsDynamic()  # shape (3, Nk, Nr)
#     beam_loading_vortex = beam_l.getBeamLoadingHarmonicsVortex()  # shape (3, Nk, Nr)

#     for i, k in enumerate(nks):
#         magnitudes_dynamic[k].append(np.abs(beam_loading_dynamic[2, k*B, nr]))
#         magnitudes_vortex[k].append(np.abs(beam_loading_vortex[2, k*B, nr]))

# # Plotting
# plt.figure(figsize=(8, 5))
# for i, k in enumerate(nks):
#     plt.plot(FACTORS, np.array(magnitudes_dynamic[k]) / FACTORS, '-^', color=colors[i], label=f'k={k}')
#     plt.plot(FACTORS, np.array(magnitudes_vortex[k]) / FACTORS, marker='s', color=colors[i], linestyle='dashed')

# # plt.hlines(0.1, FACTORS[0], FACTORS[-1], colors='k', linestyles='dotted')
# plt.xlabel('$F/F_0$')
# plt.ylabel('$|F_k|/ (F/F_0)$')
# # plt.title('Beam loading magnitude vs FACTOR for different k')
# plt.xscale('log')  # since FACTORS vary over orders of magnitude
# plt.yscale('log')  # optional, often loadings scale linearly with FACTOR
# plt.grid(True, which='both', ls='--', alpha=0.5)
# plt.legend()
# plt.tight_layout()
# plt.show()

# # Plotting
# plt.figure(figsize=(8, 5))
# for i, k in enumerate(nks):
#     plt.plot(FACTORS, (np.array(magnitudes_dynamic[k]) /np.array(magnitudes_vortex[k])), '-^', color=colors[i], label=f'k={k}')

# plt.hlines(np.sqrt(2)-1, FACTORS[0], FACTORS[-1], colors='k', linestyles='dotted')
# plt.xlabel('$F/F_0$')
# plt.ylabel('$|F_k^{{dynamic}}/F_k^{{vortex}}|$')
# # plt.title('Beam loading magnitude vs FACTOR for different k')
# plt.xscale('log')  # since FACTORS vary over orders of magnitude
# plt.yscale('log')  # optional, often loadings scale linearly with FACTOR
# plt.grid(True, which='both', ls='--', alpha=0.5)
# plt.legend()
# plt.tight_layout()
# plt.show()


FACTOR = 1.0
VMIN = VMIN + 20 * np.log10(FACTOR)
VMAX = VMAX + 20 * np.log10(FACTOR)
beam_l = BeamLoadings(
    twist_rad=twist,
    chord_m=chord,
    radius_m=r_outer,
    Uz0_mps=U_flow * np.sqrt(FACTOR),
    Tprime_Npm=dT / dr * FACTOR,
    Qprime_Npm=dQ / dr * FACTOR,
    B=B,
    Dcylinder_m=D_bras,
    Lcylinder_m=g,
    Omega_rads=Omega,
    rho_kgm3=rho0,
    c_mps=c0,
    kmax=Nk,
    nb=NBEAMS
)

blade_l = BladeLoadings(
    twist_rad= twist,
    chord_m= chord,
    radius_m=r_outer,
    Uz0_mps=U_flow * np.sqrt(FACTOR),
    Tprime_Npm=dT / dr * FACTOR,
    Qprime_Npm=dQ / dr * FACTOR,
    B=B,
    Dcylinder_m=D_bras,
    Lcylinder_m=g,
    Omega_rads=Omega,
    rho_kgm3=rho0,
    c_mps=c0,
    kmax=Nk,
    nb=NBEAMS
)


# HANSON (stator only)
axis_prop = np.array([0.0, 0.0, 1.0]) # z-direction propeller...
origin_prop = np.array([0.0, 0.0, 0.0]) # ... at z=0
han = HansonModel(
radius_m=r_outer, # blade radius stations [m] of size Nr + 1
axis=axis_prop, origin=origin_prop,
# radial=np.array([1, 0, 0]), # coordinate system (not needed here)
Omega_rads=Omega, # rotation speed [rad/s]
rho_kgm3=rho0, # fluid density [kg/m^3]
c_mps= c0, # speed of sound [m/s]
nb=NBEAMS # number of beams (irrelevant)
)

if __name__ == "__main__":
    # _____________ PLOTTING & RESULTS _______________

    # angular coordinates
    theta = np.linspace(0.0, np.pi, Ntheta, endpoint=True)
    phi   = np.linspace(0.0, 2.0 * np.pi, Nphi, endpoint=True)

    # 2D mesh
    theta_m, phi_m = np.meshgrid(theta, phi, indexing='ij')
    # shapes: (Ntheta, Nphi)

    # flatten
    R_arr     = np.full(theta_m.size, R)
    theta_arr = theta_m.ravel()
    phi_arr   = phi_m.ravel()

    X = R_arr * np.sin(theta_arr) * np.cos(phi_arr)
    Y = R_arr * np.sin(theta_arr) * np.sin(phi_arr)
    Z = R_arr * np.cos(theta_arr)

    x_cart = np.array([X, Y, Z])


    ############## save results to a file as the computation takes some time

    p_scattered = np.load(f'./Data/current/NACA0012_rotor/p_scattered_{MODE}_m{int(m)}_{casename}.npy') * FACTOR
    # p_direct_blade = np.load(f'./Data/current/NACA0012_rotor/p_direct_{MODE}_m{int(m)}.npy')

    BLH = blade_l.getBladeLoadingHarmonics(QS=True) # QS blade loading, used to compute the QS circulation!
    beam_loading = beam_l.getBeamLoadingHarmonics(BLH=BLH) 
    beam_loading_dynamic = beam_l.getBeamLoadingHarmonicsDynamic(BLH=BLH) # shape 3, Nk, Nr
    beam_loading_temporal = beam_l.getBeamLoadingHarmonicsVortex(BLH=BLH)
    
    p_direct_beam, _ = han.getPressureStator(x_cart, m*B, beam_loading) 
    p_dynamic_beam, _ = han.getPressureStator(x_cart, m*B, beam_loading_dynamic) 
    p_vortex_beam, _ = han.getPressureStator(x_cart, m*B, beam_loading_temporal) 


    # Plot maps:
    # 1) 2D contours
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) =  plt.subplots(3, 2, figsize=(10, 5), sharex=True, sharey=True)

    levels = np.linspace(VMIN, VMAX, 20, endpoint=False)
    fig, ax1 = plot_directivity_contour(
        np.rad2deg(theta_m), np.rad2deg(phi_m), p_scattered, title='blade scattered noise (SM)', fig=fig, ax=ax1, cmap='jet', levels=levels, xlabel=None,
    )
    fig, ax2 = plot_directivity_contour(
        np.rad2deg(theta_m), np.rad2deg(phi_m), p_vortex_beam, title=r'beam direct noise ($\rho\nabla(\omega \times v)$)', fig=fig, ax=ax2, cmap='jet', ylabel=None, levels=levels, xlabel=None,
    )
    fig, ax3 = plot_directivity_contour(
        np.rad2deg(theta_m), np.rad2deg(phi_m), p_dynamic_beam, title=r'beam direct noise ($1/2\rho\nabla^2(|u|^2)$)', fig=fig, ax=ax3, cmap='jet',  levels=levels,xlabel=None, 
    )

    fig, ax4 = plot_directivity_contour(
        np.rad2deg(theta_m), np.rad2deg(phi_m), p_vortex_beam + p_dynamic_beam,
        title=r'beam direct noise (total)', fig=fig, ax=ax4, cmap='jet',  levels=levels, ylabel=None,  xlabel=None, 
    )

    fig, ax5 = plot_directivity_contour(
        np.rad2deg(theta_m), np.rad2deg(phi_m), p_scattered + p_dynamic_beam,
        title=r'PIN dynamic + scattered loading', fig=fig, ax=ax5, cmap='jet',  levels=levels, 
    )
    fig, ax6 = plot_directivity_contour(
        np.rad2deg(theta_m), np.rad2deg(phi_m), p_direct_beam,
        title='beam total (reference)', fig=fig, ax=ax6, cmap='jet',  levels=levels, ylabel=None,

    )

    plt.show()
    plt.close()

    # 1) 3D blobs
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(321, projection="3d")
    ax2 = fig.add_subplot(322, projection="3d")
    ax3 = fig.add_subplot(323, projection="3d")
    ax4 = fig.add_subplot(324, projection="3d")
    ax5 = fig.add_subplot(325, projection="3d")
    ax6 = fig.add_subplot(326, projection="3d")


    fig, ax1 = plot_3D_directivity(
        p_scattered, theta_m, phi_m, title='blade scattered noise (SM)', fig=fig, ax=ax1, valmin=VMIN, valmax=VMAX,
    )
    fig, ax2 = plot_3D_directivity(
        p_vortex_beam, theta_m, phi_m, title=r'beam direct noise ($\rho\nabla(\omega \times v)$)', fig=fig, ax=ax2, valmin=VMIN, valmax=VMAX,
    )
    fig, ax3 = plot_3D_directivity(
        p_dynamic_beam, theta_m, phi_m, title=r'beam direct noise ($1/2\rho\nabla^2(|u|^2)$)', fig=fig, ax=ax3, valmin=VMIN, valmax=VMAX,
    )
    fig, ax4 = plot_3D_directivity(
        p_vortex_beam + p_dynamic_beam, theta_m, phi_m,
        title=r'beam direct noise (total)', fig=fig, ax=ax4, valmin=VMIN, valmax=VMAX,
    )
    fig, ax5 = plot_3D_directivity(
        p_scattered + p_dynamic_beam, theta_m, phi_m,
        title=r'PIN dynamic + scattered loading', fig=fig, ax=ax5, valmin=VMIN, valmax=VMAX,
    )
    fig, ax6 = plot_3D_directivity(
        p_direct_beam, theta_m, phi_m,
        title='beam total (reference)', fig=fig, ax=ax6, valmin=VMIN, valmax=VMAX,
    )
    plt.show()

        # 3D phase
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(321, projection="3d")
    ax2 = fig.add_subplot(322, projection="3d")
    ax3 = fig.add_subplot(323, projection="3d")
    ax4 = fig.add_subplot(324, projection="3d")
    ax5 = fig.add_subplot(325, projection="3d")
    ax6 = fig.add_subplot(326, projection="3d")


    fig, ax1 = plot_3D_phase_directivity(
        p_scattered, theta_m, phi_m, title='blade scattered noise', fig=fig, ax=ax1, valmin=VMIN, valmax=VMAX,
    )
    fig, ax2 = plot_3D_phase_directivity(
        p_vortex_beam, theta_m, phi_m, title='beam noise (vortex)', fig=fig, ax=ax2, valmin=VMIN, valmax=VMAX,
    )
    fig, ax3 = plot_3D_phase_directivity(
        p_dynamic_beam, theta_m, phi_m, title='beam noise (dynamic)', fig=fig, ax=ax3, valmin=VMIN, valmax=VMAX,
    )
    # fig, ax4 = plot_3D_directivity(
    #     p_direct_beam + p_direct_blade + p_scattered, theta_m, phi_m,
    #       title='total loading noise', fig=fig, ax=ax4, valmin=VMIN, valmax=VMAX,
    # )
    fig, ax4 = plot_3D_phase_directivity(
        p_vortex_beam + p_dynamic_beam, theta_m, phi_m,
        title='blade noise (total)', fig=fig, ax=ax4, valmin=VMIN, valmax=VMAX,
    )

    fig, ax5 = plot_3D_phase_directivity(
        p_scattered + p_dynamic_beam, theta_m, phi_m,
        title='direct blade + scattered blade', fig=fig, ax=ax5, valmin=VMIN, valmax=VMAX,
    )

    fig, ax6 = plot_3D_phase_directivity(
        p_direct_beam, theta_m, phi_m,
        title='direct blade + direct beam', fig=fig, ax=ax6, valmin=VMIN, valmax=VMAX,
    )
    plt.show()

