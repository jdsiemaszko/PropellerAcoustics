
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from Constants.data_assim import getGojonData, getHarmonicsFromData
from Constants.helpers import plot_directivity_contour, plot_phase_directivity_contour, p_to_SPL, read_force_file
from SourceMode.Configurations_NACA0012 import m_surface



# from SourceMode.Configurations_NACA0012 import D20L20W00_D180 as sourceArray
# SUFFIX = '_D20L20W20_D180'

from SourceMode.Configurations_NACA0012 import D20L20W00_D360 as sourceArray
SUFFIX = '_D360_HR'

# sourceArray.numerics['CompactnessCorrection'] = True
sourceArray.numerics['CompactnessCorrection'] = False


MODE = 'half'
FILE = 'TOTAL_DIR'

mss = np.arange(1, 6, 1)

for m in mss:
    print(f'saving diagrams for m={m}')
    ms = np.array([m])

    B = sourceArray.B
    NDIPOLES = sourceArray.Nsources

    r_inner, Fz, Fphi  = read_force_file('./Data/Zamponi2026/FS_ISAE_2_8000.txt') # reuse the radial stations from data

    D_bras = sourceArray.green.radius * 2
    g = -1 * sourceArray.green.origin[2]
    BLH, BLH_S, BLH_US, _ = sourceArray.getLoading(Fz, Fphi, D_bras, g, steady_only=False) # compute loading on the fly, return PIN for reuse
    PIN = sourceArray.PIN


    # save gradients in the far-field (run once per observer and m)

    mplot = ms[0]
    ind_m = np.where(m_surface == ms[0])[0][0]


    gradG_surface = np.zeros((sourceArray.Nchildren, 3, m_surface.shape[0], sourceArray.green.getBoundaryPoints().shape[1] ,NDIPOLES,))
    G_surface = np.zeros((sourceArray.Nchildren, m_surface.shape[0], sourceArray.green.getBoundaryPoints().shape[1] ,NDIPOLES,))

    for index, sm in enumerate(sourceArray.children):

        gradG_surface[index, :, :, :, :] = np.load(f'./Data/current/NACA0012_rotor/gradG_surface_sm_{index}_{MODE}{SUFFIX}.npy') # shape (3, Nm, Nz, Nyi)
        print(f'pre-computing far-field gradients {index+1}')

        G_surface[index, :, :, :] = np.load(f'./Data/current/NACA0012_rotor/G_surface_sm_{index}_{MODE}{SUFFIX}.npy') # shape (Nm, Nz, Nyi)
        print(f'pre-computing far-field G {index+1}')


    print(f'parsed surface data')



    # manual computation!
    pmB_loading = sourceArray.getPressure(sourceArray.green.getBoundaryPoints(), mplot, gradG_surface[:, :, ind_m:ind_m+1, :, :])
    pmB_thickness = sourceArray.getThicknessPressure(sourceArray.green.getBoundaryPoints(), mplot, G=G_surface[:, ind_m:ind_m+1, :, :])

    pmB_total = pmB_loading + pmB_thickness



    index = np.where(PIN.k == mplot * B)[0][0] # index of the desired mode 

    PIN._numerics['only_linear'] = True 
    PIN._numerics['only_nonlinear'] = False
    PIN._numerics['include_vortex_sources'] = True
    PIN._numerics['include_thickness_sources'] = False
    p_PIN_loading = PIN.getStrutPressureHarmonics()[:, index, :]

    PIN._numerics['only_linear'] = True 
    PIN._numerics['only_nonlinear'] = False
    PIN._numerics['include_vortex_sources'] = False
    PIN._numerics['include_thickness_sources'] = True
    p_PIN_thickness = PIN.getStrutPressureHarmonics()[:, index, :]

    PIN._numerics['only_linear'] = False 
    PIN._numerics['only_nonlinear'] = True
    PIN._numerics['include_vortex_sources'] = True
    PIN._numerics['include_thickness_sources'] = True
    p_PIN_nonlinear = PIN.getStrutPressureHarmonics()[:, index, :]

    PIN._numerics['only_linear'] = True
    PIN._numerics['only_nonlinear'] = False
    PIN._numerics['include_vortex_sources'] = True
    PIN._numerics['include_thickness_sources'] = True
    p_PIN_total = PIN.getStrutPressureHarmonics()[:, index, :]


    PIN._numerics['only_linear'] = False
    PIN._numerics['only_nonlinear'] = False
    PIN._numerics['include_vortex_sources'] = True
    PIN._numerics['include_thickness_sources'] = True
    p_PIN_total_incl_nonlinear = PIN.getStrutPressureHarmonics()[:, index, :]

    # if does not exist, will throw an error, which is fin
    # pmB_total = pmB_loading

    z_edges = sourceArray.green.panel_z_edges
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    th_edges = sourceArray.green.panel_th_edges
    th_centers = (th_edges[:-1] + th_edges[1:]) / 2
    # TH, PHI = np.meshgrid(th_centers, z_centers, indexing='ij')
    PHI, TH = np.meshgrid(z_centers, th_centers, indexing='ij')


    # pin range

    thetab = PIN.theta_beam
    radius = PIN.seg_radius
    TH_PIN, PHI_PIN = np.meshgrid(thetab, radius, indexing='ij')

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    folder_name = f"./Figures/SurfacePressureComponents_M{mplot}"
    os.makedirs(folder_name, exist_ok=True)


    components = [
        # ==========================================================
        # 1) Loading contributions
        # ==========================================================
        {
            "name": "loading_scattering",
            "title": "Loading contribution",
            "data": pmB_loading[:, 0],
            'theta' : TH,
            'phi' : PHI,
        },

        # ==========================================================
        # 2) Thickness contributions
        # ==========================================================
        {
            "name": "thickness_scattering",
            "title": "Thickness contribution",
            "data": pmB_thickness[:, 0],
            'theta' : TH,
            'phi' : PHI,
        },

        {
            "name": "total_scattering",
            "title": "Total model",
            "data": pmB_total[:, 0],
                    'theta' : TH,
            'phi' : PHI,
        },

            {
            "name": "loading_PIN",
            "data": p_PIN_loading,
                    'theta' : TH_PIN,
            'phi' : PHI_PIN,
        },        {
            "name": "thickness_PIN",
            "data": p_PIN_thickness,
            'theta' : TH_PIN,
            'phi' : PHI_PIN,
        },
        {
            "name": "nonlinear_PIN",
            "data": p_PIN_nonlinear,
            'theta' : TH_PIN,
            'phi' : PHI_PIN,
        },
        {
            "name": "total_PIN",
            "data": p_PIN_total,
            'theta' : TH_PIN,
            'phi' : PHI_PIN,
        },

            {
            "name": "total_plus_nolinear_PIN",
            "data": p_PIN_total_incl_nonlinear,
            'theta' : TH_PIN,
            'phi' : PHI_PIN,
        },



    ]

    rtip, rroot = sourceArray.r1, sourceArray.r0

    VMIN, VMAX = 80, 130
    levels = np.linspace(VMIN, VMAX, 21)
    levels_phase = np.linspace(-np.pi, np.pi, 21)


    mappables = {}
    reference_xrange = rtip - rroot
    reference_width = 6.0
    reference_height = 4.0

    for comp in components:

        Z = comp['data']
        TH = comp['theta']
        PHI = comp['phi']
        print(f'component {comp['name']}, max SPL: {p_to_SPL(Z).max()}dB')

        HEIGHT = reference_height

        xrange = PHI.max() - PHI.min()

        WIDTH = reference_width * xrange / reference_xrange

        fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))

        fig, ax, mappable = plot_directivity_contour(
            Phi=PHI,
            Theta=np.rad2deg(TH),
            magnitudes=Z,
            xlabel="z [m]",
            ylabel=r"$\theta$ [deg]",
            # title=comp["title"],
            levels=levels,
            fig=fig,
            ax=ax
        )

        # ax.set_xlim(rroot, rtip)
        ax.set_aspect((360 / (rtip - rroot))**(-1))
        if PHI.max() > rtip:
            ax.axvline(rroot, color='k', alpha=0.7)
            ax.axvline(rtip, color='k', alpha=0.7)

        # store mappable for colorbar later
        mappables[comp["name"]] = mappable

        fig.savefig(
            os.path.join(folder_name, f"p_surface_{comp['name']}.pdf"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(WIDTH, HEIGHT))

        fig2, ax2, mappable_phase = plot_phase_directivity_contour(
            Phi=PHI,
            Theta=np.rad2deg(TH),
            magnitudes=Z,
            xlabel="z [m]",
            ylabel=r"$\theta$ [deg]",
            # title=comp["title"],
            levels=levels_phase,
            fig=fig2,
            ax=ax2
        )

        # ax.set_xlim(rroot, rtip)
        ax2.set_aspect((360 / (rtip - rroot))**(-1))
        if PHI.max() > rtip:
            ax2.axvline(rroot, color='k', alpha=0.7)
            ax2.axvline(rtip, color='k', alpha=0.7)

        fig2.savefig(
            os.path.join(folder_name, f"p_surface_{comp['name']}_phase.pdf"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig2)

    fig = plt.figure(figsize=(1.5, 5))

    cax = fig.add_axes([0.35, 0.05, 0.3, 0.9])

    cbar = fig.colorbar(
        mappables["loading_scattering"],
        cax=cax,
    )

    cbar.set_label("SPL [dB]")

    fig.savefig(
        os.path.join(folder_name, "colorbar_spl.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig = plt.figure(figsize=(1.5, 5))

    cax = fig.add_axes([0.35, 0.05, 0.3, 0.9])

    cbar = fig.colorbar(
        mappable_phase,
        cax=cax,
    )

    cbar.set_label("SPL [dB]")

    fig.savefig(
        os.path.join(folder_name, "colorbar_phase.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)