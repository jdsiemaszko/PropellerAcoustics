import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os 
from Constants.data_assim import getGojonData, getHarmonicsFromData
from Constants.helpers import plot_directivity_contour, plot_phase_directivity_contour, p_to_SPL, read_force_file

components = [
        # ==========================================================
        # 1) Loading contributions
        # ==========================================================
        {
            "name": "loading_scattering",
            "title": "Loading contribution",
            'color' : 'r',
            'linestyle': 'dashed',
            'marker' : 's',
            # 'label' : ''
        },

        # ==========================================================
        # 2) Thickness contributions
        # ==========================================================
        {
            "name": "thickness_scattering",
            "title": "Thickness contribution",
            'color' : 'b',
            'linestyle': 'dashed',
            'marker' : 's',
        },
        {
            "name": "total_scattering",
            'color' : 'k',
            'linestyle': 'dashed',
            'marker' : 's',
        },

                # ==========================================================
        # 1) Loading contributions
        # ==========================================================
        {
            "name": "loading_scattering_cp",
            'color' : 'r',
            'linestyle': 'dotted',
            'marker' : 's',
            # 'label' : ''
        },

        # ==========================================================
        # 2) Thickness contributions
        # ==========================================================
        {
            "name": "thickness_scattering_cp",
            'color' : 'b',
            'linestyle': 'dotted',
            'marker' : 's',
        },

        {
            "name": "total_scattering_cp",
            'color' : 'k',
            'linestyle': 'dotted',
            'marker' : 's',
        },



            {
            "name": "loading_PIN",
        'color' : 'r',
            'linestyle': 'solid',
            'marker' : '^',
        },        {
            "name": "thickness_PIN",
                    'color' : 'b',
            'linestyle': 'solid',
            'marker' : '^',
        },
        {
            "name": "nonlinear_PIN",
                'color' : 'm',
            'linestyle': 'solid',
            'marker' : '^',
        },
        {
            "name": "total_PIN",
                    'color' : 'k',
            'linestyle': 'solid',
            'marker' : '^',
        },

            {
            "name": "total_plus_nolinear_PIN",
            'color' : 'c',
            'linestyle': 'solid',
            'marker' : '^',
        },



    ]

SUFFIX = 'D20L20_D180_v2'
mplot = 5
index_m = mplot-1
# folder_name = f"./Figures/SurfacePressureComponents_{SUFFIX}_M{mplot}_RdBu"
# folder_name = os.path.join(os.curdir, 'Figures', f"SurfacePressureComponents_{SUFFIX}_M{mplot}_RdBu")


folder_name = f'./Data/current/surface_pressure/'

rtip, rroot = 0.1, 0.1*0.16

VMIN, VMAX = 80, 130
levels = np.linspace(VMIN, VMAX, 21)
levels_phase = np.linspace(-np.pi, np.pi, 21)


mappables = {}
reference_xrange = rtip - rroot
reference_width = 3.0
reference_height = 3.0
reference_width_long = reference_width * 2 * rtip / reference_xrange
RADIUS = 0.01


for index_comp, comp in enumerate(components):
    period = 2*np.pi

    # TH = np.load(os.path.join(folder_name, f"th_surface_{comp['name']}.npy"))
    # PHI = np.load(os.path.join(folder_name, f"phi_surface_{comp['name']}.npy"))
    # Z = np.load(os.path.join(folder_name, f"p_surface_{comp['name']}.npy"))


    Z  = np.load(os.path.join(folder_name, f"Z_{index_m}_{index_comp}.npy"))
    PHI  = np.load(os.path.join(folder_name, f"PHI_{index_m}_{index_comp}.npy"))
    TH = np.load(os.path.join(folder_name, f"TH_{index_m}_{index_comp}.npy"))


    print(f'component {comp['name']}, max SPL: {p_to_SPL(Z).max()}dB')

    HEIGHT = reference_height

    xrange = PHI.max() - PHI.min()

    WIDTH = reference_width * xrange / reference_xrange

    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))

    fig, ax, mappable = plot_directivity_contour(
        Phi=PHI,
        Theta=TH,
        magnitudes=Z,
        xlabel="$r$ [m]",
        ylabel=r"$\theta$ [rad]",
        # title=comp["title"],
        levels=levels,
        fig=fig,
        ax=ax,
        cmap='jet'
    )
    ax.set_ylim(0, 2 * np.pi)

    # ax.set_xlim(rroot, rtip)
    ax.set_aspect((2 * np.pi / (rtip - rroot))**(-1))
    if PHI.max() > rtip:
        ax.axvline(rroot, color='white', linestyle='dashed', linewidth=3)
        ax.axvline(rtip, color='white', linestyle='dashed', linewidth=3)
        ax.set_xticks(np.linspace(0, 2*rtip, 11))
    else:
        ax.set_xticks(np.linspace(0, rtip, 6))
    ax.set_xlim(PHI.min(),PHI.max())


    # store mappable for colorbar later
    mappables[comp["name"]] = mappable

    # save the data for easy plotting later
    np.save(os.path.join(folder_name, f"p_surface_{comp['name']}.npy"), Z)
    np.save(os.path.join(folder_name, f"th_surface_{comp['name']}.npy"), TH)
    np.save(os.path.join(folder_name, f"PHI_surface_{comp['name']}.npy"), PHI)

    fig.tight_layout()
    plt.show()

    fig.savefig(
        os.path.join(folder_name, f"p_surface_{comp['name']}.pdf"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # fig2, ax2 = plt.subplots(figsize=(WIDTH, HEIGHT))

    # fig2, ax2, mappable_phase = plot_phase_directivity_contour(
    #     Phi=PHI,
    #     Theta=TH,
    #     magnitudes=Z,
    #     xlabel="$r$ [m]",
    #     ylabel=r"$\theta$ [rad]",
    #     # title=comp["title"],
    #     levels=levels_phase,
    #     fig=fig2,
    #     ax=ax2
    # )
    # ax2.set_ylim(0, 2 * np.pi)


    # # ax.set_xlim(rroot, rtip)
    # ax2.set_aspect((2 * np.pi / (rtip - rroot))**(-1))
    # if PHI.max() > rtip:
    #     ax.axvline(rroot, color='white', linestyle='dashed')
    #     ax.axvline(rtip, color='white', linestyle='dashed')
    # plt.tight_layout()

    # fig2.savefig(
    #     os.path.join(folder_name, f"p_surface_{comp['name']}_phase.pdf"),
    #     dpi=300,
    #     bbox_inches="tight",
    # )
    # plt.close(fig2)

fig = plt.figure(figsize=(1, 2.5))
cax = fig.add_axes([0.35, 0.05, 0.3, 0.9])

cbar = fig.colorbar(
    mappables["loading_scattering"],
    cax=cax,
)

cbar.set_label("SPL [dB]")
cbar.set_ticks(np.arange(VMIN, VMAX+1, 10))

fig.savefig(
    os.path.join(folder_name, "colorbar_spl.pdf"),
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# fig = plt.figure(figsize=(1.5, 5))

# cax = fig.add_axes([0.35, 0.05, 0.3, 0.9])

# cbar = fig.colorbar(
#     mappable_phase,
#     cax=cax,
# )

# cbar.set_label("SPL [dB]")

# fig.savefig(
#     os.path.join(folder_name, "colorbar_phase.pdf"),
#     dpi=300,
#     bbox_inches="tight",
# )
# plt.close(fig)