
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from Constants.data_assim import getGojonData, getHarmonicsFromData
from PotentialInteraction.PIN import PotentialInteraction
from Constants.helpers import read_force_file, plot_3D_directivity, plot_3D_phase_directivity, plot_beam_azimuth, plot_rotation_arrow

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

MODE = 'half'
FILE = 'TOTAL_DIR'
Ntheta = 18
Nphi = 36
B=2

m_surface = np.arange(1, 11, 1)
ms = [5]
SUFFIX = 'D20L20_D180_v2'
shape='D'

import os
folder_name = f'./Figures/{SUFFIX}_{ms[0]}'
os.makedirs(folder_name, exist_ok=True)

components = [
    {
        "name": "pin_total_loading_only",
        "title": "PIN Model (vortex only)",
    },

    {
        "name": "scattering_total",
        "title": "Scattering Model",
    },
    {
        "name": "experiment",
        "title": "Experiment",
    },
    {
        "name": "pin_total",
        "title": "PIN Model (incl. blade thickness)",
    },

    # parts of the scattering model
        {
        "name": "direct_thickness",
        "title": r"$\langle G_0 Q\rangle_\Omega$",
    },
    {
        "name": "scattered_thickness",
        "title": r"$\langle G_s Q\rangle_\Omega$",
    },
    {
        "name": "direct_loading_steady",
        "title": r"$-\langle \boldsymbol{\nabla} G_0 \circ \boldsymbol{F}_0\rangle_\Omega$",
    },
    {
        "name": "scattered_loading_steady",
        "title": r"$-\langle \boldsymbol{\nabla} G_s \circ \boldsymbol{F}_0\rangle_\Omega$",
    },
    {
        "name": "direct_loading_unsteady",
        "title": r"$-\sum_{k>0}\langle \boldsymbol{\nabla} G_0 \circ \boldsymbol{F}_k\rangle_\Omega$",
    },
    {
        "name": "scattered_loading_unsteady",
        "title": r"$-\sum_{k>0}\langle \boldsymbol{\nabla} G_s \circ \boldsymbol{F}_k\rangle_\Omega$",
    },

            {
        "name": "scattered_loading",
        "title": r"$-\sum_{k>0}\langle \boldsymbol{\nabla} G_s \circ \boldsymbol{F}_k\rangle_\Omega$",
    },



    {       "name": "compact_scattered_loading_steady",
        "title": r"$-\langle \boldsymbol{\nabla} G_s \circ \boldsymbol{F}_0\rangle_\Omega$",
    },
    {
        "name": "compact_scattered_loading_unsteady",
        "title": r"$-\sum_{k>0}\langle \boldsymbol{\nabla} G_s \circ \boldsymbol{F}_k\rangle_\Omega$",
    },

            {
        "name": "compact_scattered_loading",
        "title": r"$-\sum_{k>0}\langle \boldsymbol{\nabla} G_s \circ \boldsymbol{F}_k\rangle_\Omega$",
    },
            {
        "name": "compact_scattered_thickness",
        "title": r"$\langle G_s Q\rangle_\Omega$",
    },

        {
        "name": "pin_nonlinear",
        "title": r"1/2rhov^2",
    },

                {
        "name": "pin_beam_loading_only",
        "title": r"1/2rhov^2",
    },

                    {
    "name": "pin_beam_thickness_only",
        "title": r"1/2rhov^2",
    },
]

VMIN, VMAX = 35, 65
harmonic = int(ms[0] * B)
R0 = 1.4
R1 = R0 * 1.1

for comp in components:
    print(comp['name'])

    # save data for component first
    comp["data"] = np.load(os.path.join(folder_name, f"data_directivities_{comp['name']}.npy"))
    comp["theta"] = np.load(os.path.join(folder_name, f"theta_directivities_{comp['name']}.npy"))
    comp["phi"] = np.load(os.path.join(folder_name, f"phi_directivities_{comp['name']}.npy"))

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
    plot_rotation_arrow(R1, PHI_EXTENT=[10, 80], fig=fig, ax=ax)
    plt.show()

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
    plot_rotation_arrow(R1, PHI_EXTENT=[10, 80], fig=fig, ax=ax)

    fig.savefig(
        os.path.join(
            folder_name,
            f"directivity_phase_{comp['name']}.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


fig = plt.figure(figsize=(1, 2.5))
cax = fig.add_axes([0.35, 0.05, 0.3, 0.9])

cbar = fig.colorbar(
    SPL_mappable,
    cax=cax,
)

cbar.set_label("SPL [dB w.r.t. 20e-6 Pa]")
cbar.set_ticks(np.arange(VMIN, VMAX+1, 10))

fig.savefig(
    os.path.join(folder_name, "colorbar_spl_small.pdf"),
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


fig = plt.figure(figsize=(1, 2.5))
cax = fig.add_axes([0.35, 0.05, 0.3, 0.9])

cbar = fig.colorbar(
    phase_mappable,
    cax=cax,
)

ticks = np.arange(-np.pi, np.pi + 1e-10, np.pi/2)
labels = [
    r"$-\pi$",
    r"$-\pi/2$",
    r"$0$",
    r"$\pi/2$",
    r"$\pi$",
]

cbar.set_ticks(ticks)
cbar.set_ticklabels(labels)

cbar.set_label("Phase [rad]")

fig.savefig(
    os.path.join(folder_name, "colorbar_phase_small.pdf"),
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)