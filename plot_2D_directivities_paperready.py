import os

import numpy as np
import matplotlib.pyplot as plt

from Constants.helpers import p_to_SPL


# ============================================================
# Configuration
# ============================================================

SUFFIX = 'D20L20_D180_v2'

ms = [5]

folder_name = f'./Figures/{SUFFIX}_{ms[0]}'

os.makedirs(folder_name, exist_ok=True)

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


# ============================================================
# Select components to plot
# ============================================================

PLOT_COMPONENTS = [
    # 'experiment',
    'scattering_total',
    'pin_total',
]

MARKERS = [
    's', '^'
]
COLORS = [
    'b', 'r'
]

LINESTYLES=[
    'dashed',
    'dotted'
]

# ============================================================
# Select reference azimuth station
# ============================================================

# Column index in the (Ntheta, Nphi) arrays
phi_index = 1


# ============================================================
# Plot settings
# ============================================================

VMIN, VMAX = 35, 65

component_titles = {
    "pin_total_loading_only": "PIN Model (vortex only)",
    "scattering_total": "Scattering Model",
    "experiment": "Experiment",
    "pin_total": "PIN Model (incl. blade thickness)",
    "direct_thickness": r"$\langle G_0 Q\rangle_\Omega$",
    "scattered_thickness": r"$\langle G_s Q\rangle_\Omega$",
    "direct_loading_steady":
        r"$-\langle \boldsymbol{\nabla}G_0\circ\boldsymbol{F}_0\rangle_\Omega$",
    "scattered_loading_steady":
        r"$-\langle \boldsymbol{\nabla}G_s\circ\boldsymbol{F}_0\rangle_\Omega$",
    "direct_loading_unsteady":
        r"$-\sum_{k>0}\langle \boldsymbol{\nabla}G_0\circ\boldsymbol{F}_k\rangle_\Omega$",
    "scattered_loading_unsteady":
        r"$-\sum_{k>0}\langle \boldsymbol{\nabla}G_s\circ\boldsymbol{F}_k\rangle_\Omega$",
    "scattered_loading":
        r"$-\sum_{k>0}\langle \boldsymbol{\nabla}G_s\circ\boldsymbol{F}_k\rangle_\Omega$",
    "compact_scattered_loading_steady":
        r"$-\langle \boldsymbol{\nabla}G_s\circ\boldsymbol{F}_0\rangle_\Omega$",
    "compact_scattered_loading_unsteady":
        r"$-\sum_{k>0}\langle \boldsymbol{\nabla}G_s\circ\boldsymbol{F}_k\rangle_\Omega$",
    "compact_scattered_loading":
        r"$-\sum_{k>0}\langle \boldsymbol{\nabla}G_s\circ\boldsymbol{F}_k\rangle_\Omega$",
    "compact_scattered_thickness":
        r"$\langle G_s Q\rangle_\Omega$",
    "pin_nonlinear": r"$1/2\rho v^2$",
    "pin_beam_loading_only": r"$1/2\rho v^2$",
    "pin_beam_thickness_only": r"$1/2\rho v^2$",
}


# ============================================================
# Load first component to determine angular grid
# ============================================================

theta = np.load(
    os.path.join(
        folder_name,
        f"theta_directivities_{PLOT_COMPONENTS[0]}.npy"
    )
)

phi = np.load(
    os.path.join(
        folder_name,
        f"phi_directivities_{PLOT_COMPONENTS[0]}.npy"
    )
)

assert theta.ndim == 2
assert phi.ndim == 2

Ntheta, Nphi = theta.shape

assert phi_index < Nphi

print(f"Ntheta = {Ntheta}")
print(f"Nphi   = {Nphi}")
print(f"Reference phi index = {phi_index}")


# ============================================================
# Reference azimuth
# ============================================================

phi_ref = phi[0, phi_index]

print(
    f"Reference azimuth = "
    f"{np.rad2deg(phi_ref):.2f} deg"
)


# ============================================================
# Create polar plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(6, 6),
    subplot_kw={"projection": "polar"},
)


for component_name, MARKER, COLOR, LINESTYLE in zip(PLOT_COMPONENTS, MARKERS, COLORS, LINESTYLES):

    print(f"\nLoading {component_name}")

    data = np.load(
        os.path.join(
            folder_name,
            f"data_directivities_{component_name}.npy"
        )
    )

    theta_comp = np.load(
        os.path.join(
            folder_name,
            f"theta_directivities_{component_name}.npy"
        )
    )

    phi_comp = np.load(
        os.path.join(
            folder_name,
            f"phi_directivities_{component_name}.npy"
        )
    )

    assert theta_comp.ndim == 2
    assert phi_comp.ndim == 2

    # Data are stored as a flattened array
    data = data.reshape(theta_comp.shape)

    Ntheta_comp, Nphi_comp = theta_comp.shape

    # ========================================================
    # Reference azimuth station
    # ========================================================

    # Use the selected column as the reference station
    theta_station_1 = theta_comp[:, phi_index]
    sweep_1 = data[:, phi_index]

    phi_ref_comp = phi_comp[0, phi_index]

    # ========================================================
    # Find ACTUAL station at phi_ref + pi
    # ========================================================

    phi_target = np.mod(
        phi_ref_comp + np.pi,
        2 * np.pi
    )

    # One azimuth value for each column
    phi_stations = phi_comp[0, :]

    # Periodic angular difference
    phi_error = np.angle(
        np.exp(
            1j * (phi_stations - phi_target)
        )
    )

    # Closest actual measurement station
    phi_index_2 = np.argmin(
        np.abs(phi_error)
    )

    phi_2 = phi_stations[phi_index_2]

    print(
        f"  phi_ref      = "
        f"{np.rad2deg(phi_ref_comp):8.3f} deg"
    )

    print(
        f"  phi_ref + pi = "
        f"{np.rad2deg(phi_target):8.3f} deg"
    )

    print(
        f"  actual phi_2  = "
        f"{np.rad2deg(phi_2):8.3f} deg "
        f"(index {phi_index_2})"
    )

    # ========================================================
    # First branch
    # ========================================================

    theta_1 = np.mod(
        theta_station_1,
        2 * np.pi
    )

    order_1 = np.argsort(theta_1)

    theta_1 = theta_1[order_1]

    sweep_1 = sweep_1[order_1]

    # ========================================================
    # Second branch
    #
    # Use the ACTUAL data at phi_ref + pi.
    # Do not shift theta or reuse the first sweep.
    # ========================================================

    theta_station_2 = theta_comp[:, phi_index_2]

    sweep_2 = data[:, phi_index_2]

    theta_2 = np.mod(
        theta_station_2,
        2 * np.pi
    )

    order_2 = np.argsort(theta_2)

    theta_2 = theta_2[order_2]

    sweep_2 = sweep_2[order_2]

    # ========================================================
    # Plot the two branches independently
    # ========================================================

    label = component_titles.get(
        component_name,
        component_name
    )

    theta_combined = np.concatenate([
    theta_1,
    -theta_2,
    ])

    sweep_combined = np.concatenate([
        sweep_1,
        sweep_2,
    ])

    # Sort by polar angle
    order = np.argsort(theta_combined)

    theta_combined = theta_combined[order]
    sweep_combined = sweep_combined[order]

    # Close the curve by repeating the first point
    theta_combined = np.concatenate([
        theta_combined,
        [theta_combined[0]],
    ])

    sweep_combined = np.concatenate([
        sweep_combined,
        [sweep_combined[0]],
    ])

    line, = ax.plot(
        theta_combined,
        p_to_SPL(sweep_combined),
        linewidth=2,
        label=label,
        marker=MARKER,
        linestyle=LINESTYLE,
        color=COLOR
    )


# ============================================================
# Polar formatting
# ============================================================

ax.set_theta_zero_location("N")

ax.set_theta_direction(1)

ax.set_ylim(
    VMIN,
    VMAX
)

ax.set_yticks(
    np.arange(
        VMIN,
        VMAX + 1,
        10
    )
)

ax.legend(
    loc="upper right",
    fontsize=10,
)


# ============================================================
# Save
# ============================================================

outfile = os.path.join(
    folder_name,
    f"polar_station_{phi_index}.pdf"
)

fig.savefig(
    outfile,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

plt.close(fig)

print(f"\nSaved: {outfile}")