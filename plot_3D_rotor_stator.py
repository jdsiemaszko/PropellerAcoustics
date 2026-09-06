import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import rcParams
from matplotlib import cm, colors
import matplotlib
from Constants.helpers import p_to_SPL
# ============================================================
# Propeller above a circular stator, line-art style for paper
# ============================================================

plt.close("all")

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------
view_az = -45
view_el = 34
annotate = True
dpi = 600

fig_w = 25.5 * 2 / 2.54  # cm -> inches
fig_h = 18.0 * 2 / 2.54

# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------

# Hub
hub_radius = 0.56
hub_height = 0.4

# Blades
num_blades = 2
blade_radius = 3.5
blade_chord = 3.5 * 0.25
pitch_angle = -10.0       # degrees
t_max = 0.12
root_blend = 0.2
blade_phase = -50.0+10+15+20        # degrees

# Annular strip
strip_radius = 0.68 * blade_radius
strip_width = 0.22

# Stator
stator_radius = 0.42
stator_length = 2 * blade_radius
stator_center = np.array([0.0, 0.0, 0.0])

# Placement
gap = blade_chord * 0.02/0.025
hub_center_z = (
    stator_center[2]
    + gap
)

# ------------------------------------------------------------
# STYLE
# ------------------------------------------------------------

face_light = (0.93, 0.93, 0.95)
face_cap = (0.90, 0.90, 0.93)
face_side = (0.84, 0.84, 0.88)

edge_col = (0.0, 0.0, 0.0)

lw_out = 2.0
lw_arrow = 2.5
lw_thin = 1.2

omega_col = (0.45, 0.45, 0.45)

fsz = 11

# Times New Roman if installed
rcParams["font.family"] = "Times New Roman"

# ------------------------------------------------------------
# FIGURE SETUP
# ------------------------------------------------------------

fig = plt.figure(
    figsize=(fig_w, fig_h),
    facecolor="white"
)

ax = fig.add_axes([0, 0, 1, 1], projection="3d",     computed_zorder=False)

ax.set_box_aspect((1, 1, 1))
ax.set_axis_off()

ax.view_init(
    elev=view_el,
    azim=view_az
)

# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

def add_patch3d(ax, vertices, facecolor, edgecolor=edge_col,
                linewidth=lw_out, zorder=0):
    """Add a filled polygon to a 3-D axes."""
    poly = Poly3DCollection(
        [vertices],
        facecolors=[facecolor],
        edgecolors=[edgecolor],
        linewidths=linewidth, zorder=zorder
    )
    ax.add_collection3d(poly)
    return poly


def plot3(ax, x, y, z, **kwargs):
    """MATLAB-like plot3 wrapper."""
    return ax.plot(
        np.asarray(x),
        np.asarray(y),
        np.asarray(z),
        **kwargs
    )[0]


# ------------------------------------------------------------
# DRAW STATOR
# ------------------------------------------------------------

# PARSE surface data

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
mplot = 1
index_m = mplot-1

# index_comp = 5 # SCATTERING TOTAL
index_comp=10 # PIN TOTAL

folder_name = f'./Data/current/surface_pressure/'
import os

Z  = np.load(os.path.join(folder_name, f"Z_{index_m}_{index_comp}.npy"))
Z = p_to_SPL(Z)
PHI  = np.load(os.path.join(folder_name, f"PHI_{index_m}_{index_comp}.npy"))
TH = np.load(os.path.join(folder_name, f"TH_{index_m}_{index_comp}.npy"))
stator_length = PHI.max()/0.1 * blade_radius
# tc = np.linspace(0, 2 * np.pi, 120)
out_pdf = f"./Figures/stator_colormap_3D_{index_m}_{index_comp}.pdf"

# s = np.array([0.0, stator_length])

# Yc = stator_radius * np.cos(tc)
# Zc = stator_radius * np.sin(tc)

# Xsurf = np.tile(s, (len(tc), 1)) + stator_center[0]
# Ysurf = np.tile(Yc[:, None], (1, 2)) + stator_center[1]
# Zsurf = np.tile(Zc[:, None], (1, 2)) + stator_center[2]

# ax.plot_surface(
#     Xsurf,
#     Ysurf,
#     Zsurf,
#     color=face_side,
#     edgecolor="none",
#     shade=False,
#     zorder=99
# )

# Xsurf = (
#     (PHI - PHI.min())
#     / (PHI.max() - PHI.min())
#     * stator_length
#     + stator_center[0]
# )
Xsurf = PHI * stator_length / PHI.max()

# TH -> circumferential coordinate
Ysurf = (
    stator_radius * np.cos(TH)
    + stator_center[1]
)

Zsurf = (
    stator_radius * np.sin(TH)
    + stator_center[2]
)

# ------------------------------------------------------------
# Colormap
# ------------------------------------------------------------

norm = colors.Normalize(
    vmin=100,
    vmax=125
)

cmap = matplotlib.colormaps.get_cmap("Blues")

facecolors = cmap(norm(Z))

# ------------------------------------------------------------
# Plot colored stator surface
# ------------------------------------------------------------

ax.plot_surface(
    Xsurf,
    Ysurf,
    Zsurf,
    facecolors=facecolors,
    edgecolor="none",
    shade=False,
    zorder=99
)

# ------------------------------------------------------------
# Colorbar
# ------------------------------------------------------------

mappable = cm.ScalarMappable(
    norm=norm,
    cmap=cmap
)
mappable.set_array(Z)

# cbar = fig.colorbar(
#     mappable,
#     ax=ax,
#     pad=0.05,
#     shrink=0.7
# )

# cbar.set_label(r"$Z$", fontsize=fsz)

# End caps
tc = np.linspace(0, 2 * np.pi, 120)
s = np.array([0.0, stator_length])
Yc = stator_radius * np.cos(tc)
Zc = stator_radius * np.sin(tc)
for sj in s:
    vertices = np.column_stack([
        np.full_like(tc, sj + stator_center[0]),
        Yc + stator_center[1],
        Zc + stator_center[2]
    ])

    add_patch3d(
        ax,
        vertices,
        face_cap,
        edgecolor=edge_col,
        linewidth=lw_out
    )

# Silhouette generatrices
az = np.deg2rad(view_az)
el = np.deg2rad(view_el)

vcam = np.array([
    np.cos(el) * np.cos(az),
    np.cos(el) * np.sin(az),
    np.sin(el)
])

phi_sil = np.arctan2(-vcam[1], vcam[2])

for pp in [phi_sil, phi_sil + np.pi]:

    plot3(
        ax,
        [s[0], s[-1]] + stator_center[0],
        stator_radius * np.cos(pp) * np.ones(2)
            + stator_center[1],
        stator_radius * np.sin(pp) * np.ones(2)
            + stator_center[2],
        color=edge_col,
        linewidth=lw_out
    )
    # pass

# ------------------------------------------------------------
# DRAW HUB
# ------------------------------------------------------------

# MATLAB cylinder() equivalent
theta_hub = np.linspace(0, 2 * np.pi, 60)

Xhub = hub_radius * np.cos(theta_hub)
Yhub = hub_radius * np.sin(theta_hub)

z0 = hub_center_z - hub_height / 2
z1 = hub_center_z + hub_height / 2

XH = np.tile(Xhub[:, None], (1, 2))
YH = np.tile(Yhub[:, None], (1, 2))
ZH = np.column_stack([
    np.full(len(theta_hub), z0),
    np.full(len(theta_hub), z1)
])

ax.plot_surface(
    XH,
    YH,
    ZH,
    color=face_light,
    edgecolor=None,
    shade=False,
    zorder=100
)

# Top cap
vertices_top = np.column_stack([
    Xhub,
    Yhub,
    np.full_like(Xhub, z1)
])

add_patch3d(
    ax,
    vertices_top,
    face_light,
    edgecolor=edge_col,
    linewidth=lw_out,
        zorder=101

)

# Bottom circular edge
plot3(
    ax,
    Xhub,
    Yhub,
    np.full_like(Xhub, z0),
    color=edge_col,
    linewidth=lw_thin,
        zorder=101

)

# Visible side generatrices
# phi_cam = np.deg2rad(view_az)

# for pe in [phi_cam, phi_cam + np.pi]:

#     plot3(
#         ax,
#         hub_radius * np.cos(pe) * np.ones(2),
#         hub_radius * np.sin(pe) * np.ones(2),
#         np.array([z0, z1]),
#         color=edge_col,
#         linewidth=lw_thin,
#         zorder=101

#     )


# ------------------------------------------------------------
# NACA 0012 PROFILE AND BLADES
# ------------------------------------------------------------

naca_points = 20

x = np.linspace(0, blade_chord, naca_points)

xc = x / blade_chord

yt = (
    5 * t_max * blade_chord
    * (
        0.2969 * np.sqrt(xc)
        - 0.1260 * xc
        - 0.3516 * xc**2
        + 0.2843 * xc**3
        - 0.1015 * xc**4
    )
)

Xprof = np.concatenate([x, x[::-1]]) - blade_chord / 2
Zprof = np.concatenate([yt, -yt[::-1]])

# Spanwise mesh
nspan = 32

Yspan = np.linspace(
    hub_radius - root_blend,
    blade_radius,
    nspan
)

# MATLAB:
# chord_scale = linspace(1.0, 1.0, nspan);
chord_scale = np.linspace(1.0, 1.0, nspan)

Xmesh = chord_scale[:, None] * Xprof[None, :]
Zmesh = chord_scale[:, None] * Zprof[None, :]
Ymesh = np.tile(Yspan[:, None], (1, len(Xprof)))

# Pitch rotation
pitchRad = np.deg2rad(pitch_angle)

Xrot = (
    Xmesh * np.cos(pitchRad)
    - Zmesh * np.sin(pitchRad)
)

Zrot = (
    Xmesh * np.sin(pitchRad)
    + Zmesh * np.cos(pitchRad)
)

Yrot = Ymesh

i_le = 0
i_te = naca_points - 1

# ------------------------------------------------------------
# BLADE-HUB JUNCTION
# ------------------------------------------------------------

# Since chord_scale is constant this is simply 1,
# but retain the interpolation structure of the MATLAB code.
cs_root = np.interp(
    hub_radius,
    Yspan,
    chord_scale
)

Xr0 = (
    Xprof * np.cos(pitchRad)
    - Zprof * np.sin(pitchRad)
)

Zr0 = (
    Xprof * np.sin(pitchRad)
    + Zprof * np.cos(pitchRad)
)

Yj = np.sqrt(
    np.maximum(
        hub_radius**2 - (cs_root * Xr0)**2,
        0
    )
)

cs_j = np.interp(
    Yj,
    Yspan,
    chord_scale,
    left=cs_root,
    right=cs_root
)

Xroot0 = 1.003 * cs_j * Xr0
Zroot0 = cs_j * Zr0
Yroot0 = 1.003 * Yj


# ------------------------------------------------------------
# DRAW BLADES
# ------------------------------------------------------------

for k in range(num_blades):

    theta = (
        np.deg2rad(blade_phase)
        + k * 2 * np.pi / num_blades
    )

    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

    # Flatten coordinates, rotate, reshape
    points = np.column_stack([
        Xrot.ravel(),
        Yrot.ravel(),
        Zrot.ravel()
    ])

    coords = points @ Rz.T

    coords[:, 2] += hub_center_z

    Xp = coords[:, 0].reshape(Xrot.shape)
    Yp = coords[:, 1].reshape(Yrot.shape)
    Zp = coords[:, 2].reshape(Zrot.shape)

    # Blade surface
    ax.plot_surface(
        Xp,
        Yp,
        Zp,
        color=face_light,
        edgecolor="none",
        shade=False,
        zorder=100
    )

    # Tip cap
    vertices_tip = np.column_stack([
        Xp[-1, :],
        Yp[-1, :],
        Zp[-1, :]
    ])

    add_patch3d(
        ax,
        vertices_tip,
        face_light,
        edgecolor=edge_col,
        linewidth=lw_out,
        zorder=100


    )

    # Leading edge
    plot3(
        ax,
        Xp[:, i_le],
        Yp[:, i_le],
        Zp[:, i_le],
        color=edge_col,
        linewidth=lw_out,
        zorder=100

    )

    # Trailing edge
    plot3(
        ax,
        Xp[:, i_te],
        Yp[:, i_te],
        Zp[:, i_te],
        color=edge_col,
        linewidth=lw_out,
        zorder=100

    )

    # Blade-root / hub intersection
    root_points = np.column_stack([
        Xroot0,
        Yroot0,
        Zroot0
    ])

    root = root_points @ Rz.T

    plot3(
        ax,
        root[:, 0],
        root[:, 1],
        root[:, 2] + hub_center_z,
        color=edge_col,
        linewidth=lw_thin,
        alpha=0.5,
        zorder=101

    )


# ------------------------------------------------------------
# ROTOR DISC
# ------------------------------------------------------------

h_disc = plot3(
    ax,
    blade_radius * np.cos(tc),
    blade_radius * np.sin(tc),
    np.full_like(tc, hub_center_z),
    color=(0.45, 0.45, 0.45),
    linewidth=2.4,
    linestyle="-",
    zorder=100
)

# Matplotlib supports alpha directly
h_disc.set_alpha(0.55)


# ------------------------------------------------------------
# ROTATION DIRECTION ARROW
# ------------------------------------------------------------

z_om = hub_center_z + 0.9
r_om = 0.75

# Curved arrow:
# increasing theta = counter-clockwise viewed from +z
t_om = np.deg2rad(
    np.linspace(-55+30, 235+30, 100)
)

x_om = r_om * np.cos(t_om)
y_om = r_om * np.sin(t_om)
z_om_vec = np.full_like(t_om, z_om)

# Arrow curve
plot3(
    ax,
    x_om,
    y_om,
    z_om_vec,
    color=edge_col,
    linewidth=lw_arrow,
    zorder=101
)

# ------------------------------------------------------------
# ARROWHEAD
# ------------------------------------------------------------

P = np.array([
    x_om[-1],
    y_om[-1],
    z_om_vec[-1]
])

T = np.array([
    -np.sin(t_om[-1]),
    np.cos(t_om[-1]),
    0
])

T /= np.linalg.norm(T)

h = 0.22
alpha = np.deg2rad(25)

radial_inward = np.array([
    -np.cos(t_om[-1]),
    -np.sin(t_om[-1]),
    0
])

Q1 = P - h * (
    np.cos(alpha) * T
    + np.sin(alpha) * radial_inward
)

Q2 = P - h * (
    np.cos(alpha) * T
    - np.sin(alpha) * radial_inward
)

plot3(
    ax,
    [Q1[0], P[0]],
    [Q1[1], P[1]],
    [Q1[2], P[2]],
    color=edge_col,
    linewidth=lw_arrow,
        zorder=101
)

plot3(
    ax,
    [Q2[0], P[0]],
    [Q2[1], P[1]],
    [Q2[2], P[2]],
    color=edge_col,
    linewidth=lw_arrow,
        zorder=101
)
ax.plot([0, 0], [0, 0], [-blade_radius*0.5, blade_radius*0.8],
         color='k', linestyle='dashed', zorder=102)


# ------------------------------------------------------------
# AXIS LIMITS
# ------------------------------------------------------------

ax.set_xlim(-4.4, 5.7)
ax.set_ylim(-4.2, 4.4)
ax.set_zlim(-1.8, 5.2)

# Make the 3-D projection fill the axes
ax.set_box_aspect((
    5.7 - (-4.4),
    4.4 - (-4.2),
    5.2 - (-1.8)
))



# ------------------------------------------------------------
# EXPORT
# ------------------------------------------------------------

fig.savefig(
    out_pdf,
    format="pdf",
    dpi=dpi,
    facecolor="white",
    bbox_inches=None
)

plt.show()