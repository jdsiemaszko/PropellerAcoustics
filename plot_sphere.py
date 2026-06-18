import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Constants.helpers import read_force_file, plot_3D_directivity, plot_3D_phase_directivity, plot_beam_azimuth, plot_rotation_arrow

# -----------------------------
# Inputs: arrays of angles
# theta = polar angle (0..pi)
# phi   = azimuth angle (0..2pi)
# -----------------------------
theta = np.linspace(3 / 18 * np.pi, np.pi - 3 / 18 * np.pi, 10)
phi   = np.linspace(np.pi/2, np.pi/2, 10)

# Ensure same length (example curve pairing)
N = min(len(theta), len(phi))
theta = theta[:N]
phi = phi[:N]

# -----------------------------
# Convert spherical -> Cartesian (unit sphere)
# -----------------------------
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# -----------------------------
# Plot
# -----------------------------
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")

ax.view_init(25, 45)


# Unit sphere wireframe (optional but helpful)
u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 50)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))
ax.plot_wireframe(xs, ys, zs, color='lightgray', alpha=0.3)

# Curve on sphere
# ax.plot(x, y, z, 'b-', linewidth=2, label="curve")

# Points on curve
ax.scatter(x, y, z, color='k', marker='o', s=40)
R0 = 1.2
R1 = R0 * 1.2
# -----------------------------
# Direction arrows along curve
# -----------------------------
# dx = np.gradient(x[:-1])
# dy = np.gradient(y[:-1])
# dz = np.gradient(z[:-1])
dx = x[1:] - x[:-1]
dy = y[1:] - y[:-1]
dz = z[1:] - z[:-1]

ax.quiver(
    x[:-1], y[:-1], z[:-1],
    dx, dy, dz,
    # length=0.15,
    normalize=False,
    color='r'
)
plot_beam_azimuth(R1, fig, ax)
plot_rotation_arrow(R1, PHI_EXTENT=[20, 90], fig=fig, ax=ax)
# -----------------------------
# Formatting
# -----------------------------
# ax.set_title("Curve on Unit Sphere with Direction Arrows")
ax.set_box_aspect([1, 1, 1])
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_axis_off()

plt.show()