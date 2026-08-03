from TailoredGreen.CylinderGreen import CylinderGreen
from TailoredGreen.TailoredGreen import TailoredGreen
from SourceMode.SourceMode import SourceMode, SourceModeArray
from Constants.const import p_to_SPL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py

# ----------------- READ DATA --------------------
DATAPATH = './Validation/harmonics_ISAE2_D20_L20-1.h5'

NSEG = 20

datafile = h5py.File(DATAPATH, 'r')

R = np.array(datafile['R'][0]) # 1.62
SPL = np.array(datafile['SPL']) # 1.62
phi = np.array(datafile['phi'][0]) # pi/2
theta = np.array(datafile['theta'][:, 0]) # 0 -> pi
r = np.array(datafile['r'][:, 0]) # discretization in the radial dir
ddr = r[5] - r[4]
r_bounds = np.concatenate(([r[0]-ddr], r)) + ddr/2
dr = np.diff(r_bounds)
BPF = np.array(datafile['BPF'][:, 0]) # in Hz?
BOmega = BPF[0] #HZ !!!!

B = 2 # ?

Omega = BOmega / B * 2 * np.pi

Fk_phi_i = np.array(datafile["Fk_phi_imag"])
Fk_phi_r = np.array(datafile["Fk_phi_real"])

Fk_z_i = np.array(datafile["Fk_z_imag"])
Fk_z_r = np.array(datafile["Fk_z_real"])

Nr, Nk = Fk_phi_i.shape # (20, 40)

Fk_z = Fk_z_r + 1j * Fk_z_i
Fk_phi = Fk_phi_r + 1j * Fk_phi_i
# Fk_phi = np.zeros_like(Fk_z, dtype=np.complex128)
Fk_r = np.zeros_like(Fk_z, dtype=np.complex128)

loading = np.stack(
    (Fk_r.T, Fk_z.T, Fk_phi.T) # note: Fkphi is opposite to the motion!
) # (3, Nk, Nr)
loading = np.concatenate((np.zeros((3, 1, Nr)), loading), axis=1) # (3, Nk, Nr)
loading_per_unit_span = loading/ dr[None, None, :] # force PER UNIT LENGTH

p_i = np.array(datafile['p_imag'])
p_r = np.array(datafile['p_real'])

p = p_r + 1j * p_i

shift = np.pi
x_polar = np.array([
    R * np.ones_like(theta),
    theta, 
    (phi+shift) * np.ones_like(theta)
])
x_cartesian = R * np.array([
    np.sin(theta) * np.cos(phi+shift),
    np.sin(theta) * np.sin(phi+shift),
    np.cos(theta),
])

m = np.round(BPF / BOmega)
MAXK = Nk
GAMMA = np.deg2rad(10)
loading_magnitude = loading[1, :, :] / np.cos(GAMMA) # still complex!
# loading_magnitude = np.conjugate(loading_magnitude)
# ----------------------- SETUP -----------------------

RAD_CYL = 0.1
SPACING = 0.11 * 2 # slightly more than one diameter away
axis_cyl = np.array([0.0, 0.0, 1.0])
origin_cyl = np.array([0.0, SPACING, 0.0])
cg = CylinderGreen(radius=RAD_CYL, axis=axis_cyl, origin=origin_cyl, dim=3, 
                        numerics={
                    'nmax': 32,
                    'Nq_prop': 64,
                    'eps_k' : 1e-6,
                    'eps_radius' : 1e-2
                })

sourceArray = SourceModeArray(BLH=loading_magnitude / dr[None, :], # loading per unit span
                         B = B,
                         Omega=Omega, gamma = np.ones(len(r_bounds)) * -GAMMA,
                         axis=np.array([0.0, 0.0, 1.]), origin=np.array([0.0, 0.0, 0.0]),
                           radial = np.array([1.0, 0.0, 0.0]), # why this?
                           radius=r_bounds,
                         green = cg,
                         numerics={'Ndipoles' : 32}
                         )

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")
# source.plotSelf(fig, ax)
sourceArray.plotSelf(fig, ax)
ax.scatter(x_cartesian[0], x_cartesian[1], x_cartesian[2], marker='x', color='k')
# ax.set_box_aspect([1, 1, 1])
ax.set_aspect('equal')
ax.set_axis_off()
plt.show()
plt.close()

sourceArray.plotFarFieldPressure(m=np.array([5]), Nphi=36, Ntheta=18, R=R,
                                 valmax=65, valmin=10)

p_model = sourceArray.getPressure(x_cartesian, m)

fig, ax = plt.subplots(figsize=(4, 3))
# plot_directivity(fig, ax, x, spl_from_p(p_model)[:, m_to_plot-1])
# plot_directivity(fig, ax2, x, spl_from_p(p)[:, m_to_plot-1])

for color, mode in zip(['r', 'b', 'g'], [1, 5, 10]):
    ax.plot(np.rad2deg(theta), p_to_SPL(p_model)[:, mode-1] , color=color, marker='s', linestyle='dashed', label=f'm={mode}')
    ax.plot(np.rad2deg(theta), p_to_SPL(p)[:, mode-1], color=color, marker='x', markersize=10)
    # ax.plot(np.rad2deg(theta), SPL[:, mode-1], color=color)

ax.legend()
ax.set_xlabel('Polar angle [deg]')
ax.set_ylabel('Modal SPL [dB]')
ax.grid()
plt.tight_layout()
plt.show()
plt.close()