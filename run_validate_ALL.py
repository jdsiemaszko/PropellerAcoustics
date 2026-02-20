from Hanson.far_field import HansonModel
from Hanson.near_field import NearFieldHansonModel
from SourceMode.SourceMode import SourceModeArray
from TailoredGreen.CylinderGreen import CylinderGreen
from TailoredGreen.TailoredGreen import TailoredGreen
from Constants.helpers import p_to_SPL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py


# _________ INPUTS __________

# arbitrary 
ms = np.array([1, 5, 10]) # modes to plot

# ASSUMPTIONS
NBLADES = 2
TWIST = 10 # deg
CHORD = 0.025 #m
NBEAMS = 1 # does nothing
RHO = 1.2#kgm^-3
SOS = 340 # ms^-1

# VELLA ET AL DATA
DATAPATH = './Validation/harmonics_ISAE2_D20_L20-1.h5'

datafile = h5py.File(DATAPATH, 'r')

R = np.array(datafile['R'][0]) # 1.62
SPL = np.array(datafile['SPL']) #
phi = np.array(datafile['phi'][0]) # pi/2
theta = np.array(datafile['theta'][:, 0]) # 0 -> pi
r = np.array(datafile['r'][:, 0]) # discretization in the radial dir
ddr = r[5] - r[4]
r_bounds = np.concatenate(([r[0]-ddr], r))
dr = np.diff(r_bounds)
BPF = np.array(datafile['BPF'][:, 0]) # in Hz?

BOmega = BPF[0] #HZ !!!!

OMEGA = BOmega /NBLADES * 2 * np.pi

Fk_phi_i = np.array(datafile["Fk_phi_imag"])
Fk_phi_r = np.array(datafile["Fk_phi_real"])

Fk_z_i = np.array(datafile["Fk_z_imag"])
Fk_z_r = np.array(datafile["Fk_z_real"])

Nr, Nk = Fk_phi_i.shape # (20, 40)

Fk_z = Fk_z_r + 1j * Fk_z_i
Fk_phi = Fk_phi_r + 1j * Fk_phi_i
Fk_r = np.zeros_like(Fk_z, dtype=np.complex128)

loading = np.stack(
    (Fk_r.T, Fk_z.T, Fk_phi.T)
) # (3, Nk, Nr)
loading = np.concatenate((np.zeros((3, 1, Nr)), loading), axis=1)
loading_per_unit_span = loading/ dr[None, None, :] # force PER UNIT LENGTH
loading_per_unit_span_magnitude = loading_per_unit_span[1, :, :] / np.cos(np.deg2rad(TWIST)) # convert to blade-aligned loading per unit span

p_i = np.array(datafile['p_imag'])
p_r = np.array(datafile['p_real'])

p_data = p_r + 1j * p_i

shift = np.pi

x_cartesian = R * np.array([
    np.sin(theta) * np.cos(phi+shift),
    np.sin(theta) * np.sin(phi+shift),
    np.cos(theta),
])

m = np.round(BPF / BOmega)


# _________ NUMERICAL INPUTS __________
NSEG = len(r)
KMAX = Nk
NTHETA = 18*2 # discretization in the polar angle
NPHI = 36*2 # discretization in the azimuth
NDIPOLES = 256 # discterization of each source mode
# DATA ASSIMILATION (VELLA ET AL. 2026)

# _________ SETUP ____________
twist_array = np.deg2rad(TWIST) * np.ones(NSEG+1)
chord_array = CHORD * np.ones(NSEG+1)
radius_array = r_bounds

# GREEN'S FUNCTION MODULE
gf = TailoredGreen(dim=3) # free-field version!

# HANSON MODULE
axis_prop = np.array([0.0, 0.0, 1.0]) # z-direction propeller...
origin_prop = np.array([0.0, 0.0, 0.0]) # ... at z=0
HANSON_VELLA = HansonModel(twist_rad = twist_array, chord_m = chord_array,
                    loadings_Npm = loading_per_unit_span_magnitude,
                    axis=axis_prop, origin=origin_prop,
                    radius_m=radius_array, B=NBLADES, nb=NBEAMS,
                     Omega_rads=OMEGA, rho_kgm3=RHO, c_mps=SOS)

HANSON_NEARFIELD = NearFieldHansonModel(twist_rad = twist_array, chord_m = chord_array,
                    loadings_Npm = loading_per_unit_span_magnitude,
                    axis=axis_prop, origin=origin_prop,
                                radius_m=radius_array, B=NBLADES, nb=NBEAMS,
                            Omega_rads=OMEGA, rho_kgm3=RHO, c_mps=SOS)

# SOURCE MODE MODULE
axis_prop = np.array([0.0, 0.0, 1.0]) # z-direction propeller...
origin_prop = np.array([0.0, 0.0, 0.0]) # ... at z=0

sourceArray = SourceModeArray(BLH=loading_per_unit_span_magnitude, # loading per unit span, magnitude only
                        B = NBLADES,
                        Omega=OMEGA, gamma = twist_array,
                        axis=axis_prop, origin=origin_prop,
                        radius=radius_array,
                        green = gf,
                        numerics={'Ndipoles' : NDIPOLES},
                        c = SOS
                        )

# _____________ PLOTTING & RESULTS _______________

# 1) plot geometry

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")
sourceArray.plotSelf(fig, ax)
# ax.set_box_aspect([1, 1, 1])
ax.set_aspect('equal')
ax.set_axis_off()
plt.show()
plt.close()

#2) plot directivity
# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(111, projection="3d")
# sourceArray.plotFarFieldPressure(m=np.array([M]), Nphi=NPHI, Ntheta=NTHETA, R=ROBS,
#                                   valmax=65, valmin=10,
#                                   fig=fig, ax=ax
#                                   )
# plt.show()
# plt.close(fig)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")
HANSON_VELLA.plotDirectivity(fig, ax, m=5, R=R,
                        valmax=65, valmin=10,
                        Nphi=NPHI, Ntheta=NTHETA,
                        )
plt.show()
plt.close(fig)

p_hanson, _ = HANSON_VELLA.getPressureRotor(x_cartesian, m=ms)
p_nf, _ = HANSON_NEARFIELD.getPressureRotor(x_cartesian, m=ms)
p_sourceMode = sourceArray.getPressure(x_cartesian, m=ms)

fig, ax = plt.subplots(figsize=(4, 3))
for ind, (color, mode) in enumerate(zip(['r', 'b', 'g'], ms)):

    index_data = np.where(m == mode)[0][0]
    ax.plot(np.rad2deg(theta), p_to_SPL(p_hanson)[:, ind] , color=color, marker='x', label=f'm={mode}')
    ax.plot(np.rad2deg(theta), p_to_SPL(p_nf)[:, ind] , color=color, marker='s', linestyle='dotted')
    ax.plot(np.rad2deg(theta), p_to_SPL(p_sourceMode)[:, ind], color=color, marker='^', linestyle='dashed')
    # ax.plot(np.rad2deg(theta), p_to_SPL(p_data)[:, index], color=color, marker='o', linestyle='dashdot')
    ax.plot(np.rad2deg(theta), SPL[:, index_data], color=color, marker='o', linestyle='dashdot')  # should be the same


ax.legend()
ax.set_xlabel('Polar angle [deg]')
ax.set_ylabel('Modal SPL [dB]')
ax.grid()
plt.tight_layout()
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(4, 3))
# plot_directivity(fig, ax, x, p_to_SPL(p_model)[:, m_to_plot-1])
# plot_directivity(fig, ax2, x, p_to_SPL(p)[:, m_to_plot-1])

for index, (color, mode) in enumerate(zip(['r', 'b', 'g'], ms)):
    index_data = np.where(m == mode)[0][0]

    ax.plot(np.rad2deg(theta), np.rad2deg(np.angle(p_hanson))[:, index], color=color, label=f'm={mode}', marker='x')
    ax.plot(np.rad2deg(theta), np.rad2deg(np.angle(p_nf))[:, index], color=color, marker='s', linestyle='dotted')
    ax.plot(np.rad2deg(theta), np.rad2deg(np.angle(p_sourceMode))[:, index], color=color, marker='^', linestyle='dashed')
    ax.plot(np.rad2deg(theta), np.rad2deg(np.angle(-np.conjugate(p_data)))[:, index_data], color=color, marker='o', linestyle='dashdot')


ax.legend()
ax.set_xlabel('Polar angle [deg]')
ax.set_ylabel('Phase [deg]')
ax.grid()
plt.tight_layout()
plt.show()
plt.close()


