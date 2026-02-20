from PotentialInteraction.main import HansonModel
from PotentialInteraction.near_field import NearFieldHansonModel
from SourceMode.SourceMode import SourceModeArray
from TailoredGreen.CylinderGreen import CylinderGreen
from TailoredGreen.TailoredGreen import TailoredGreen
from Constants.const import p_to_SPL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# _________ INPUTS __________
SOS = 1.0 # m/s
RHO = 1.0 # kgm^-3
CHORD = 1.0 # m
TWIST = 45 # deg
NBLADES = 1
NBEAMS = 1 # radial - 1, diameterical - 2
R0_PROP = 0.5 # m
R1_PROP = 1.5 # m
OMEGA = 1.0 # rad/s 
K_BPF = OMEGA * NBLADES / SOS # 1
ROBS = 100.0 / K_BPF # far field!

K = 1 # K'th forcing harmonic
M = 2 # m'th pressure mode to resolve
FK_ABS = 1.0 # force magnitude in N/m!

# _________ NUMERICAL INPUTS __________
NSEG = 1 # number of radial prop segments
KMAX = 128 # max loading harmonic number
NTHETA = 18*2 # discretization in the polar angle
NPHI = 36*2 # discretization in the azimuth
MMAX = 32 # maximum resolved mode number in cylinder green's function
NQ_PROP = 64 # discretization in the z wavenumber in cylinder green's function
EPS_RADIUS = 1e-3 # cutoff for cylinder distance. points are ignored if distance to cylinder < a * (1 + EPS_RADIUS)
NDIPOLES = 64 # discterization of each source mode

# _________ SETUP ____________

twist_array = np.deg2rad(TWIST) * np.ones(NSEG+1)
chord_array = CHORD * np.ones(NSEG+1)
radius_array = np.linspace(R0_PROP, R1_PROP, NSEG+1, endpoint=True)
loadings = np.zeros((KMAX+1, 1))
loadings[K:K+4, :] = FK_ABS
loadings_3D = np.zeros((3, KMAX+1, 1))
loadings_3D[1, :, :] = loadings[:, :] * np.cos(np.deg2rad(TWIST))
loadings_3D[2, :, :] = loadings[:, :] * np.sin(np.deg2rad(TWIST))


# GREEN'S FUNCTION MODULE
gf = TailoredGreen(dim=3) # free-field version!
# gf.plotFarFieldGradient(M * OMEGA * NBLADES / SOS, y = np.array([0.0, 0.0, 0.0]).reshape(3, 1), R=ROBS)
# gf.plotFarFieldGradient(M * OMEGA * NBLADES / SOS, y = np.array([1.0, 0.0, 0.0]).reshape(3, 1), R=ROBS)
# gf.plotFarFieldGradient(M * OMEGA * NBLADES / SOS, y = np.array([0.0, 1.0, 0.0]).reshape(3, 1), R=ROBS)
# gf.plotFarFieldGradient(M * OMEGA * NBLADES / SOS, y = np.array([0.0, 0.0, 1.0]).reshape(3, 1), R=ROBS)

# gf.plotDirectivity(M * OMEGA * NBLADES / SOS, y = np.array([0.0, 0.0, 0.0]).reshape(3, 1), R=ROBS)
# gf.plotDirectivity(M * OMEGA * NBLADES / SOS, y = np.array([1.0, 0.0, 0.0]).reshape(3, 1), R=ROBS)
# gf.plotDirectivity(M * OMEGA * NBLADES / SOS, y = np.array([0.0, 1.0, 0.0]).reshape(3, 1), R=ROBS)
# gf.plotDirectivity(M * OMEGA * NBLADES / SOS, y = np.array([0.0, 0.0, 1.0]).reshape(3, 1), R=ROBS)


# HANSON MODULE
HANSON_VELLA = HansonModel(twist_rad = twist_array, chord_m = chord_array,
                    radius_m=radius_array, B=NBLADES, nb=NBEAMS,
                    Dcylinder_m=0.0, Lcylinder_m=0.0, Omega_rads=OMEGA, rho_kgm3=RHO, c_ms=SOS, kmax=KMAX)

HANSON_NEARFIELD = NearFieldHansonModel(twist_rad = twist_array, chord_m = chord_array,
                                radius_m=radius_array, B=NBLADES, nb=NBEAMS,
                            Dcylinder_m=0.0, Lcylinder_m=0.0, Omega_rads=OMEGA, rho_kgm3=RHO, c_ms=SOS, kmax=KMAX)

# SOURCE MODE MODULE
axis_prop = np.array([0.0, 0.0, 1.0]) # z-direction propeller...
origin_prop = np.array([0.0, 0.0, 0.0]) # ... at z=0

sourceArray = SourceModeArray(BLH=loadings, # loading per unit span
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

##### 2) plot directivity
# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(111, projection="3d")
# sourceArray.plotFarFieldPressure(m=np.array([M]), Nphi=NPHI, Ntheta=NTHETA, R=ROBS,
#                                   valmax=65, valmin=10,
#                                   fig=fig, ax=ax
#                                   )
# plt.show()
# plt.close(fig)

# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(111, projection="3d")
# HANSON_VELLA.plotDirectivity(fig, ax, m=M, R=ROBS,
#                         valmax=65, valmin=10,
#                         Nphi=NPHI, Ntheta=NTHETA,
#                         # mode='beam',
#                         #   mode='total',
#                         mode='blade',
#                         loadings=loadings_3D
#                         )
# plt.show()
# plt.close(fig)

phi = np.pi/2
NLINE = NTHETA* 2
theta = np.linspace(0, np.pi, NLINE)
x_cartesian = ROBS * np.array([
    np.sin(theta) * np.cos(phi),
    np.sin(theta) * np.sin(phi),
    np.cos(theta),
])

x_polar = np.array([np.ones(NLINE) * ROBS, theta, np.ones(NLINE) * phi])

# ms = np.array([1,5,10])
ms = np.array([1,2,5])

p_hanson, _ = HANSON_VELLA.getHansonPressure(x_polar, m=ms, loading=loadings_3D,  B=NBLADES, Omega=OMEGA, nb=NBEAMS, multiplier=NBLADES)
p_nf, _ = HANSON_NEARFIELD.getHansonPressure(x_polar, m=ms, loading=loadings_3D,  B=NBLADES, Omega=OMEGA, nb=NBEAMS, multiplier=NBLADES)
p_sourceMode = sourceArray.getPressure(x_cartesian, m=ms)

fig, ax = plt.subplots(figsize=(4, 3))
for index, (color, mode) in enumerate(zip(['r', 'b', 'g'], ms)):
    ax.plot(np.rad2deg(theta), p_to_SPL(p_hanson)[:, index] , color=color, marker='x', label=f'm={mode}')
    ax.plot(np.rad2deg(theta), p_to_SPL(p_nf)[:, index] , color=color, marker='s', linestyle='dotted')
    ax.plot(np.rad2deg(theta), p_to_SPL(p_sourceMode)[:, index], color=color, marker='^', linestyle='dashed')

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
    ax.plot(np.rad2deg(theta), np.rad2deg(np.angle(p_hanson))[:, index], color=color, label=f'm={mode}', marker='x')
    ax.plot(np.rad2deg(theta), np.rad2deg(np.angle(p_nf))[:, index], color=color, marker='s', linestyle='dotted')
    ax.plot(np.rad2deg(theta), np.rad2deg(np.angle(p_sourceMode))[:, index], color=color, marker='^', linestyle='dashed')

ax.legend()
ax.set_xlabel('Polar angle [deg]')
ax.set_ylabel('Phase [deg]')
ax.grid()
plt.tight_layout()
plt.show()
plt.close()


