"""
combine results from thickness and loading noise scattered on the beam
compare to direct beam noise evaluation
suggest model for thickness noise as PIN -> sources/sinks :)
the last point seems significant at 2-3 x BPF:
-- at 1xBPF direct noise dominant?
-- at 2-3xBPF thickness noise/ scattering of thickness noise dominant?
-- at 4+xBPF scattering of loading noise dominant
"""

# open G and gradG on surface. Compute resulting green terms, compute loading and thickness noise (direct + scattered)
# open exp data, evaluate all at exp observer positions
# compare spectra at points, directivities, ...
# profit?

# from scattering_vs_PIN import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from Constants.data_assim import getGojonData, getHarmonicsFromData
from PotentialInteraction.PIN import PotentialInteraction
from Constants.helpers import read_force_file, plot_3D_directivity, plot_3D_phase_directivity, plot_beam_azimuth, plot_rotation_arrow
MODE = 'half'
FILE = 'DIR_COMPONENTS'
Ntheta = 18
Nphi = 36


# BEGINNING OF HEADER
# vary configuration
from SourceMode.Configurations_NACA0012 import m_surface


# from SourceMode.Configurations_NACA0012 import D20L20W20_D180 as sourceArray # pick configuration
# SUFFIX = '_D20L20W20_D180'

# from SourceMode.Configurations_NACA0012 import D20L20W00_D180 as sourceArray # pick configuration
# SUFFIX = '_D180_MR'
# shape = 'D'

# from SourceMode.Configurations_NACA0012 import D20L20W00_D360 as sourceArray # pick configuration
# SUFFIX = '_D360_HR'

from SourceMode.Configurations_NACA0012 import PARROT_D20L20W00_D180 as sourceArray # pick configuration
SUFFIX = 'PARROT_D20L20_D180'
shape = 'PARROT'

sourceArray.numerics['CompactnessCorrection'] = True
# sourceArray.numerics['CompactnessCorrection'] = False


NDIPOLES = sourceArray.Ndipoles
ms = np.array([2])

r_inner, Fz, Fphi  = read_force_file('./Data/Zamponi2026/FS_ISAE_2_8000.txt') # reuse the radial stations from data

if shape == "PARROT":
    rt, t =  np.loadtxt('./Data/Parrot2024/thrust_Npm.csv', skiprows=1, delimiter=',').T # radius/r1, thrust in Npm
    rq, q =  np.loadtxt('./Data/Parrot2024/torque_Nmpm.csv', skiprows=1, delimiter=',').T # radius/r1, torque in Nmpm

    r_inner = sourceArray.seg_radius
    r1 = sourceArray.r1
    Fz = np.interp(r_inner/r1, rt, t) # same radial array
    Q = np.interp(r_inner/r1, rq, q) 
    Fphi = Q / r_inner

    TTARGET = 2.15 / sourceArray.B # Newtons
    QTARGET = 25 / 1000 / sourceArray.B # Newton-radian-meters
    Fz *= TTARGET / np.trapezoid(Fz, r_inner)  # rescale to target
    Fphi *= QTARGET / np.trapezoid(Fphi * r_inner, r_inner) # rescale to target

BLH, _, _, _ = sourceArray.getLoading(Fz, Fphi, steady_only=False) # compute loading on the fly, return PIN for reuse
PIN = sourceArray.PIN
D_bras = sourceArray.green.radius * 2
g = -1 * sourceArray.green.origin[2]
B = sourceArray.B
c = sourceArray.chord
Omega = sourceArray.Omega

if shape == 'PARROT':
    Omega *= -1
c0 = sourceArray.SoS
han = sourceArray.getHanson()
# END OF HEADER

datadir = './Experimental/dataverse_files'
# casefile = f'ISAE_2_D{int(1000*D_bras)}_L{int(1000*g)}'

data, BPF, freq, x_cart_data, theta_data, phi_data, theta_exp, phi_exp, casefile = getGojonData(datadir, D_bras, g, shape=shape, B=sourceArray.B, RPM=int(Omega * 60/2/np.pi))

data_modal, ms_data = getHarmonicsFromData(data, freq.T, BPF)
data = data_modal[np.where(ms[0] == ms_data)]

peq_data = np.sqrt(np.array(data) / 2) # "equivalent" mode amplitude from the data

theta_m_data, phi_m_data = np.meshgrid(theta_data, phi_data, indexing='ij')
theta_arr_data = theta_m_data.ravel()
phi_arr_data = phi_m_data.ravel()          

                            

# angular coordinates
theta = np.linspace(0.0, np.pi, Ntheta, endpoint=True)
phi   = np.linspace(0.0, 2.0 * np.pi, Nphi, endpoint=True)

# 2D mesh
theta_m, phi_m = np.meshgrid(theta, phi, indexing='ij')
# shapes: (Ntheta, Nphi)

# flatten
R = np.max(np.linalg.norm(x_cart_data, axis=0))
R_arr     = np.full(theta_m.size, R)
theta_arr = theta_m.ravel()
phi_arr   = phi_m.ravel()

X = R_arr * np.sin(theta_arr) * np.cos(phi_arr)
Y = R_arr * np.sin(theta_arr) * np.sin(phi_arr)
Z = R_arr * np.cos(theta_arr)

x_cart = np.array([X, Y, Z])



##### -------------------------------- SCATTERED LOADING NOISE ------------------------------------------
#### save gradients in the far-field (run once per observer and m)
# for index, sm in enumerate(sourceArray.children):

#     gradG_surface = np.load(f'./Data/current/NACA0012_rotor/gradG_surface_sm_{index}_{MODE}{SUFFIX}.npy') # shape (3, Nm, Nz, Ny)
#     print(f'pre-computing far-field gradients {index+1}')

#     gradG = sm.getScatteringGreenGradient(x_cart, m_surface * B * np.abs(Omega) / c0, gradG_surface) # shape (3, Nm, Nx, Ny)
#     np.save(f'./Data/current/NACA0012_rotor/gradG_sm_{index}_{MODE}_{FILE}{SUFFIX}.npy', gradG)


# extract and rearrange
gradG_arr = np.zeros((sourceArray.seg_radius.shape[0], 3, ms.shape[0], x_cart.shape[1], NDIPOLES), dtype=np.complex128)
ind_m = np.where(m_surface == ms[0])[0][0]
for index, sm in enumerate(sourceArray.children):
    gradG_arr[index] = np.load(f'./Data/current/NACA0012_rotor/gradG_sm_{index}_{MODE}_{FILE}{SUFFIX}.npy')[:, ind_m, :, :].reshape(3, ms.shape[0], x_cart.shape[1], NDIPOLES)




# steady/unsteady loading
BLH_S = np.zeros_like(BLH)
BLH_S[:, 0, :] = BLH[:, 0, :]
BLH_US = np.zeros_like(BLH)
BLH_US[:, 1:, :] = BLH[:, 1:, :]
# BLH_US[1, 1:, :] = BLH[1, 1:, :]

p_scattered_loading_S = sourceArray.getScatteredPressure(x_cart, ms, gradG=gradG_arr, BLH=np.transpose(BLH_S, axes=[2, 0, 1]))
p_direct_loading_S = sourceArray.getDirectPressure(x_cart, ms, BLH=np.transpose(BLH_S, axes=[2, 0, 1]))

p_scattered_loading_US = sourceArray.getScatteredPressure(x_cart, ms, gradG=gradG_arr, BLH=np.transpose(BLH_US, axes=[2, 0, 1]))
p_direct_loading_US = sourceArray.getDirectPressure(x_cart, ms, BLH=np.transpose(BLH_US, axes=[2, 0, 1]))

# p_scattered = np.load(f'./Data/current/NACA0012_rotor/p_scattered_{MODE}_m{int(m)}_{casename}{SUFFIX}.npy')
# p_direct_blade = np.load(f'./Data/current/NACA0012_rotor/p_direct_{MODE}_m{int(m)}_{casename}{SUFFIX}.npy')




#### -------------------------------- SCATTERED Thickness NOISE ------------------------------------------

## save gradients in the far-field (run once per observer and m)
# for index, sm in enumerate(sourceArray.children):
#     G_surface = np.load(f'./Data/current/NACA0012_rotor/G_surface_sm_{index}_{MODE}{SUFFIX}.npy') # shape (Nm, Nz, Ny)
#     print(f'pre-computing far-field G {index+1}')

#     G = sm.getScatteringGreen(x_cart, m_surface * B * np.abs(Omega) / c0, G_surface) # shape (Nm, Nx, Ny)
#     np.save(f'./Data/current/NACA0012_rotor/G_sm_{index}_{MODE}_{FILE}{SUFFIX}.npy', G)

Nr = sourceArray.seg_radius.shape[0]

# G_arr = np.zeros((Nr, m_surface.shape[0], x_cart.shape[1], NDIPOLES), dtype=np.complex128)

G_arr = np.zeros((Nr, ms.shape[0], x_cart.shape[1], NDIPOLES), dtype=np.complex128) # pick only the ones we neeed
ind_m = np.where(m_surface == ms[0])[0][0]
for index, sm in enumerate(sourceArray.children):
    G_arr[index] = np.load(f'./Data/current/NACA0012_rotor/G_sm_{index}_{MODE}_{FILE}{SUFFIX}.npy')[ind_m, :, :] # extract only the m we need for plotting!


p_scattered_thickness = sourceArray.getThicknessPressureScattered(x_cart, ms, G=G_arr)
p_direct_thickness = sourceArray.getThicknessPressureDirect(x_cart, ms)


# scattering total
p_total_scattering = p_direct_thickness + p_direct_loading_S + p_scattered_loading_S + p_direct_loading_US + p_scattered_thickness + p_scattered_loading_US 


# shift to relative phase w.r.t. mic one at phi=0
ind_theta_ref = 0
ind_phi_ref = np.where(phi==0)[0][0]
ind_combined = ind_theta_ref * phi.shape[0] + ind_phi_ref

phase_ref = np.angle(p_total_scattering[ind_combined, :])


p_direct_thickness *= np.exp(-1j * phase_ref)
p_direct_loading_S  *= np.exp(-1j * phase_ref)
p_scattered_loading_S *= np.exp(-1j * phase_ref)
p_direct_loading_US  *= np.exp(-1j * phase_ref)
p_scattered_loading_US *= np.exp(-1j * phase_ref)
p_scattered_thickness *= np.exp(-1j * phase_ref)
p_total_scattering *= np.exp(-1j * phase_ref)

# p_direct_thickness *= np.exp(-1j * np.angle(p_direct_thickness[ind_combined, :]))
# p_direct_loading  *= np.exp(-1j * np.angle(p_direct_loading[ind_combined, :]))
# p_scattered_loading *= np.exp(-1j * np.angle(p_scattered_loading[ind_combined, :]))
# p_scattered_thickness *= np.exp(-1j * np.angle(p_scattered_thickness[ind_combined, :]))
# p_total_scattering *= np.exp(-1j * np.angle(p_total_scattering[ind_combined, :]))






# + p_beam_total 
# + p_beam_loading + p_beam_thickness

# 3D SPL diagram
fig = plt.figure(figsize=(7, 7))
ax1 = fig.add_subplot(421, projection="3d")
ax2 = fig.add_subplot(422, projection="3d")
ax3 = fig.add_subplot(423, projection="3d")
ax4 = fig.add_subplot(424, projection="3d")
ax5 = fig.add_subplot(425, projection="3d")
ax6 = fig.add_subplot(426, projection="3d")
ax7 = fig.add_subplot(427, projection="3d")
ax8 = fig.add_subplot(428, projection="3d")



VMIN, VMAX = 10, 65
# fig, ax1 = plot_3D_directivity(
#     p_direct_thickness[:, 0], theta_m, phi_m, title='Direct Thickness Noise', fig=fig, ax=ax1, valmin=VMIN, valmax=VMAX,
# )
# fig, ax2 = plot_3D_directivity(
#     p_scattered_thickness[:, 0], theta_m, phi_m, title='Scattered Thickness Noise', fig=fig, ax=ax2, valmin=VMIN, valmax=VMAX,
# )
# fig, ax3 = plot_3D_directivity(
#     p_direct_loading_S[:, 0],theta_m, phi_m, title='Direct Steady Loading Noise', fig=fig, ax=ax3, valmin=VMIN, valmax=VMAX,
# )
# fig, ax4 = plot_3D_directivity(
#     p_scattered_loading_S[:, 0], theta_m, phi_m, title='Scattered Steady Loading Noise', fig=fig, ax=ax4, valmin=VMIN, valmax=VMAX,
# )
# fig, ax5 = plot_3D_directivity(
#     p_direct_loading_US[:, 0],theta_m, phi_m, title='Direct Unsteady Loading Noise', fig=fig, ax=ax5, valmin=VMIN, valmax=VMAX,
# )
# fig, ax6 = plot_3D_directivity(
#     p_scattered_loading_US[:, 0], theta_m, phi_m, title='Scattered Unsteady Loading Noise', fig=fig, ax=ax6, valmin=VMIN, valmax=VMAX,
# )
# fig, ax7 = plot_3D_directivity(
#     p_total_scattering[:, 0], theta_m, phi_m, title='Model Total', fig=fig, ax=ax7, valmin=VMIN, valmax=VMAX,
# )
# fig, ax8 = plot_3D_directivity(
#     peq_data[0, :, :], np.deg2rad(theta_m_data), np.deg2rad(phi_m_data), title='Experiment', fig=fig, ax=ax8, valmin=VMIN, valmax=VMAX,
# )
# fig.suptitle(f"Directivities of $\hat{{p}}_{{{ms[0] * B:.0f}}}$")

fig, ax1, _ = plot_3D_directivity(
    p_direct_thickness[:, 0], theta_m, phi_m, title=fr'$\langle G_0 Q\rangle_\Omega$', fig=fig, ax=ax1, valmin=VMIN, valmax=VMAX,
plot_cbar=False,
)
fig, ax2, _  = plot_3D_directivity(
    p_scattered_thickness[:, 0], theta_m, phi_m, title=fr'$\langle G_s Q\rangle_\Omega$', fig=fig, ax=ax2, valmin=VMIN, valmax=VMAX,
plot_cbar=False,
)
fig, ax3, _  = plot_3D_directivity(
    p_direct_loading_S[:, 0],theta_m, phi_m, title=fr'$-\langle \boldsymbol{{\nabla}} G_0 \circ \boldsymbol{{F}}_0\rangle_\Omega$', fig=fig, ax=ax3, valmin=VMIN, valmax=VMAX,
plot_cbar=False,
)
fig, ax4, _  = plot_3D_directivity(
    p_scattered_loading_S[:, 0], theta_m, phi_m, title=fr'$-\langle \boldsymbol{{\nabla}} G_s \circ \boldsymbol{{F}}_0\rangle_\Omega$', fig=fig, ax=ax4, valmin=VMIN, valmax=VMAX,
plot_cbar=False,
)
fig, ax5, _  = plot_3D_directivity(
    p_direct_loading_US[:, 0],theta_m, phi_m, title=fr'$-\sum_{{k>0}}\langle \boldsymbol{{\nabla}} G_0 \circ \boldsymbol{{F}}_k\rangle_\Omega$', fig=fig, ax=ax5, valmin=VMIN, valmax=VMAX,
plot_cbar=False,
)
fig, ax6, _  = plot_3D_directivity(
    p_scattered_loading_US[:, 0], theta_m, phi_m, title=fr'$-\sum_{{k>0}}\langle \boldsymbol{{\nabla}} G_s \circ \boldsymbol{{F}}_k\rangle_\Omega$', fig=fig, ax=ax6, valmin=VMIN, valmax=VMAX,
plot_cbar=False,
)
fig, ax7, _  = plot_3D_directivity(
    p_total_scattering[:, 0], theta_m, phi_m, title='Model Total', fig=fig, ax=ax7, valmin=VMIN, valmax=VMAX,
plot_cbar=False,
)
fig, ax8, mappable  = plot_3D_directivity(
    peq_data[0, :, :], np.deg2rad(theta_m_data), np.deg2rad(phi_m_data), title='Experiment', fig=fig, ax=ax8, valmin=VMIN, valmax=VMAX,
plot_cbar=False,
)

cbar = fig.colorbar(mappable, ax=[ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8], shrink=0.7, pad=0.1)
cbar.set_label("Directivity [dB]")

R0 = 1.1
R1 = R0 * 1.1
for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
    plot_beam_azimuth(R0, fig, ax)
    plot_rotation_arrow(R1, PHI_EXTENT=[20, 90], fig=fig, ax=ax)
    ax.set_xlim(-R0, R0)
    ax.set_ylim(-R0, R0)
    ax.set_zlim(-R0, R0)



# 3D phase diagram
fig = plt.figure(figsize=(7, 7))
ax1 = fig.add_subplot(421, projection="3d")
ax2 = fig.add_subplot(422, projection="3d")
ax3 = fig.add_subplot(423, projection="3d")
ax4 = fig.add_subplot(424, projection="3d")
ax5 = fig.add_subplot(425, projection="3d")
ax6 = fig.add_subplot(426, projection="3d")
ax7 = fig.add_subplot(427, projection="3d")
ax8 = fig.add_subplot(428, projection="3d")



VMIN, VMAX = 10, 65
fig, ax1, _  = plot_3D_phase_directivity(
    p_direct_thickness[:, 0], theta_m, phi_m, title=fr'$\langle G_0 Q\rangle_\Omega$', fig=fig, ax=ax1, valmin=VMIN, valmax=VMAX,
plot_cbar=False,
)
fig, ax2, _  = plot_3D_phase_directivity(
    p_scattered_thickness[:, 0], theta_m, phi_m, title=fr'$\langle G_s Q\rangle_\Omega$', fig=fig, ax=ax2, valmin=VMIN, valmax=VMAX,
plot_cbar=False,
)
fig, ax3, _  = plot_3D_phase_directivity(
    p_direct_loading_S[:, 0],theta_m, phi_m, title=fr'$-\langle \boldsymbol{{\nabla}} G_0 \circ \boldsymbol{{F}}_0\rangle_\Omega$', fig=fig, ax=ax3, valmin=VMIN, valmax=VMAX,
plot_cbar=False,
)
fig, ax4, _  = plot_3D_phase_directivity(
    p_scattered_loading_S[:, 0], theta_m, phi_m, title=fr'$-\langle \boldsymbol{{\nabla}} G_s \circ \boldsymbol{{F}}_0\rangle_\Omega$', fig=fig, ax=ax4, valmin=VMIN, valmax=VMAX,
plot_cbar=False,
)
fig, ax5, _  = plot_3D_phase_directivity(
    p_direct_loading_US[:, 0],theta_m, phi_m, title=fr'$-\sum_{{k>0}}\langle \boldsymbol{{\nabla}} G_0 \circ \boldsymbol{{F}}_k\rangle_\Omega$', fig=fig, ax=ax5, valmin=VMIN, valmax=VMAX,
plot_cbar=False,
)
fig, ax6, _  = plot_3D_phase_directivity(
    p_scattered_loading_US[:, 0], theta_m, phi_m, title=fr'$-\sum_{{k>0}}\langle \boldsymbol{{\nabla}} G_s \circ \boldsymbol{{F}}_k\rangle_\Omega$', fig=fig, ax=ax6, valmin=VMIN, valmax=VMAX,
plot_cbar=False,
)
fig, ax7, _  = plot_3D_phase_directivity(
    p_total_scattering[:, 0], theta_m, phi_m, title='Model Total', fig=fig, ax=ax7, valmin=VMIN, valmax=VMAX,
plot_cbar=False,
)
fig, ax8, mappable  = plot_3D_phase_directivity(
    peq_data[0, :, :], np.deg2rad(theta_m_data), np.deg2rad(phi_m_data), title='Experiment', fig=fig, ax=ax8, valmin=VMIN, valmax=VMAX,
plot_cbar=False,
)
# fig.suptitle(rf"Directivities of $\hat{{p}}_{{{ms[0] * B:.0f}}}$")
cbar = fig.colorbar(mappable, ax=[ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8], shrink=0.7, pad=0.1)
cbar.set_label("Phase [rad]")
for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
    plot_beam_azimuth(R0, fig, ax)
    plot_rotation_arrow(R1, PHI_EXTENT=[20, 90], fig=fig, ax=ax)
    ax.set_xlim(-R0, R0)
    ax.set_ylim(-R0, R0)
    ax.set_zlim(-R0, R0)

plt.show()
