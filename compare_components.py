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
from Constants.helpers import read_force_file, plot_3D_directivity, plot_3D_phase_directivity
MODE = 'half'
FILE = 'DIR_COMPONENTS'
Ntheta = 18
Nphi = 36


# BEGINNING OF HEADER
# vary configuration
from SourceMode.Configurations_NACA0012 import m_surface

# from SourceMode.Configurations_NACA0012 import D20L20W00_D360 as sourceArray # pick configuration
# SUFFIX = '_D360_HR'

# from SourceMode.Configurations_NACA0012 import D20L20W20_D180 as sourceArray # pick configuration
# SUFFIX = '_D20L20W20_D180'

from SourceMode.Configurations_NACA0012 import D20L20W00_D180 as sourceArray # pick configuration
SUFFIX = '_D180_MR'

sourceArray.numerics['CompactnessCorrection'] = True

NDIPOLES = sourceArray.Ndipoles
ms = np.array([3])

r_inner, Fz, Fphi  = read_force_file('./Data/Zamponi2026/FS_ISAE_2_8000.txt') # reuse the radial stations from data
BLH, _, _, _ = sourceArray.getLoading(Fz, Fphi, steady_only=False) # compute loading on the fly, return PIN for reuse
PIN = sourceArray.PIN
D_bras = sourceArray.green.radius * 2
g = -1 * sourceArray.green.origin[2]
B = sourceArray.B
c = sourceArray.chord[0]
Omega = sourceArray.Omega
c0 = sourceArray.SoS
han = sourceArray.getHanson()
# END OF HEADER

datadir = './Experimental/dataverse_files'
# casefile = f'ISAE_2_D{int(1000*D_bras)}_L{int(1000*g)}'

data, BPF, freq, x_cart_data, theta_data, phi_data, theta_exp, phi_exp, casefile = getGojonData(datadir, D_bras, g, shape='D', B=sourceArray.B, RPM=int(Omega * 60/2/np.pi))

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

#     gradG = sm.getScatteringGreenGradient(x_cart, m_surface * B * Omega / c0, gradG_surface) # shape (3, Nm, Nx, Ny)
#     np.save(f'./Data/current/NACA0012_rotor/gradG_sm_{index}_{MODE}_{FILE}{SUFFIX}.npy', gradG)


# extract and rearrange
gradG_arr = np.zeros((sourceArray.seg_radius.shape[0], 3, ms.shape[0], x_cart.shape[1], NDIPOLES), dtype=np.complex128)
ind_m = np.where(m_surface == ms[0])[0][0]
for index, sm in enumerate(sourceArray.children):
    gradG_arr[index] = np.load(f'./Data/current/NACA0012_rotor/gradG_sm_{index}_{MODE}_{FILE}{SUFFIX}.npy')[:, ind_m, :, :].reshape(3, ms.shape[0], x_cart.shape[1], NDIPOLES)


p_scattered_loading = sourceArray.getScatteredPressure(x_cart, ms, gradG=gradG_arr)
p_direct_loading = sourceArray.getDirectPressure(x_cart, ms)

# p_scattered = np.load(f'./Data/current/NACA0012_rotor/p_scattered_{MODE}_m{int(m)}_{casename}{SUFFIX}.npy')
# p_direct_blade = np.load(f'./Data/current/NACA0012_rotor/p_direct_{MODE}_m{int(m)}_{casename}{SUFFIX}.npy')




#### -------------------------------- SCATTERED Thickness NOISE ------------------------------------------

### save gradients in the far-field (run once per observer and m)
# for index, sm in enumerate(sourceArray.children):
#     G_surface = np.load(f'./Data/current/NACA0012_rotor/G_surface_sm_{index}_{MODE}{SUFFIX}.npy') # shape (Nm, Nz, Ny)
#     print(f'pre-computing far-field G {index+1}')

#     G = sm.getScatteringGreen(x_cart, m_surface * B * Omega / c0, G_surface) # shape (Nm, Nx, Ny)
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
p_total_scattering = p_direct_thickness + p_direct_loading + p_scattered_loading + p_scattered_thickness


# shift to relative phase w.r.t. mic one at phi=0
ind_theta_ref = 0
ind_phi_ref = np.where(phi==0)[0][0]
ind_combined = ind_theta_ref * phi.shape[0] + ind_phi_ref

phase_ref = np.angle(p_total_scattering[ind_combined, :])


p_direct_thickness *= np.exp(-1j * phase_ref)
p_direct_loading  *= np.exp(-1j * phase_ref)
p_scattered_loading *= np.exp(-1j * phase_ref)
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
ax1 = fig.add_subplot(321, projection="3d")
ax2 = fig.add_subplot(322, projection="3d")
ax3 = fig.add_subplot(323, projection="3d")
ax4 = fig.add_subplot(324, projection="3d")
ax5 = fig.add_subplot(325, projection="3d")
ax6 = fig.add_subplot(326, projection="3d")


VMIN, VMAX = 10, 65
fig, ax1 = plot_3D_directivity(
    p_direct_thickness[:, 0], theta_m, phi_m, title='Direct Thickness', fig=fig, ax=ax1, valmin=VMIN, valmax=VMAX,
)
fig, ax2 = plot_3D_directivity(
    p_scattered_thickness[:, 0], theta_m, phi_m, title='Scattered Thickness', fig=fig, ax=ax2, valmin=VMIN, valmax=VMAX,
)
fig, ax3 = plot_3D_directivity(
    p_direct_loading[:, 0],theta_m, phi_m, title='Direct Loading', fig=fig, ax=ax3, valmin=VMIN, valmax=VMAX,
)
fig, ax4 = plot_3D_directivity(
    p_scattered_loading[:, 0], theta_m, phi_m, title='Scattered Loading', fig=fig, ax=ax4, valmin=VMIN, valmax=VMAX,
)
fig, ax5 = plot_3D_directivity(
    p_total_scattering[:, 0], theta_m, phi_m, title='Scattering total', fig=fig, ax=ax5, valmin=VMIN, valmax=VMAX,
)
fig, ax6 = plot_3D_directivity(
    peq_data[0, :, :], np.deg2rad(theta_m_data), np.deg2rad(phi_m_data), title='Experiment (total)', fig=fig, ax=ax6, valmin=VMIN, valmax=VMAX,
)
fig.suptitle(f"Directivities of $\hat{{p}}_{{{ms[0] * B:.0f}}}$")


# 3D phase diagram
fig = plt.figure(figsize=(7, 7))
ax1 = fig.add_subplot(321, projection="3d")
ax2 = fig.add_subplot(322, projection="3d")
ax3 = fig.add_subplot(323, projection="3d")
ax4 = fig.add_subplot(324, projection="3d")
ax5 = fig.add_subplot(325, projection="3d")
ax6 = fig.add_subplot(326, projection="3d")


VMIN, VMAX = 10, 65
fig, ax1 = plot_3D_phase_directivity(
    p_direct_thickness[:, 0], theta_m, phi_m, title='Direct Thickness', fig=fig, ax=ax1, valmin=VMIN, valmax=VMAX,
)
fig, ax2 = plot_3D_phase_directivity(
    p_scattered_thickness[:, 0], theta_m, phi_m, title='Scattered Thickness', fig=fig, ax=ax2, valmin=VMIN, valmax=VMAX,
)
fig, ax3 = plot_3D_phase_directivity(
    p_direct_loading[:, 0],theta_m, phi_m,  title='Direct Loading', fig=fig, ax=ax3, valmin=VMIN, valmax=VMAX,
)
fig, ax4 = plot_3D_phase_directivity(
    p_scattered_loading[:, 0], theta_m, phi_m, title='Scattered Loading', fig=fig, ax=ax4, valmin=VMIN, valmax=VMAX,
)
fig, ax5 = plot_3D_phase_directivity(
    p_total_scattering[:, 0], theta_m, phi_m, title='Scattering total', fig=fig, ax=ax5, valmin=VMIN, valmax=VMAX,
)
fig, ax6 = plot_3D_phase_directivity(
    peq_data[0, :, :], np.deg2rad(theta_m_data), np.deg2rad(phi_m_data), title='Experiment (total)', fig=fig, ax=ax6, valmin=VMIN, valmax=VMAX,
)
fig.suptitle(f"Directivities of $\hat{{p}}_{{{ms[0] * B:.0f}}}$")
plt.show()
