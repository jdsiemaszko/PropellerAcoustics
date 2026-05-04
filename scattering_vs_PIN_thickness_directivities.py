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

from scattering_vs_PIN import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from Constants.data_assim import getGojonData
SUFFIX = '_D360_HR'
# SUFFIX = '_HIGHRES'
m_surface = np.arange(1, 11, 1) # depending on the datafile chosen


ms = np.array([2]) # plot harmonic m of BPF

# datadir = './Experimental/dataverse_files'
# casefile = f'ISAE_2_D{int(1000*D_bras)}_L{int(1000*g)}'

# data, BPF, freq, x_cart, theta, phi, theta_exp, phi_exp, casefile = getGojonData(datadir, D_bras, g, shape='D', B=2, RPM=8000)

# Ntheta, Nphi = x_cart.shape[1], x_cart.shape[2]
# x_cart_flat = x_cart.reshape(3, Ntheta, Nphi, order='C')

# angular coordinates
theta = np.linspace(0.0, np.pi, Ntheta, endpoint=True)
phi   = np.linspace(0.0, 2.0 * np.pi, Nphi, endpoint=True)

# 2D mesh
theta_m, phi_m = np.meshgrid(theta, phi, indexing='ij')
# shapes: (Ntheta, Nphi)

# flatten
R_arr     = np.full(theta_m.size, R)
theta_arr = theta_m.ravel()
phi_arr   = phi_m.ravel()

X = R_arr * np.sin(theta_arr) * np.cos(phi_arr)
Y = R_arr * np.sin(theta_arr) * np.sin(phi_arr)
Z = R_arr * np.cos(theta_arr)

x_cart = np.array([X, Y, Z])

# -------------------------------- SCATTERED Thickness NOISE ------------------------------------------

# save gradients in the far-field (run once per observer and m)

for index, sm in enumerate(sourceArray.children):

    G_surface = np.load(f'./Data/current/NACA0012_rotor/G_surface_sm_{index}_{MODE}{SUFFIX}.npy') # shape (Nm, Nz, Ny)
    print(f'pre-computing far-field G {index+1}')

    G = sm.getScatteringGreen(x_cart, m_surface * B * Omega / c0, G_surface) # shape (Nm, Nx, Ny)
    np.save(f'./Data/current/NACA0012_rotor/G_sm_{index}_{MODE}.npy', G)

Nr = sourceArray.seg_radius.shape[0]

# G_arr = np.zeros((Nr, m_surface.shape[0], x_cart.shape[1], NDIPOLES), dtype=np.complex128)

G_arr = np.zeros((Nr, ms.shape[0], x_cart.shape[1], NDIPOLES), dtype=np.complex128) # pick only the ones we neeed
ind_m = np.where(m_surface == ms[0])[0][0]
for index, sm in enumerate(sourceArray.children):
    G_arr[index] = np.load(f'./Data/current/NACA0012_rotor/G_sm_{index}_{MODE}.npy')[ind_m, :, :] # extract only the m we need for plotting!

p_thickness_hanson, _ = han.getThicknessNoiseRotor(x_cart, ms, c * np.ones_like(r_inner), 0.082 * np.ones_like(r_inner)) # NACA0012

# BL  =  beam_l.getBeamLoadingHarmonics(BLH=BLH)
PIN._numerics['include_vortex_sources'] = False
PIN._numerics['include_thickness_sources'] = True
             
BL = PIN.getStrutLoadingHarmonics()
p_thickness_beam, _ = han.getPressureStator(x_cart, ms*B, BL) # loading beam noise due to blade thickness!

p_scattered_thickness = sourceArray.getThicknessPressureScattered(x_cart, ms, G=G_arr)
p_direct_thickness = sourceArray.getThicknessPressureDirect(x_cart, ms)


# 3D SPL diagram
fig = plt.figure(figsize=(7, 7))
ax1 = fig.add_subplot(221, projection="3d")
ax2 = fig.add_subplot(222, projection="3d")
ax3 = fig.add_subplot(223, projection="3d")
ax4 = fig.add_subplot(224, projection="3d")

VMIN, VMAX = 10, 65
fig, ax1 = plot_3D_directivity(
    p_thickness_hanson[:, 0], theta_m, phi_m, title='blade direct thickness noise (Hanson)', fig=fig, ax=ax1, valmin=VMIN, valmax=VMAX,
)
fig, ax2 = plot_3D_directivity(
    p_direct_thickness[:, 0], theta_m, phi_m, title='blade direct thickness noise (Scattering)', fig=fig, ax=ax2, valmin=VMIN, valmax=VMAX,
)
fig, ax3 = plot_3D_directivity(
    p_thickness_beam[:, 0], theta_m, phi_m, title='blade loading noise due to beam thickness', fig=fig, ax=ax3, valmin=VMIN, valmax=VMAX,
)
fig, ax4 = plot_3D_directivity(
    p_scattered_thickness[:, 0], theta_m, phi_m,
    title='scattered blade thickness noise', fig=fig, ax=ax4, valmin=VMIN, valmax=VMAX,
)
plt.show()


# 3D phase diagram
fig = plt.figure(figsize=(7, 7))
ax1 = fig.add_subplot(221, projection="3d")
ax2 = fig.add_subplot(222, projection="3d")
ax3 = fig.add_subplot(223, projection="3d")
ax4 = fig.add_subplot(224, projection="3d")


fig, ax1 = plot_3D_phase_directivity(
    p_thickness_hanson[:, 0], theta_m, phi_m, title='blade direct thickness noise (Hanson)', fig=fig, ax=ax1, valmin=VMIN, valmax=VMAX,
)
fig, ax2 = plot_3D_phase_directivity(
    p_direct_thickness[:, 0], theta_m, phi_m, title='blade direct thickness noise (Scattering)', fig=fig, ax=ax2, valmin=VMIN, valmax=VMAX,
)
fig, ax3 = plot_3D_phase_directivity(
    p_thickness_beam[:, 0], theta_m, phi_m, title='blade loading noise due to beam thickness', fig=fig, ax=ax3, valmin=VMIN, valmax=VMAX,
)
fig, ax4 = plot_3D_phase_directivity(
    p_scattered_thickness[:, 0], theta_m, phi_m,
    title='scattered blade thickness noise', fig=fig, ax=ax4, valmin=VMIN, valmax=VMAX,
)
plt.show()
