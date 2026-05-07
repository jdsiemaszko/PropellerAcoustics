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
from Constants.data_assim import getGojonData, getHarmonicsFromData

SUFFIX = '_D360_HR'
# SUFFIX = '_D180_MR'
# SUFFIX = '_D90_LR'
m_surface = np.arange(1, 11, 1) # harmonics to extract
sourceArray.numerics['CompactnessCorrection'] = True
# sourceArray.numerics['CompactnessCorrection'] = False


NDIPOLES = sourceArray.Ndipoles

ms = np.array([2])

datadir = './Experimental/dataverse_files'
# casefile = f'ISAE_2_D{int(1000*D_bras)}_L{int(1000*g)}'

data, BPF, freq, x_cart_data, theta_data, phi_data, theta_exp, phi_exp, casefile = getGojonData(datadir, D_bras, g, shape='D', B=2, RPM=8000)

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
R_arr     = np.full(theta_m.size, R)
theta_arr = theta_m.ravel()
phi_arr   = phi_m.ravel()

X = R_arr * np.sin(theta_arr) * np.cos(phi_arr)
Y = R_arr * np.sin(theta_arr) * np.sin(phi_arr)
Z = R_arr * np.cos(theta_arr)

x_cart = np.array([X, Y, Z])

##### -------------------------------- SCATTERED LOADING NOISE ------------------------------------------
##### save gradients in the far-field (run once per observer and m)
# for index, sm in enumerate(sourceArray.children):

#     gradG_surface = np.load(f'./Data/current/NACA0012_rotor/gradG_surface_sm_{index}_{MODE}{SUFFIX}.npy') # shape (3, Nm, Nz, Ny)
#     print(f'pre-computing far-field gradients {index+1}')

#     gradG = sm.getScatteringGreenGradient(x_cart, m_surface * B * Omega / c0, gradG_surface) # shape (3, Nm, Nx, Ny)
#     np.save(f'./Data/current/NACA0012_rotor/gradG_sm_{index}_{MODE}{SUFFIX}.npy', gradG)


# extract and rearrange
gradG_arr = np.zeros((sourceArray.seg_radius.shape[0], 3, ms.shape[0], x_cart.shape[1], NDIPOLES), dtype=np.complex128)
ind_m = np.where(m_surface == ms[0])[0][0]
for index, sm in enumerate(sourceArray.children):
    gradG_arr[index] = np.load(f'./Data/current/NACA0012_rotor/gradG_sm_{index}_{MODE}{SUFFIX}.npy')[:, ind_m, :, :].reshape(3, ms.shape[0], x_cart.shape[1], NDIPOLES)


p_scattered_loading = sourceArray.getScatteredPressure(x_cart, ms, gradG=gradG_arr)
p_direct_loading = sourceArray.getDirectPressure(x_cart, ms)

# p_scattered = np.load(f'./Data/current/NACA0012_rotor/p_scattered_{MODE}_m{int(m)}_{casename}{SUFFIX}.npy')
# p_direct_blade = np.load(f'./Data/current/NACA0012_rotor/p_direct_{MODE}_m{int(m)}_{casename}{SUFFIX}.npy')

PIN._numerics['include_vortex_sources'] = True
PIN._numerics['include_thickness_sources'] = False
beam_loading = PIN.getStrutLoadingHarmonics() 

p_beam_loading, _ = han.getPressureStator(x_cart, ms*B, beam_loading) # mind the indexing change for m
p_blade_loading, _ = han.getPressureRotor(x_cart, ms, BLH) 



#### -------------------------------- SCATTERED Thickness NOISE ------------------------------------------

#### save gradients in the far-field (run once per observer and m)

# for index, sm in enumerate(sourceArray.children):
#     G_surface = np.load(f'./Data/current/NACA0012_rotor/G_surface_sm_{index}_{MODE}{SUFFIX}.npy') # shape (Nm, Nz, Ny)
#     print(f'pre-computing far-field G {index+1}')

#     G = sm.getScatteringGreen(x_cart, m_surface * B * Omega / c0, G_surface) # shape (Nm, Nx, Ny)
#     np.save(f'./Data/current/NACA0012_rotor/G_sm_{index}_{MODE}{SUFFIX}.npy', G)

Nr = sourceArray.seg_radius.shape[0]

# G_arr = np.zeros((Nr, m_surface.shape[0], x_cart.shape[1], NDIPOLES), dtype=np.complex128)

G_arr = np.zeros((Nr, ms.shape[0], x_cart.shape[1], NDIPOLES), dtype=np.complex128) # pick only the ones we neeed
ind_m = np.where(m_surface == ms[0])[0][0]
for index, sm in enumerate(sourceArray.children):
    G_arr[index] = np.load(f'./Data/current/NACA0012_rotor/G_sm_{index}_{MODE}{SUFFIX}.npy')[ind_m, :, :] # extract only the m we need for plotting!

p_blade_thickness, _ = han.getThicknessNoiseRotor(x_cart, ms, c * np.ones_like(r_inner), 0.082 * np.ones_like(r_inner)) # NACA0012

# BL  =  beam_l.getBeamLoadingHarmonics(BLH=BLH)

PIN._numerics['include_vortex_sources'] = False
PIN._numerics['include_thickness_sources'] = True
beam_loading = PIN.getStrutLoadingHarmonics() 

p_beam_thickness, _ = han.getPressureStator(x_cart, ms*B, beam_loading) # loading beam noise due to blade thickness, not to be confused with beam thickness noise, which is zero since the beam is stationary


PIN._numerics['include_vortex_sources'] = True
PIN._numerics['include_thickness_sources'] = True
beam_loading = PIN.getStrutLoadingHarmonics()

p_beam_total , _ = han.getPressureStator(x_cart, ms*B, beam_loading)

p_scattered_thickness = sourceArray.getThicknessPressureScattered(x_cart, ms, G=G_arr)
p_direct_thickness = sourceArray.getThicknessPressureDirect(x_cart, ms)


# scattering total
p_total_scattering = p_direct_thickness + p_direct_loading + p_scattered_loading + p_scattered_thickness

# pin total
p_total_pin = p_blade_loading + p_blade_thickness + p_beam_total
p_total_pin_loading = p_blade_loading + p_blade_thickness + p_beam_loading

# + p_beam_total 
# + p_beam_loading + p_beam_thickness

# 3D SPL diagram
fig = plt.figure(figsize=(7, 7))
ax1 = fig.add_subplot(221, projection="3d")
ax2 = fig.add_subplot(222, projection="3d")
ax3 = fig.add_subplot(223, projection="3d")
ax4 = fig.add_subplot(224, projection="3d")

VMIN, VMAX = 10, 65
fig, ax1 = plot_3D_directivity(
    p_total_pin_loading[:, 0], theta_m, phi_m, title='PIN Model (vortex only)', fig=fig, ax=ax1, valmin=VMIN, valmax=VMAX,
)
fig, ax2 = plot_3D_directivity(
    p_total_scattering[:, 0], theta_m, phi_m, title='Scattering Model', fig=fig, ax=ax2, valmin=VMIN, valmax=VMAX,
)
fig, ax3 = plot_3D_directivity(
    peq_data[0, :, :], np.deg2rad(theta_m_data), np.deg2rad(phi_m_data), title='Experiement', fig=fig, ax=ax3, valmin=VMIN, valmax=VMAX,
)
fig, ax4 = plot_3D_directivity(
    p_total_pin[:, 0], theta_m, phi_m, title='PIN Model (incl. blade thickness)', fig=fig, ax=ax4, valmin=VMIN, valmax=VMAX,
)
fig.suptitle(f"Directivities of $\hat{{p}}_{{{ms[0] * B:.0f}}}$")
plt.show()


# 3D phase diagram
fig = plt.figure(figsize=(7, 7))
ax1 = fig.add_subplot(221, projection="3d")
ax2 = fig.add_subplot(222, projection="3d")
ax3 = fig.add_subplot(223, projection="3d")
ax4 = fig.add_subplot(224, projection="3d")

VMIN, VMAX = 10, 65
fig, ax1 = plot_3D_phase_directivity(
    p_total_pin_loading[:, 0], theta_m, phi_m, title='PIN Model (vortex only)', fig=fig, ax=ax1, valmin=VMIN, valmax=VMAX,
)
fig, ax2 = plot_3D_phase_directivity(
    p_total_scattering[:, 0], theta_m, phi_m, title='Scattering Model', fig=fig, ax=ax2, valmin=VMIN, valmax=VMAX,
)
fig, ax3 =  plot_3D_phase_directivity(
    peq_data[0, :, :], np.deg2rad(theta_m_data), np.deg2rad(phi_m_data), title='Experiement', fig=fig, ax=ax3, valmin=VMIN, valmax=VMAX,
)
fig, ax4 =  plot_3D_phase_directivity(
    p_total_pin[:, 0], theta_m, phi_m, title='PIN Model (incl. blade thickness)', fig=fig, ax=ax4, valmin=VMIN, valmax=VMAX,
)
fig.suptitle(f"Directivities of $\hat{{p}}_{{{ms[0] * B:.0f}}}$")
plt.show()
