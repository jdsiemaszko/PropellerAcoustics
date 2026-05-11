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
from Constants.helpers import read_force_file, plot_3D_directivity, plot_3D_phase_directivity, p_to_SPL, spl_from_autopower, plot_complex_curve
MODE = 'half'
Ntheta = 18
Nphi = 36


# BEGINNING OF HEADER
# vary configuration
from SourceMode.Configurations_NACA0012 import m_surface

from SourceMode.Configurations_NACA0012 import D20L20W00_D180 as sourceArray # pick configuration
SUFFIX = '_D180_MR'

sourceArray.numerics['CompactnessCorrection'] = True

NDIPOLES = sourceArray.Ndipoles
ms = np.array([2]) # harmonic to plot
phi_plot = 90 # phi_experimental to plot, in degrees

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

# our exp data, extracted from time signal
data = np.load('./Experimental/dataverse_files/D20L20_8000RPM.npz')

freqs = data['freqs']
phase = -data['phase']   # shape: (n_freq, n_theta, n_phi), mind we need to conjugate!
Pxx = data['Pxx']  # shape: (n_freq, n_theta, n_phi)
phi = 180 - data['phi'] # shape Nphi, mind switch from exp to numerical frame
theta = 90 - data['theta'] # shape Ntheta
radius = 1.62 # assume
phi_index = np.where(phi==phi_plot)[0][0]

# pick the right data
phi = phi[phi_index] # float!
Pxx = Pxx[ms[0]-1, :, phi_index] # harmonics from 1 to 10 along axis 0
phase = phase[ms[0]-1, :, phi_index]

x_cart = np.array([
    radius * np.cos(np.deg2rad(phi)) * np.sin(np.deg2rad(theta)),
    radius * np.sin(np.deg2rad(phi)) * np.sin(np.deg2rad(theta)),
    radius * np.cos(np.deg2rad(theta)) * np.ones_like(phi),
]) # observer positions, shape 3, Ntheta

PIN._numerics['include_vortex_sources'] = True
PIN._numerics['include_thickness_sources'] = False
beam_loading = PIN.getStrutLoadingHarmonics() 

p_beam_loading, _ = han.getPressureStator(x_cart, ms*B, beam_loading) # mind the indexing change for m
p_blade_loading, _ = han.getPressureRotor(x_cart, ms, BLH) 

p_blade_thickness, _ = han.getThicknessNoiseRotor(x_cart, ms, c * np.ones_like(r_inner), 0.082 * np.ones_like(r_inner)) # NACA0012

PIN._numerics['include_vortex_sources'] = False
PIN._numerics['include_thickness_sources'] = True
beam_loading = PIN.getStrutLoadingHarmonics() 

p_beam_thickness, _ = han.getPressureStator(x_cart, ms*B, beam_loading) # loading beam noise due to blade thickness, not to be confused with beam thickness noise, which is zero since the beam is stationary


PIN._numerics['include_vortex_sources'] = True
PIN._numerics['include_thickness_sources'] = True
beam_loading = PIN.getStrutLoadingHarmonics()



##### -------------------------------- SCATTERED LOADING NOISE ------------------------------------------
#### save gradients in the far-field (run once per observer and m)
for index, sm in enumerate(sourceArray.children):

    gradG_surface = np.load(f'./Data/current/NACA0012_rotor/gradG_surface_sm_{index}_{MODE}{SUFFIX}.npy') # shape (3, Nm, Nz, Ny)
    print(f'pre-computing far-field gradients {index+1}')

    gradG = sm.getScatteringGreenGradient(x_cart, m_surface * B * Omega / c0, gradG_surface) # shape (3, Nm, Nx, Ny)
    np.save(f'./Data/current/NACA0012_rotor/gradG_sm_{index}_{MODE}{SUFFIX}.npy', gradG)


# extract and rearrange
gradG_arr = np.zeros((sourceArray.seg_radius.shape[0], 3, ms.shape[0], x_cart.shape[1], NDIPOLES), dtype=np.complex128)
ind_m = np.where(m_surface == ms[0])[0][0]
for index, sm in enumerate(sourceArray.children):
    gradG_arr[index] = np.load(f'./Data/current/NACA0012_rotor/gradG_sm_{index}_{MODE}{SUFFIX}.npy')[:, ind_m, :, :].reshape(3, ms.shape[0], x_cart.shape[1], NDIPOLES)


p_scattered_loading = sourceArray.getScatteredPressure(x_cart, ms, gradG=gradG_arr)
p_direct_loading = sourceArray.getDirectPressure(x_cart, ms)

# p_scattered = np.load(f'./Data/current/NACA0012_rotor/p_scattered_{MODE}_m{int(m)}_{casename}{SUFFIX}.npy')
# p_direct_blade = np.load(f'./Data/current/NACA0012_rotor/p_direct_{MODE}_m{int(m)}_{casename}{SUFFIX}.npy')




#### -------------------------------- SCATTERED Thickness NOISE ------------------------------------------

### save gradients in the far-field (run once per observer and m)
for index, sm in enumerate(sourceArray.children):
    G_surface = np.load(f'./Data/current/NACA0012_rotor/G_surface_sm_{index}_{MODE}{SUFFIX}.npy') # shape (Nm, Nz, Ny)
    print(f'pre-computing far-field G {index+1}')

    G = sm.getScatteringGreen(x_cart, m_surface * B * Omega / c0, G_surface) # shape (Nm, Nx, Ny)
    np.save(f'./Data/current/NACA0012_rotor/G_sm_{index}_{MODE}{SUFFIX}.npy', G)

Nr = sourceArray.seg_radius.shape[0]

# G_arr = np.zeros((Nr, m_surface.shape[0], x_cart.shape[1], NDIPOLES), dtype=np.complex128)

G_arr = np.zeros((Nr, ms.shape[0], x_cart.shape[1], NDIPOLES), dtype=np.complex128) # pick only the ones we neeed
ind_m = np.where(m_surface == ms[0])[0][0]
for index, sm in enumerate(sourceArray.children):
    G_arr[index] = np.load(f'./Data/current/NACA0012_rotor/G_sm_{index}_{MODE}{SUFFIX}.npy')[ind_m, :, :] # extract only the m we need for plotting!


p_beam_total , _ = han.getPressureStator(x_cart, ms*B, beam_loading)

p_scattered_thickness = sourceArray.getThicknessPressureScattered(x_cart, ms, G=G_arr)
p_direct_thickness = sourceArray.getThicknessPressureDirect(x_cart, ms)


# scattering total
p_total_scattering = p_direct_thickness + p_direct_loading + p_scattered_loading + p_scattered_thickness

# pin total
p_total_pin = p_blade_loading + p_blade_thickness + p_beam_total
p_total_pin_loading = p_blade_loading + p_blade_thickness + p_beam_loading

phase_ref_scattering = np.angle(p_total_scattering[0])

# experimental
fig, ax = plot_complex_curve(theta, spl_from_autopower(Pxx), phase, valmax=65, valmin=10,
                             plot_kwargs={'color':'k', 'linestyle':None,'marker':'s', 'label':'Experimental'},)

# numerical
fig, ax = plot_complex_curve(theta, p_to_SPL(p_total_scattering),
        np.angle(p_total_scattering *np.exp(-1j * np.angle(p_total_scattering[0]))) # angle w.r.t x_cart[0] - i.e., the first microphone
        , valmax=65, valmin=10, fig=fig, ax=ax,
        plot_kwargs={'color':'r', 'linestyle':'dashed','marker':'s', 'label':'Scattering'})

fig, ax = plot_complex_curve(theta, p_to_SPL(p_total_pin),
        np.angle(p_total_scattering *np.exp(-1j * np.angle(p_total_pin[0]))) # angle w.r.t x_cart[0] - i.e., the first microphone
        , valmax=65, valmin=10, fig=fig, ax=ax,
        plot_kwargs={'color':'b', 'linestyle':'dashed','marker':'o', 'label':'PIN (incl. thickness)'})


fig, ax = plot_complex_curve(theta, p_to_SPL(p_total_pin_loading),
        np.angle(p_total_scattering *np.exp(-1j * np.angle(p_total_pin_loading[0]))) # angle w.r.t x_cart[0] - i.e., the first microphone
        , valmax=65, valmin=10, fig=fig, ax=ax,
        plot_kwargs={'color':'g', 'linestyle':'dashed','marker':'^', 'label':'PIN (loading only)'})
ax.legend()
plt.show()

