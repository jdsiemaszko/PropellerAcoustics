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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D

# BEGINNING OF HEADER
FILE='TOTAL'
MODE = 'half'
from Constants.helpers import read_force_file, plot_3D_directivity, plot_3D_phase_directivity, p_to_SPL, spl_from_autopower, plot_BPF_peaks
from Constants.data_assim import getGojonData
# vary configuration
from SourceMode.Configurations_NACA0012 import m_surface

# from SourceMode.Configurations_NACA0012 import D20L20W00_D360 as sourceArray # pick configuration
# SUFFIX = '_D360_HR'

# from SourceMode.Configurations_NACA0012 import D20L20W20_D180 as sourceArray # pick configuration
# SUFFIX = '_D20L20W20_D180'

from SourceMode.Configurations_NACA0012 import D20L20W00_D180 as sourceArray # pick configuration
SUFFIX = '_D180_MR'
shape='D'

# from SourceMode.Configurations_NACA0012 import D10L20W00_D180 as sourceArray # pick configuration
# SUFFIX = '_D10L20_D180'
# shape='D'

# from SourceMode.Configurations_NACA0012 import D15L20W00_D180 as sourceArray # pick configuration
# SUFFIX = 'D15L20_D180'
# shape='D'

# from SourceMode.Configurations_NACA0012 import PARROT_D20L20W00_D180 as sourceArray # pick configuration
# SUFFIX = 'PARROT_D20L20_D180'
# shape = 'PARROT'

# from SourceMode.Configurations_NACA0012 import PARROT_D20L20W00_D36_1_10 as sourceArray # pick configuration
# SUFFIX = 'PARROT_D20L20_D36_1_10'
# shape = 'PARROT'

# from SourceMode.Configurations_NACA0012 import PARROT_D20L20W00_D36_5_10 as sourceArray # pick configuration
# SUFFIX = 'PARROT_D20L20_D36_5_10'
# shape = 'PARROT'

# from SourceMode.Configurations_NACA0012 import PARROT_D20L21W00_D180 as sourceArray # pick configuration
# SUFFIX = 'PARROT_D20L20_D180_v2'
# shape = 'PARROT'

# from SourceMode.Configurations_NACA0012 import PARROT_D20L20W00_D180_10_37 as sourceArray # pick configuration
# SUFFIX = 'PARROT_D20L20_D180_10_37'
# shape = 'PARROT'

# from SourceMode.Configurations_NACA0012 import PARROT_D20L20W00_D180_10_37 as sourceArray # pick configuration
# SUFFIX = 'PARROT_D20L20_D180_10_37'
# shape = 'PARROT'

# from SourceMode.Configurations_NACA0012 import PARROT_D20L20W00_D360 as sourceArray # pick configuration
# SUFFIX = 'PARROT_D20L20_D360'
# shape = 'PARROT'

# from SourceMode.Configurations_NACA0012 import D20L20W00_D180_6000RPM as sourceArray
# SUFFIX = 'D20L20_D180_6000RPM'
# shape='D'

# from SourceMode.Configuration_Porous_NACA0012 import D20L20_porous as sourceArray
# SUFFIX = 'D20L20_POROUS_v2'
# shape='D'

# from SourceMode.Configuration_Porous_NACA0012 import D20L20_porous as sourceArray
# SUFFIX = 'D20L20_POROUS_v3'
# shape='D'

# sourceArray.numerics['CompactnessCorrection'] = True
sourceArray.numerics['CompactnessCorrection'] = False


NDIPOLES = sourceArray.Nsources

r_inner, Fz, Fphi  = read_force_file('./Data/Zamponi2026/FS_ISAE_2_8000.txt') # reuse the radial stations from data
Omega_ref = 8000/60*2*np.pi
if shape == "PARROT":

    rt, t =  np.loadtxt('./Data/Parrot2024/thrust_Npm.csv', skiprows=1, delimiter=',').T # radius/r1, thrust in Npm
    rq, q =  np.loadtxt('./Data/Parrot2024/torque_Nmpm.csv', skiprows=1, delimiter=',').T # radius/r1, torque in Nmpm

    q /= 1.125

    r_inner = sourceArray.seg_radius
    r1 = sourceArray.r1
    Fz = np.interp(r_inner/r1, rt, t) # same radial array
    Q = np.interp(r_inner/r1, rq, q) 
    Fphi = Q / r_inner

    TTARGET = 2.15 / sourceArray.B # Newtons
    QTARGET = 25 / 1000 / sourceArray.B # Newton-radian-meters
    Fz *= TTARGET / np.trapezoid(Fz, r_inner)  # rescale to target
    Fphi *= QTARGET / np.trapezoid(Fphi * r_inner, r_inner) # rescale to target

# rescale loading!
elif shape == 'D' and sourceArray.Omega != Omega_ref:
    print(f'rescaling the loading from {Omega_ref} to {sourceArray.Omega} rad/s')
    Fz *= (sourceArray.Omega/Omega_ref)**2
    Fphi *= (sourceArray.Omega/Omega_ref)**2

    
D_bras = sourceArray.green.radius * 2
g = -1 * sourceArray.green.origin[2]
BLH, BLH_S, BLH_US, _ = sourceArray.getLoading(Fz, Fphi, D_bras, g, steady_only=False) # compute loading on the fly, return PIN for reuse
PIN = sourceArray.PIN

# g= 0.02
B = sourceArray.B
c = sourceArray.chord
Omega = sourceArray.Omega
if shape == 'PARROT':
    Omega *= -1

c0 = sourceArray.SoS
han = sourceArray.getHanson()
# END OF HEADER

ind_theta = 6     # 60 to -60 in 10
ind_phi = 9          # 0 to 350 in 10
datadir = './Experimental/dataverse_files'
# casefile = f'ISAE_2_D{int(1000*D_bras)}_L{int(1000*g)}'

data, BPF, freq, x_cart_data, theta_data, phi_data, theta_exp, phi_exp, casefile = getGojonData(datadir, D_bras, g, shape=shape, B=2, 
                                                                                                RPM=int(Omega * 60/2/np.pi)
                                                                                                )
data = data[:, ind_theta, ind_phi]
x_cart = x_cart_data[:, ind_theta, ind_phi].reshape((3, 1))
theta = theta_data[ind_theta]
phi = phi_data[ind_phi]

print(theta, phi)

Nr = len(r_inner)
ms = np.arange(1, 11, 1) # harmonics to extract



# pSmB_model_rotor = han.getPressureRotor(x_cart, ms, 
#                                     #    blade_l.getBladeLoadingHarmonics()
#                                     BLH_S
#                                        )[0][0]

# pUSmB_model_rotor = han.getPressureRotor(x_cart, ms, 
#                                     #    blade_l.getBladeLoadingHarmonics()
#                                     BLH_US
#                                        )[0][0]
# pLmB_model_rotor = pSmB_model_rotor + pUSmB_model_rotor
# SPL_rotor_S = p_to_SPL(pSmB_model_rotor)
# SPL_rotor_US = p_to_SPL(pUSmB_model_rotor)

# BLH_S[2, :, :] = 0
# BLH_US[2, :, :] = 0

pLUSmB_model_rotor = han.getPressureRotor(x_cart, ms, 
                                    #    blade_l.getBladeLoadingHarmonics()
                                    BLH_US
                                       )[0][0]

pLSmB_model_rotor = han.getPressureRotor(x_cart, ms, 
                                    #    blade_l.getBladeLoadingHarmonics()
                                    BLH_S
                                       )[0][0]



ptmB_model_rotor = han.getThicknessNoiseRotor(x_cart, ms, sourceArray.seg_chord, 0.0822 * np.ones_like(r_inner))[0][0] # NACA0012
# BL  =  beam_l.getBeamLoadingHarmonics(BLH=BLH)


PIN._numerics['only_linear'] = True 
PIN._numerics['only_nonlinear'] = False


PIN._numerics['include_vortex_sources'] = True
PIN._numerics['include_thickness_sources'] = False
BL = PIN.getStrutLoadingHarmonics()
pmB_model_beam_loading = han.getPressureStator(x_cart, ms*B, BL)[0][0]

PIN._numerics['include_vortex_sources'] = False
PIN._numerics['include_thickness_sources'] = True
BL = PIN.getStrutLoadingHarmonics()
pmB_model_beam_thickness = han.getPressureStator(x_cart, ms*B, BL)[0][0]

PIN._numerics['include_vortex_sources'] = True
PIN._numerics['include_thickness_sources'] = True
BL = PIN.getStrutLoadingHarmonics()
pmB_model_beam_total = han.getPressureStator(x_cart, ms*B, BL)[0][0]


pmB_model_rotor_total = pLSmB_model_rotor + pLUSmB_model_rotor + ptmB_model_rotor
pmB_model_rotor_loading = pLSmB_model_rotor + pLUSmB_model_rotor
pmB_model_total = pLSmB_model_rotor + pLUSmB_model_rotor + ptmB_model_rotor + pmB_model_beam_total # assuming coherent
# pmB_model_total = np.sqrt(np.abs(pmB_model_rotor_total)**2 + np.abs(pmB_model_beam)**2) # assuming incoherent



Nchildren = len(sourceArray.children)
# SUFFIX = '

# -------------------------------- SCATTERED LOADING NOISE ------------------------------------------
# save gradients in the far-field (run once per observer and m)
for index, sm in enumerate(sourceArray.children):

    gradG_surface = np.load(f'./Data/current/NACA0012_rotor/gradG_surface_sm_{index}_{MODE}{SUFFIX}.npy') # shape (3, Nm, Nz, Ny)
    print(f'pre-computing far-field gradients {index+1}')

    gradG = sm.getScatteringGreenGradient(x_cart, ms*B * np.abs(Omega)  / c0, gradG_surface) # shape (3, Nm, Nx, Ny)
    np.save(f'./Data/current/NACA0012_rotor/gradG_sm_{index}_{MODE}_{ind_theta}_{ind_phi}_{FILE}{SUFFIX}.npy', gradG)



# extract and rearrange

gradG_arr = np.zeros((Nchildren, 3, ms.shape[0], x_cart.shape[1], NDIPOLES), dtype=np.complex128)
for index, sm in enumerate(sourceArray.children):
    gradG_arr[index] = np.load(f'./Data/current/NACA0012_rotor/gradG_sm_{index}_{MODE}_{ind_theta}_{ind_phi}_{FILE}{SUFFIX}.npy')



# p_direct_s = sourceArray.getDirectPressure(x_cart, ms, BLH=np.transpose(BLH_S, axes=[2, 0, 1]))[0]
# p_direct_us = sourceArray.getDirectPressure(x_cart, ms, BLH=np.transpose(BLH_US, axes=[2, 0, 1]))[0]

sourceArray.updateBLH(BLH_US)
p_scattered_us = sourceArray.getScatteredPressure(x_cart, ms, gradG=gradG_arr)[0]
p_direct_us = sourceArray.getDirectPressure(x_cart, ms)[0]

sourceArray.updateBLH(BLH_S)
p_scattered_s = sourceArray.getScatteredPressure(x_cart, ms, gradG=gradG_arr)[0]
p_direct_s = sourceArray.getDirectPressure(x_cart, ms)[0]




# np.save(f'./Data/current/NACA0012_rotor/p_s_spectrum_{MODE}_{ind_theta}_{ind_phi}.npy', p_scattered)
# p_scattered = np.load(f'./Data/current/NACA0012_rotor/p_s_spectrum_{MODE}_{ind_theta}_{ind_phi}.npy')


# -------------------------------- SCATTERED Thickness NOISE ------------------------------------------

# save gradients in the far-field (run once per observer and m)
for index, sm in enumerate(sourceArray.children):

    G_surface = np.load(f'./Data/current/NACA0012_rotor/G_surface_sm_{index}_{MODE}{SUFFIX}.npy') # shape (Nm, Nz, Ny)
    print(f'pre-computing far-field gradients {index+1}')

    G = sm.getScatteringGreen(x_cart, ms*B * np.abs(Omega)  / c0, G_surface) # shape (Nm, Nx, Ny)
    np.save(f'./Data/current/NACA0012_rotor/G_sm_{index}_{MODE}_{ind_theta}_{ind_phi}_{FILE}{SUFFIX}.npy', G)


G_arr = np.zeros((Nchildren, ms.shape[0], x_cart.shape[1], NDIPOLES), dtype=np.complex128)
for index, sm in enumerate(sourceArray.children):
    G_arr[index] = np.load(f'./Data/current/NACA0012_rotor/G_sm_{index}_{MODE}_{ind_theta}_{ind_phi}_{FILE}{SUFFIX}.npy')


p_scattered_thickness = sourceArray.getThicknessPressureScattered(x_cart, ms, G=G_arr)[0]
p_direct_thickness = sourceArray.getThicknessPressureDirect(x_cart, ms)[0]

# np.save(f'./Data/current/NACA0012_rotor/p_s_spectrum_thickness_{MODE}_{ind_theta}_{ind_phi}.npy', p_scattered_thickness)
# p_scattered_thickness = np.load(f'./Data/current/NACA0012_rotor/p_s_spectrum_thickness_{MODE}_{ind_theta}_{ind_phi}.npy')

for element in [p_direct_s, p_direct_us, p_scattered_s, p_scattered_us, p_direct_thickness, p_scattered_thickness]:
    element[np.isnan(element)] = 0.0

p_total_scattering = p_direct_s + p_direct_us + p_scattered_s + p_scattered_us + p_direct_thickness + p_scattered_thickness
p_rotor_total = p_direct_s + p_direct_us + p_direct_thickness
p_total_minus_scattered_thickness = p_total_scattering - p_scattered_thickness

# SPLS PIN MODEL
SPL_rotor_loading = p_to_SPL(pmB_model_rotor_loading)
SPL_rotor_S = p_to_SPL(pLSmB_model_rotor)
SPL_rotor_US = p_to_SPL(pLUSmB_model_rotor)

SPL_rotor_total = p_to_SPL(pmB_model_rotor_total)
SPL_rotor_thickness = p_to_SPL(ptmB_model_rotor)
SPL_beam_loading = p_to_SPL(pmB_model_beam_loading)
SPL_beam_thickness = p_to_SPL(pmB_model_beam_thickness)

SPL_total_PIN = p_to_SPL(pmB_model_total)

# SPLS TOTAL SCATTERING
SPL_direct_s = p_to_SPL(p_direct_s)
SPL_direct_us = p_to_SPL(p_direct_us)
SPL_scattered_s = p_to_SPL(p_scattered_s)
SPL_scattered_us = p_to_SPL(p_scattered_us)
SPL_scattered = p_to_SPL(p_scattered_s + p_scattered_us)
SPL_direct_thickness = p_to_SPL(p_direct_thickness)
SPL_scattered_thickness = p_to_SPL(p_scattered_thickness)
SPL_SM_rotor_total = p_to_SPL(p_rotor_total)
SPL_total_scattering = p_to_SPL(p_total_scattering)
SPL_total_scattering_minus_scattered_thickness = p_to_SPL(p_total_minus_scattered_thickness)

# SPL_total = p_to_SPL(p_rms_total) # same computation

fig, ax = plt.subplots(figsize=(12, 5))

# ax.plot(ms, SPL_rotor_S, label=f"Steady Loading Noise (PIN)", color='r', marker='^')
# ax.plot(ms, SPL_rotor_US, label=f"Unsteady Loading Noise (PIN)", color='g', marker='^')

# ax.plot(ms, SPL_rotor_S, label=f"Steady Loading Noise (PIN)", color='r', marker='^')
# ax.plot(ms, SPL_rotor_US, label=f"Unsteady Loading Noise (PIN)", color='g', marker='^')

# ax.plot(ms, SPL_rotor_thickness, label=f"Thickness Noise (PIN)", color='b', marker='^')
# ax.plot(ms, SPL_rotor_total, label=f"Rotor Total (PIN)", color='y', marker='^')
# ax.plot(ms, SPL_beam_loading, label=f"Beam Loading due to Blade Loading", color='m', marker='^')
# ax.plot(ms, SPL_beam_thickness, label=f"Beam Loading due to Blade Thickness", color='c', marker='^')
# ax.plot(ms, SPL_total_PIN, label=f"Total (PIN)", color='k', marker='^')
# ax.plot(freq[0] / BPF, spl_from_autopower(data), label=f"Experimental, total", color='k', alpha=0.75)
# fig, ax = plot_BPF_peaks(fig, ax, freq[0] / BPF, spl_from_autopower(data), N0=1, N1= 25, range=0.01, 
#                          plot_kwargs={
#                              'color':'k',
#                              'linestyle':'dashed',
#                              'alpha':0.75
#                          })

# ax.plot(ms, SPL_direct_s, label=f"Steady Loading Noise (SM)", color='r', marker='s', linestyle='dashed')
# ax.plot(ms, SPL_direct_us, label=f"Unsteady Loading Noise (SM)", color='g', marker='s', linestyle='dashed')

# ax.plot(ms, SPL_direct_thickness, label=f"Thickness Noise (SM)", color='b', marker='s', linestyle='dashed')
# ax.plot(ms, SPL_SM_rotor_total, label=f"Rotor Total (SM)", color='y', marker='s', linestyle='dashed')
# # ax.plot(ms, SPL_scattered_s, label=f"Scattered Steady Loading Noise", color='m', marker='s', linestyle='dashed')
# # ax.plot(ms, SPL_scattered_us, label=f"Scattered Unsteady Loading Noise", color='tab:pink', marker='s', linestyle='dashed')
# ax.plot(ms, SPL_scattered, label=f"Scattered Loading Noise", color='m', marker='s', linestyle='dashed')
# ax.plot(ms, SPL_scattered_thickness, label=f"Scattered Thickness Noise", color='c', marker='s', linestyle='dashed')
# ax.plot(ms, SPL_total_scattering, label=f"Total (Scattering)", color='k', marker='s', linestyle='dashed')




# --- plotting ---
ax.plot(ms, SPL_rotor_S, color='r', marker='^')
ax.plot(ms, SPL_rotor_US, color='g', marker='^')
ax.plot(ms, SPL_rotor_thickness, color='b', marker='^')
ax.plot(ms, SPL_rotor_total, color='y', marker='^')
ax.plot(ms, SPL_beam_loading, color='m', marker='^')
ax.plot(ms, SPL_beam_thickness, color='c', marker='^')
ax.plot(ms, SPL_total_PIN, color='k', marker='^')

ax.plot(ms, SPL_direct_s, color='r', marker='s', linestyle='--')
ax.plot(ms, SPL_direct_us, color='g', marker='s', linestyle='--')
ax.plot(ms, SPL_direct_thickness, color='b', marker='s', linestyle='--')
ax.plot(ms, SPL_SM_rotor_total, color='y', marker='s', linestyle='--')
ax.plot(ms, SPL_scattered, color='m', marker='s', linestyle='--')
ax.plot(ms, SPL_scattered_thickness, color='c', marker='s', linestyle='--')
ax.plot(ms, SPL_total_scattering, color='k', marker='s', linestyle='--')

ax.plot(freq[0]/BPF,
        spl_from_autopower(data),
        color='0.3',
        linewidth=2)

fig, ax = plot_BPF_peaks(fig, ax, freq[0] / BPF, spl_from_autopower(data), N0=1, N1= 25, range=0.01, 
                         plot_kwargs={
                             'color':'k',
                             'linestyle':'dashed',
                             'alpha':1.0,
                             'linewidth': 2
                         })

component_handles = [
    Line2D([0], [0], color='r', lw=2, label='Steady Loading'),
    Line2D([0], [0], color='g', lw=2, label='Unsteady Loading'),
    Line2D([0], [0], color='b', lw=2, label='Thickness'),
    Line2D([0], [0], color='y', lw=2, label='Rotor Total'),
    Line2D([0], [0], color='m', lw=2, label='Beam Noise due to Loading'),
    Line2D([0], [0], color='c', lw=2, label='Beam Noise due to Thickness'),
    Line2D([0], [0], color='k', lw=2, label='Total'),
]

model_handles = [
    Line2D([0], [0], color='k', marker='^', linestyle='-',
           label='PIN'),
    Line2D([0], [0], color='k', marker='s', linestyle='--',
           label='SM'),
    Line2D([0], [0], color='0.3', lw=3,
           label='Experiment'),
]

leg1 = ax.legend(handles=component_handles,
                 title='Noise Component',
                 loc='upper left')

leg2 = ax.legend(handles=model_handles,
                #  title='Model',
                 loc='lower left')

ax.add_artist(leg1)
ax.add_artist(leg2)


# ax.plot(ms, SPL_total_scattering_minus_scattered_thickness, label=f"Direct+Scattering (Minus Thickness)", color='g', marker='s', linestyle='dashed')


# ax.legend(ncol=2, loc='upper left', fontsize=8)
ax.set_xlabel("$f^+ = f/B/\Omega$ (Hz)")
ax.set_ylabel("SPL (dB)")
ax.set_xscale('log')

ax.grid(visible=True, which='major', color='k', linestyle='-')
ax.grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.5)
# ax.set_title(f'Theta = {theta} deg, Phi = {phi} deg')
# plt.xlim(0.03333, 100)
plt.xlim(0.1, 100)

plt.ylim(0, 75)
plt.tight_layout()
plt.show()
