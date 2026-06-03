"""
compute the scattered green's function and its gradient on a surface of a scattering geometry
save the results to external files with unique identifier for reuse
"""
import numpy as np
from Constants.helpers import read_force_file
import time
import matplotlib.pyplot as plt

# manually input everything, couldn't be bothered
# from SourceMode.Configurations_NACA0012 import *
# sourceArray = D15L20W00_D180
# SUFFIX = 'D15L20_D180'

from SourceMode.Configurations_NACA0012 import PARROT_D20L20W00_D180_10_37 as sourceArray
SUFFIX = 'PARROT_D20L20_D180_10_37'

ms = np.arange(1, 11, 1)
MODE = 'half'

# # ------------------- Inputs -----------------------------
# SUFFIX = '_D20L20W10_D180'
# # SUFFIX = '_D10L20_D180'
# # SUFFIX = '_D360_HR'
# Ndipoles = 180         # 360 should be accurate up to m ~ 18?
# Nk = 40 # number of resolved loading harmonics, max frequency is Nk * Omega
# D_prop = 0.2
# D = 20 / 1000
# L = 20 / 1000
# W = 10 / 1000
# Omega = 8000 / 60 * 2 * np.pi
# NBLADES = 2
# c0 = 340.
# rho0 = 1.2

# MODE='half'

# ms = np.arange(1, 11, 1) # example, anything more than 10 is likely overkill and would require A LOT of dipoles to resolve!
 
# axis_prop = np.array([0.0, 0.0, 1.0]) # z-direction propeller...
# origin_prop = np.array([0.0, 0.0, 0.0]) # ... at z=0



# r_inner, _, _ = read_force_file('./Data/Zamponi2026/FS_ISAE_2_8000.txt') # reuse the radial stations from data
# dr = np.diff(r_inner)[0]
# r_outer = np.hstack([r_inner-dr/2, r_inner[-1]+dr/2])
# twist = np.deg2rad(10) * np.ones_like(r_outer)
# chord = 0.025 * np.ones_like(r_outer)
# t_c = 0.0809 * np.ones_like(r_outer) # NACA0012

# Nr = np.shape(r_outer)[0]-1

# # Cylinder Green module
# caxis = np.array([1.0, 0.0, 0.0])

# corigin = np.array([0.0, -W, -L])
# from TailoredGreen.HalfCylinderGreen import HalfCylinderGreen, SF_FullCylinderGreen
# cg =  HalfCylinderGreen(radius=D/2, axis=caxis, origin=corigin, dim=3, 
#                 numerics= {

#                     # # D90_LR: # excessively low-resolution
#                     # 'nmax': 8,
#                     # 'Nq_prop': 16,
#                     # 'Nq_evan': 8,
#                     # 'eps_radius' : 1e-24, # must be lower than eps_eval!
#                     # 'Nazim' : 5, # discretization of the boundary in the azimuth
#                     # 'Nax': 16, # in the axial direction
#                     # 'RMAX': 20, # max radius!
#                     # 'mode': 'uniform', # uniform or geometric, defines the spacing of the surface panels!
#                     # 'geom_factor': 1.025, # geometric stretching factor, only used if mode is 'geometric'
#                     # 'eps_eval' : 1e-8 # evaluation distance from the actual surface, as a fraction of cylinder radius!
#                     # # Note: the function is currently NOT checking if the panels are compact!

                    
#                     # D180_MR: # just right :)
#                     'nmax': 16,
#                     'Nq_prop': 32,
#                     'Nq_evan': 16,
#                     'eps_radius' : 1e-24, # must be lower than eps_eval!
#                     'Nazim' : 9, # discretization of the boundary in the azimuth
#                     'Nax': 32, # in the axial direction
#                     'RMAX': 20, # max radius!
#                     'mode': 'uniform', # uniform or geometric, defines the spacing of the surface panels!
#                     'geom_factor': 1.025, # geometric stretching factor, only used if mode is 'geometric'
#                     'eps_eval' : 1e-8 # evaluation distance from the actual surface, as a fraction of cylinder radius!
#                     # Note: the function is currently NOT checking if the panels are compact!


#                     # D360_HR: # too expensive!
#                     # 'nmax': 32,
#                     # 'Nq_prop': 64,
#                     # 'Nq_evan': 32,
#                     # 'eps_radius' : 1e-24, # must be lower than eps_eval!
#                     # 'Nazim' : 18, # discretization of the boundary in the azimuth
#                     # 'Nax': 64, # in the axial direction
#                     # 'RMAX': 20, # max radius!
#                     # 'mode': 'uniform', # uniform or geometric, defines the spacing of the surface panels!
#                     # 'geom_factor': 1.025, # geometric stretching factor, only used if mode is 'geometric'
#                     # 'eps_eval' : 1e-8 # evaluation distance from the actual surface, as a fraction of cylinder radius!
#                     # # Note: the function is currently NOT checking if the panels are compact!
#                     })

# # Source Mode assembly
# from SourceMode.SourceMode import SourceModeArray
# sourceArray = SourceModeArray(
#                         BLH=np.zeros((3, Nk, Nr)), 
#                         B = NBLADES,
#                         Omega=Omega, gamma =twist,
#                         axis=axis_prop, origin=origin_prop,
#                         radius=r_outer,
#                         green = cg,
#                         numerics={'Ndipoles' : Ndipoles},
#                         c = c0,
#                         dt = t_c * chord, # thickness distribution used for thickness noise
#                         chord = chord,
#                         )

sourceArray.plotSelf()
plt.show()
print(f'using suffix {SUFFIX}, existing files will be overwritten.')
print(f'proceed? (y/n)')
proceed = input().lower() == 'y'
if not proceed:
    print('aborting')
    exit()

# ------------------- Green's function ---------------------------

start_total = time.time()

# save green on the surface (run once per m)
for index, sm in enumerate(sourceArray.children):
    start_loop = time.time()

    print(f'pre-computing surface Greens functions: {index+1} of {len(sourceArray.children)}')
    source_positions = sm.dipole_positions

    start_compute = time.time()
    G_surface = sm.green.getGreenAtSurface( source_positions, ms*
                                            sourceArray.B * 
                                            sourceArray.Omega / 
                                            sourceArray.SoS
                                            )  # shape (Nm, Nz, Ny)
    end_compute = time.time()

    np.save(f'./Data/current/NACA0012_rotor/G_surface_sm_{index}_{MODE}{SUFFIX}.npy', G_surface)

    end_loop = time.time()

    print(f'  -> compute time: {end_compute - start_compute:.3f} s')
    print(f'  -> total loop time: {end_loop - start_loop:.3f} s')

end_total = time.time()
print(f'Total time (Green): {end_total - start_total:.3f} s')


# ------------------- Green's function gradient ---------------------------

start_total = time.time()

# save gradients on the surface (run once per m)
for index, sm in enumerate(sourceArray.children):
    start_loop = time.time()

    print(f'pre-computing surface gradients of G_t: {index+1} of {len(sourceArray.children)}')

    source_positions = sm.dipole_positions

    start_compute = time.time()
    gradG_surface = sm.green.getGreenGradAtSurface(source_positions, ms*
sourceArray.B * 
sourceArray.Omega / 
sourceArray.SoS)  # shape (3, Nm, Nz, Ny)
    end_compute = time.time()

    np.save(f'./Data/current/NACA0012_rotor/gradG_surface_sm_{index}_{MODE}{SUFFIX}.npy', gradG_surface)

    end_loop = time.time()

    print(f'  -> compute time: {end_compute - start_compute:.3f} s')
    print(f'  -> total loop time: {end_loop - start_loop:.3f} s')

end_total = time.time()
print(f'Total time (Gradient): {end_total - start_total:.3f} s')