from SourceMode.SourceMode import SourceModeArray
from SourceMode.Configurations_NACA0012 import PARROT_D20L20W00_D36_1_10 as sma
from TailoredGreen.TailoredGreen import TailoredGreen
from TailoredGreen.HalfCylinderGreen import HalfCylinderGreen
import numpy as np
import matplotlib.pyplot as plt
from Constants.helpers import read_force_file
from Constants.data_assim import read_selig_airfoil, compute_camber_thickness

# r_inner, _, _ = read_force_file('./Data/Zamponi2026/FS_ISAE_2_8000.txt') # reuse the radial stations from data
# r_inner = np.linspace(r_inner[0], r_inner[-1], 10)
# dr = np.diff(r_inner)[0]
# r_outer = np.hstack([r_inner-dr/2, r_inner[-1]+dr/2])
# rc, c = np.loadtxt('./Data/Parrot2024/chord.csv', skiprows=1, delimiter=',').T # radius, chord in meters
# rp, p = np.loadtxt('./Data/Parrot2024/pitch.csv', skiprows=1, delimiter=',').T # radius, pitch in degrees
# name, x, y = read_selig_airfoil('./Data/current/airfoils/NACA0012.dat')
# xc, camber, thickness = compute_camber_thickness(x, y)
# t_c_uniform = np.interp(np.linspace(0., 1., 1000), xc, thickness)

# green = TailoredGreen()
# numerics_cyl_midres = {
#                     # D36_MR:
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
#                     }
# caxis = np.array([1.0, 0.0, 0.0])

# cg_midres_D20L20W00 =  HalfCylinderGreen(radius=20/1000/2, axis=caxis, origin=np.array([0.0, -0/1000, -20/1000]), dim=3, numerics= numerics_cyl_midres)

# chord_parrot = np.interp(r_outer, rc, c)
# pitch_parrot = np.interp(r_outer, rp, p)
# # ref thrust of 2.15N @ RPM of 7250
# axis_prop = np.array([0.0, 0.0, 1.0]) # z-direction propeller...
# origin_prop = np.array([0.0, 0.0, 0.0]) # ... at z=0
# sma = SourceModeArray(
#                         BLH=np.zeros((3, 40, r_inner.shape[0])), 
#                         B = 2,
#                         Omega = 7250 / 60 * 2 * np.pi, # parrot rotor RPM!
#                         gamma = np.deg2rad(pitch_parrot),
#                         axis=axis_prop, origin=origin_prop,
#                         radius=r_outer,
#                         green = cg_midres_D20L20W00,
#                         numerics={'Nsources' : 36, 'Nlayers': 5},
#                         c0 = 340,
#                         dt = t_c_uniform[None, :] * chord_parrot[:, None], # Nr, Nc
#                         chord = chord_parrot,
#                         airfoil = 'naca0012'
#                         )

sma.plotSelf()

plt.show()
