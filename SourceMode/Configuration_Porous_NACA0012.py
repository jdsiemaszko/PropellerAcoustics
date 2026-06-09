from TailoredGreen.HalfCylinderGreen import HalfCylinderGreen, ImpedanceCylinderGreen
from Constants.helpers import read_force_file
from Constants.data_assim import read_selig_airfoil, compute_camber_thickness
import numpy as np
from SourceMode.SourceMode import SourceModeArray


# ------------------- Common Inputs -----------------------------
axis_prop = np.array([0.0, 0.0, 1.0]) # z-direction propeller...
origin_prop = np.array([0.0, 0.0, 0.0]) # ... at z=0
Omega_ref = 8000 / 60 * 2 * np.pi
NBLADES = 2
c0 = 340.
rho0 = 1.2
Nk = 40 # number of resolved loading harmonics, max frequency is Nk * Omega
m_surface = np.arange(1, 11, 1) # example, anything more than 10 is likely overkill and would require A LOT of dipoles to resolve!
r_inner, _, _ = read_force_file('./Data/Zamponi2026/FS_ISAE_2_8000.txt') # reuse the radial stations from data
dr = np.diff(r_inner)[0]
r_outer = np.hstack([r_inner-dr/2, r_inner[-1]+dr/2])
twist = np.deg2rad(10) * np.ones_like(r_outer)
chord = 0.025 * np.ones_like(r_outer)
# t_c = 0.0809 * np.ones_like(r_outer) # NACA0012

name, x, y = read_selig_airfoil('./Data/current/airfoils/NACA0012.dat')
xc, camber, thickness = compute_camber_thickness(x, y)
t_c_uniform = np.interp(np.linspace(0., 1., 1000), xc, thickness)

Nr = np.shape(r_outer)[0]-1
caxis = np.array([1.0, 0.0, 0.0])
D_prop = 0.2

numerics_cyl_midres = {
                    # D180_MR: # just right :)
                    'nmax': 16,
                    'Nq_prop': 32,
                    'Nq_evan': 16,
                    'eps_radius' : 1e-24, # must be lower than eps_eval!
                    'Nazim' : 9, # discretization of the boundary in the azimuth
                    'Nax': 32, # in the axial direction
                    'RMAX': 20, # max radius!
                    'mode': 'uniform', # uniform or geometric, defines the spacing of the surface panels!
                    'geom_factor': 1.025, # geometric stretching factor, only used if mode is 'geometric'
                    'eps_eval' : 1e-8 # evaluation distance from the actual surface, as a fraction of cylinder radius!
                    # Note: the function is currently NOT checking if the panels are compact!
                    }

# ------------------- Varying Inputs -----------------------------

Nsources = 180    

# ???
fref = np.array([200, 800, 1400, 2000])
betaref = np.array([
    0.5631 + 1j * 0.2578,
    0.8227 + 1j * 0.1721,
    0.8837 + 1j * 0.1290,
    0.9117 + 1j * 0.1049
])
kref = fref * 2 * np.pi / c0
imp_func = lambda k: np.interp(k, kref, betaref) * 1j * k # set such that (d/dn + imp) p = 0 on the surface. Mind n is oriented inwards

# Cylinder Green module
cg_midres_porous =  ImpedanceCylinderGreen(radius=20/1000/2, axis=caxis, origin=np.array([0.0, -0/1000, -20/1000]), dim=3, numerics= numerics_cyl_midres,
                                      impedance = imp_func)

# Source Mode assembly
D20L20_porous = SourceModeArray(
                        BLH=np.zeros((3, Nk, Nr)), 
                        B = NBLADES,
                        Omega=Omega_ref, gamma =twist,
                        axis=axis_prop, origin=origin_prop,
                        radius=r_outer,
                        green = cg_midres_porous,
                        numerics={'Nsources' : 180},
                        c0 = c0,
                        dt = t_c_uniform[None, :] * chord[:, None], # Nr, Nc
                        chord = chord,
                        airfoil = 'naca0012'
                        )



