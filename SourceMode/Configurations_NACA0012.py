from TailoredGreen.HalfCylinderGreen import HalfCylinderGreen
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

numerics_cyl_lowres = {
                    # D90_LR: # excessively low-resolution
                    'nmax': 8,
                    'Nq_prop': 16,
                    'Nq_evan': 8,
                    'eps_radius' : 1e-24, # must be lower than eps_eval!
                    'Nazim' : 5, # discretization of the boundary in the azimuth
                    'Nax': 16, # in the axial direction
                    'RMAX': 20, # max radius!
                    'mode': 'uniform', # uniform or geometric, defines the spacing of the surface panels!
                    'geom_factor': 1.025, # geometric stretching factor, only used if mode is 'geometric'
                    'eps_eval' : 1e-8 # evaluation distance from the actual surface, as a fraction of cylinder radius!
                    # Note: the function is currently NOT checking if the panels are compact!
                    }

numerics_cyl_highres = {
                    # D360_HR: # too expensive!
                    'nmax': 32,
                    'Nq_prop': 64,
                    'Nq_evan': 32,
                    'eps_radius' : 1e-24, # must be lower than eps_eval!
                    'Nazim' : 18, # discretization of the boundary in the azimuth
                    'Nax': 64, # in the axial direction
                    'RMAX': 20, # max radius!
                    'mode': 'uniform', # uniform or geometric, defines the spacing of the surface panels!
                    'geom_factor': 1.025, # geometric stretching factor, only used if mode is 'geometric'
                    'eps_eval' : 1e-8 # evaluation distance from the actual surface, as a fraction of cylinder radius!
                    # Note: the function is currently NOT checking if the panels are compact!
                    }

# ------------------- Varying Inputs -----------------------------

Nsources = 180    

# Cylinder Green module
cg_midres_D20L20W00 =  HalfCylinderGreen(radius=20/1000/2, axis=caxis, origin=np.array([0.0, -0/1000, -20/1000]), dim=3, numerics= numerics_cyl_midres)
cg_highres_D20L20W00 =  HalfCylinderGreen(radius=20/1000/2, axis=caxis, origin=np.array([0.0, -0/1000, -20/1000]), dim=3, numerics= numerics_cyl_highres)
cg_lowres_D20L20W00 =  HalfCylinderGreen(radius=20/1000/2, axis=caxis, origin=np.array([0.0, -0/1000, -20/1000]), dim=3, numerics= numerics_cyl_lowres)

# sideways shift
cg_midres_D20L20W10 =  HalfCylinderGreen(radius=20/1000/2, axis=caxis, origin=np.array([0.0, -10/1000, -20/1000]), dim=3, numerics= numerics_cyl_midres)
cg_midres_D20L20W20 =  HalfCylinderGreen(radius=20/1000/2, axis=caxis, origin=np.array([0.0, -20/1000, -20/1000]), dim=3, numerics= numerics_cyl_midres)
cg_midres_D20L20W40 =  HalfCylinderGreen(radius=20/1000/2, axis=caxis, origin=np.array([0.0, -40/1000, -20/1000]), dim=3, numerics= numerics_cyl_midres)
cg_midres_D20L20W60 =  HalfCylinderGreen(radius=20/1000/2, axis=caxis, origin=np.array([0.0, -60/1000, -20/1000]), dim=3, numerics= numerics_cyl_midres)
cg_midres_D20L20W80 =  HalfCylinderGreen(radius=20/1000/2, axis=caxis, origin=np.array([0.0, -80/1000, -20/1000]), dim=3, numerics= numerics_cyl_midres)
cg_midres_D20L20W100 =  HalfCylinderGreen(radius=20/1000/2, axis=caxis, origin=np.array([0.0, -100/1000, -20/1000]), dim=3, numerics= numerics_cyl_midres)

# smaller cylinders
cg_midres_D10L20W00 =  HalfCylinderGreen(radius=10/1000/2, axis=caxis, origin=np.array([0.0, -0/1000, -20/1000]), dim=3, numerics= numerics_cyl_midres)
cg_midres_D15L20W00 =  HalfCylinderGreen(radius=15/1000/2, axis=caxis, origin=np.array([0.0, -0/1000, -20/1000]), dim=3, numerics= numerics_cyl_midres)

# taking c/4 as reference:
cg_midres_D20L21W00 =  HalfCylinderGreen(radius=20/1000/2, axis=caxis, origin=np.array([0.0, -0/1000, -21.3523/1000]), dim=3, numerics= numerics_cyl_midres)


# Source Mode assembly
D20L20W00_D180 = SourceModeArray(
                        BLH=np.zeros((3, Nk, Nr)), 
                        B = NBLADES,
                        Omega=Omega_ref, gamma =twist,
                        axis=axis_prop, origin=origin_prop,
                        radius=r_outer,
                        green = cg_midres_D20L20W00,
                        numerics={'Nsources' : 180},
                        c0 = c0,
                        dt = t_c_uniform[None, :] * chord[:, None], # Nr, Nc
                        chord = chord,
                        airfoil = 'naca0012'
                        )

D20L20W00_D360 = SourceModeArray(
                        BLH=np.zeros((3, Nk, Nr)), 
                        B = NBLADES,
                        Omega=Omega_ref, gamma =twist,
                        axis=axis_prop, origin=origin_prop,
                        radius=r_outer,
                        green = cg_highres_D20L20W00,
                        numerics={'Nsources' : 360},
                        c0 = c0,
                        dt = t_c_uniform[None, :] * chord[:, None], # Nr, Nc
                        chord = chord,
                        airfoil = 'naca0012'
                        )

D20L20W00_D90 = SourceModeArray(
                        BLH=np.zeros((3, Nk, Nr)), 
                        B = NBLADES,
                        Omega=Omega_ref, gamma =twist,
                        axis=axis_prop, origin=origin_prop,
                        radius=r_outer,
                        green = cg_lowres_D20L20W00,
                        numerics={'Nsources' : 90},
                        c0 = c0,
                        dt = t_c_uniform[None, :] * chord[:, None], # Nr, Nc
                        chord = chord,
                        airfoil = 'naca0012'
                        )

D20L20W10_D180 = SourceModeArray(
                        BLH=np.zeros((3, Nk, Nr)), 
                        B = NBLADES,
                        Omega=Omega_ref, gamma =twist,
                        axis=axis_prop, origin=origin_prop,
                        radius=r_outer,
                        green = cg_midres_D20L20W10,
                        numerics={'Nsources' : 180},
                        c0 = c0,
                        dt = t_c_uniform[None, :] * chord[:, None], # Nr, Nc
                        chord = chord,
                        airfoil = 'naca0012'
                        )

D20L20W20_D180 = SourceModeArray(
                        BLH=np.zeros((3, Nk, Nr)), 
                        B = NBLADES,
                        Omega=Omega_ref, gamma =twist,
                        axis=axis_prop, origin=origin_prop,
                        radius=r_outer,
                        green = cg_midres_D20L20W20,
                        numerics={'Nsources' : 180},
                        c0 = c0,
                        dt = t_c_uniform[None, :] * chord[:, None], # Nr, Nc
                        chord = chord,
                        airfoil = 'naca0012'
                        )

D20L20W40_D180 = SourceModeArray(
                        BLH=np.zeros((3, Nk, Nr)), 
                        B = NBLADES,
                        Omega=Omega_ref, gamma =twist,
                        axis=axis_prop, origin=origin_prop,
                        radius=r_outer,
                        green = cg_midres_D20L20W40,
                        numerics={'Nsources' : 180},
                        c0 = c0,
                        dt = t_c_uniform[None, :] * chord[:, None], # Nr, Nc
                        chord = chord,
                        airfoil = 'naca0012'
                        )

D20L20W60_D180 = SourceModeArray(
                        BLH=np.zeros((3, Nk, Nr)), 
                        B = NBLADES,
                        Omega=Omega_ref, gamma =twist,
                        axis=axis_prop, origin=origin_prop,
                        radius=r_outer,
                        green = cg_midres_D20L20W60,
                        numerics={'Nsources' : 180},
                        c0 = c0,
                        dt = t_c_uniform[None, :] * chord[:, None], # Nr, Nc
                        chord = chord,
                        airfoil = 'naca0012'
                        )

D20L20W80_D180 = SourceModeArray(
                        BLH=np.zeros((3, Nk, Nr)), 
                        B = NBLADES,
                        Omega=Omega_ref, gamma =twist,
                        axis=axis_prop, origin=origin_prop,
                        radius=r_outer,
                        green = cg_midres_D20L20W80,
                        numerics={'Nsources' : 180},
                        c0 = c0,
                        dt = t_c_uniform[None, :] * chord[:, None], # Nr, Nc
                        chord = chord,
                        airfoil = 'naca0012'
                        )

D20L20W100_D180 = SourceModeArray(
                        BLH=np.zeros((3, Nk, Nr)), 
                        B = NBLADES,
                        Omega=Omega_ref, gamma =twist,
                        axis=axis_prop, origin=origin_prop,
                        radius=r_outer,
                        green = cg_midres_D20L20W100,
                        numerics={'Nsources' : 180},
                        c0 = c0,
                        dt = t_c_uniform[None, :] * chord[:, None], # Nr, Nc
                        chord = chord,
                        airfoil = 'naca0012'
                        )

D10L20W00_D180 = SourceModeArray(
                        BLH=np.zeros((3, Nk, Nr)), 
                        B = NBLADES,
                        Omega=Omega_ref, gamma =twist,
                        axis=axis_prop, origin=origin_prop,
                        radius=r_outer,
                        green = cg_midres_D10L20W00,
                        numerics={'Nsources' : 180},
                        c0 = c0,
                        dt = t_c_uniform[None, :] * chord[:, None], # Nr, Nc
                        chord = chord,
                        airfoil = 'naca0012'
                        )

D15L20W00_D180 = SourceModeArray(
                        BLH=np.zeros((3, Nk, Nr)), 
                        B = NBLADES,
                        Omega=Omega_ref, gamma =twist,
                        axis=axis_prop, origin=origin_prop,
                        radius=r_outer,
                        green = cg_midres_D15L20W00,
                        numerics={'Nsources' : 180},
                        c0 = c0,
                        dt = t_c_uniform[None, :] * chord[:, None], # Nr, Nc
                        chord = chord,
                        airfoil = 'naca0012'
                        )

# Parrot rotor, see "Analysis of MAV Rotors Optimized for Low Noise and Aerodynamic Efficiency with Operational Constraints" by Volsi et al. (2024)
# TODO: change pitch and chord distributions!

rc, c = np.loadtxt('./Data/Parrot2024/chord.csv', skiprows=1, delimiter=',').T # radius, chord in meters
rp, p = np.loadtxt('./Data/Parrot2024/pitch.csv', skiprows=1, delimiter=',').T # radius, pitch in degrees

chord_parrot = np.interp(r_outer, rc, c)
pitch_parrot = np.interp(r_outer, rp, p)
# ref thrust of 2.15N @ RPM of 7250
PARROT_D20L20W00_D180 = SourceModeArray(
                        BLH=np.zeros((3, Nk, Nr)), 
                        B = NBLADES,
                        Omega = 7250 / 60 * 2 * np.pi, # parrot rotor RPM!
                        gamma = np.deg2rad(pitch_parrot),
                        axis=axis_prop, origin=origin_prop,
                        radius=r_outer,
                        green = cg_midres_D20L20W00,
                        numerics={'Nsources' : 180},
                        c0 = c0,
                        dt = t_c_uniform[None, :] * chord_parrot[:, None], # Nr, Nc
                        chord = chord_parrot,
                        airfoil = 'naca0012'
                        )

PARROT_D20L21W00_D180 = SourceModeArray(
                        BLH=np.zeros((3, Nk, Nr)), 
                        B = NBLADES,
                        Omega = 7250 / 60 * 2 * np.pi, # parrot rotor RPM!
                        gamma = np.deg2rad(pitch_parrot),
                        # gamma = twist,
                        axis=axis_prop, origin=origin_prop,
                        radius=r_outer,
                        green = cg_midres_D20L21W00,
                        numerics={'Nsources' : 180},
                        c0 = c0,
                        dt = t_c_uniform[None, :] * chord_parrot[:, None], # Nr, Nc
                        chord = chord_parrot,
                        airfoil = 'naca0012'
                        )

PARROT_D20L20W00_D360 = SourceModeArray(
                        BLH=np.zeros((3, Nk, Nr)), 
                        B = NBLADES,
                        Omega = 7250 / 60 * 2 * np.pi, # parrot rotor RPM!
                        gamma = np.deg2rad(pitch_parrot),
                        axis=axis_prop, origin=origin_prop,
                        radius=r_outer,
                        green = cg_highres_D20L20W00,
                        numerics={'Nsources' : 360},
                        c0 = c0,
                        dt = t_c_uniform[None, :] * chord_parrot[:, None], # Nr, Nc
                        chord = chord_parrot,
                        airfoil = 'naca0012'
                        )


# TEST: LAYERED VERSION OF PARROT
Nrlow = 10
r_outer = np.linspace(r_outer[0], r_outer[-1], 10)
Nklow = 20
PARROT_D20L20W00_D36_1_10 = SourceModeArray(
                        BLH=np.zeros((3, Nklow, Nrlow)), 
                        B = NBLADES,
                        Omega = 7250 / 60 * 2 * np.pi, # parrot rotor RPM!
                        gamma = np.deg2rad(pitch_parrot),
                        axis=axis_prop, origin=origin_prop,
                        radius=r_outer,
                        green = cg_highres_D20L20W00,
                        numerics={'Nsources' : 36, 'Nlayers': 1},
                        c0 = c0,
                        dt = t_c_uniform[None, :] * chord_parrot[:, None], # Nr, Nc
                        chord = chord_parrot,
                        airfoil = 'naca0012'
                        )


PARROT_D20L20W00_D36_5_10 = SourceModeArray(
                        BLH=np.zeros((3, Nklow, Nrlow)), 
                        B = NBLADES,
                        Omega = 7250 / 60 * 2 * np.pi, # parrot rotor RPM!
                        gamma = np.deg2rad(pitch_parrot),
                        axis=axis_prop, origin=origin_prop,
                        radius=r_outer,
                        green = cg_highres_D20L20W00,
                        numerics={'Nsources' : 36, 'Nlayers': 5},
                        c0 = c0,
                        dt = t_c_uniform[None, :] * chord_parrot[:, None], # Nr, Nc
                        chord = chord_parrot,
                        airfoil = 'naca0012'
                        )
