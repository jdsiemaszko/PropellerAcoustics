from TailoredGreen.HalfCylinderGreen import HalfCylinderGreen, SF_FullCylinderGreen
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from SourceMode.SourceMode import SourceMode, SourceModeArray
from PotentialInteraction.beam_to_blade import BladeLoadings
from PotentialInteraction.blade_to_beam import BeamLoadings
from PotentialInteraction.placeholder import *



# ASSUMPTIONS
NBLADES = 2
TWIST = 10 # deg
CHORD = 0.025 #m
NBEAMS = 1 # does nothing
RHO = 1.2#kgm^-3
SOS = 340 # ms^-1
OMEGA=8000/60*2*np.pi

# _________ NUMERICAL INPUTS __________
NSEG = 20
KMAX = 64
NTHETA = 18 # discretization in the polar angle
NPHI = 36 # discretization in the azimuth
NDIPOLES = 36 # discterization of each source mode
# DATA ASSIMILATION (VELLA ET AL. 2026)

# _________ SETUP ____________
twist_array = np.deg2rad(TWIST) * np.ones(NSEG+1)
chord_array = CHORD * np.ones(NSEG+1)
radius_array = np.linspace(0.016, 0.1, NSEG+1)
radius_c = (radius_array[1:] + radius_array[:-1]) / 2
Uz0_mps = -np.interp(radius_c / 0.1, R_RT_UDW, UDW_EXACT) # interpolated
Tprime_Npm = np.interp(radius_c / 0.1, R_RT_EXACT, DT_EXACT)
Qprime_Npm = np.interp(radius_c / 0.1, R_RT_EXACT, DQ_EXACT)
B = 2
Dcylinder_m = 0.02
Lcylinder_m = 0.02
rho_kgm3 = 1.2
c_mps = 340
kmax = 20
nb = 1

# full cylinder
axis_cg = np.array([1.0, 0.0, 0.0])
origin_cg = np.array([0.0, 0.0, -Lcylinder_m])
cg = SF_FullCylinderGreen(radius=Dcylinder_m/2, axis=axis_cg, origin=origin_cg, dim=3, 
                        numerics={
                    'nmax': 32,
                    'Nq_prop': 128,
                    'eps_radius' : 1e-12, # must be lower than eps_eval!
                    'Nazim' : 18, # discretization of the boundary in the azimuth
                    'Nax': 128, # in the axial direction
                    'RMAX': 20, # max radius!
                    'mode': 'geometric', # uniform or geometric, defines the spacing of the surface panels!
                    'geom_factor': 1.05, # geometric stretching factor, only used if mode is 'geometric'
                    'eps_eval' : 1e-3 # evaluation distance from the actual surface, as a fraction of cylinder radius!
                    # Note: the function is currently NOT checking if the panels are compact!
                 })

# --- get blade loading on cylinder
blade_l = BladeLoadings(
    twist_rad=twist_array,
    chord_m=chord_array,
    radius_m=radius_array,
    Uz0_mps=Uz0_mps,
    Tprime_Npm=Tprime_Npm,
    Qprime_Npm=Qprime_Npm,
    B=B,
    Dcylinder_m=Dcylinder_m,
    Lcylinder_m=Lcylinder_m,
    Omega_rads=OMEGA,
    rho_kgm3=rho_kgm3,
    c_mps=c_mps,
    kmax=kmax,
    nb=nb
)

beam_l = BeamLoadings(
    twist_rad=twist_array,
    chord_m=chord_array,
    radius_m=radius_array,
    Uz0_mps=Uz0_mps,
    Tprime_Npm=Tprime_Npm,
    Qprime_Npm=Qprime_Npm,
    B=B,
    Dcylinder_m=Dcylinder_m,
    Lcylinder_m=Lcylinder_m,
    Omega_rads=OMEGA,
    rho_kgm3=rho_kgm3,
    c_mps=c_mps,
    kmax=kmax,
    nb=nb
)

axis_prop = np.array([0.0, 0.0, 1.0]) # z-direction propeller...
origin_prop = np.array([0.0, 0.0, 0.0]) # ... at z=0
sma = SourceModeArray(BLH=blade_l.getBladeLoadingMagnitude(), # loading per unit span, magnitude only
                        B = NBLADES,
                        Omega=OMEGA, gamma = twist_array,
                        axis=axis_prop, origin=origin_prop,
                        radius=radius_array,
                        green = cg,
                        numerics={'Ndipoles' : NDIPOLES},
                        c = SOS
                        )


# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(111, projection="3d")
# sma.plotSelf(fig, ax)
# ax.set_aspect('equal')
# # ax.set_axis_off()
# plt.show()
# plt.close()

beam_l.plotSurfacePressureContour(m=1)
plt.show()
plt.close()

sma.plotSurfacePressureFullCylinder(m=1, 
                                    extend_z=(0.016, 0.1)
                                    )
plt.show()

