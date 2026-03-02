from TailoredGreen.HalfCylinderGreen import HalfCylinderGreen, SF_FullCylinderGreen
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from SourceMode.SourceMode import SourceMode, SourceModeArray
from PotentialInteraction.beam_to_blade import BladeLoadings
from PotentialInteraction.blade_to_beam import BeamLoadings
from Constants.helpers import read_airfoil_table




# ASSUMPTIONS
NBLADES = 2

R0 = 0.02
R1 = 0.0889
Dcylinder_m = 0.04
Lcylinder_m = 0.014 + Dcylinder_m/2

NBEAMS = 1 # does nothing
RHO = 1.225 #kgm^-3
SOS = 340 # ms^-1
OMEGA=6420/60*2*np.pi

# _________ SETUP ____________
NSEG = 10
radius_array = np.linspace(R0, R1, NSEG+1)
radius_c = (radius_array[1:] + radius_array[:-1]) / 2

# load propeller data
df = read_airfoil_table("Data/Roger2023/7x5E-PERF.PE0")

twist_array = np.deg2rad(np.interp(radius_array, df['STATION'] * 0.0254, df["TWIST"]))# in rad, mind imperial units in input files
chord_array = 0.0254 * np.interp(radius_array, df['STATION'] * 0.0254, df["CHORD"])


# _________ NUMERICAL INPUTS __________
KMAX = 32
# NTHETA = 3 # discretization in the polar angle
# NPHI = 3 # discretization in the azimuth
NDIPOLES = 36 # discterization of each source mode
# DATA ASSIMILATION (VELLA ET AL. 2026)



data_roger = np.loadtxt('Data/Roger2023/RogerUz.csv', skiprows=1, delimiter=',').T
Uz0_mps = np.interp(radius_c, data_roger[0], data_roger[1])


B = NBLADES
rho_kgm3 = RHO
c_mps = SOS
kmax = KMAX
nb = NBEAMS

# full cylinder
axis_cg = np.array([1.0, 0.0, 0.0])
origin_cg = np.array([0.0, 0.0, -Lcylinder_m])
radial_cg = np.array([0.0, 1.0, 0.0]) # for consistency with potentialinteraction convention

cg = HalfCylinderGreen(
#cg = SF_FullCylinderGreen(
    radius=Dcylinder_m/2, axis=axis_cg, origin=origin_cg, radial=radial_cg, dim=3, 
                        numerics={
                    'nmax': 32,
                    'Nq_prop': 128,
                    'eps_radius' : 1e-12, # must be lower than eps_eval!
                    'Nazim' : 36, # discretization of the boundary in the azimuth
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
    Tprime_Npm=0.0,
    Qprime_Npm=0.0,
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


fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")
sma.plotSelf(fig, ax)
ax.set_aspect('equal')
# ax.set_axis_off()
plt.show()
plt.close()

sma.plotSurfacePressureFullCylinder(m=1, 
                                    extend_z=(0.03, 0.12)
                                    )
plt.show()

