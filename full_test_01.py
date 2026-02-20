from PotentialInteraction.main import HansonModel
from SourceMode.SourceMode import SourceModeArray
from TailoredGreen.CylinderGreen import CylinderGreen
from TailoredGreen.TailoredGreen import TailoredGreen
from Constants.const import p_to_SPL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# _________ INPUTS __________
ROBS = 1.62 # m
D_CYL = 0.02 # m
L_CYL = 0.02 # m
SOS=340.0 # m/s
RHO = 1.2 # kgm^-3
CHORD = 0.025 # m
TWIST = 10 # deg
NBLADES = 2
NBEAMS = 1 # radial - 1, diameterical - 2
R0_PROP = 0.016 # m
R1_PROP = 0.1 # m
OMEGA = 8000/60 * 2 * np.pi # rad/s

# _________ NUMERICAL INPUTS __________
NSEG = 8 # number of radial prop segments
KMAX = 32 # max loading harmonic number
NTHETA = 18 # discretization in the polar angle
NPHI = 36 # discretization in the azimuth
MMAX = 32 # maximum resolved mode number in cylinder green's function
NQ_PROP = 64 # discretization in the z wavenumber in cylinder green's function
EPS_RADIUS = 1e-3 # cutoff for cylinder distance. points are ignored if distance to cylinder < a * (1 + EPS_RADIUS)
NDIPOLES = 32 # discterization of each source mode

# _________ SETUP ____________

twist_array = np.deg2rad(TWIST) * np.ones(NSEG+1)
chord_array = CHORD * np.ones(NSEG+1)
radius_array = np.linspace(R0_PROP, R1_PROP, NSEG+1, endpoint=True)

# _________ POSITIONING _____________

# LOADING PREDICTION MODULE
# VELLA ET AL. 2026, using data from GOJON ET AL. 2023
# Dataset ISAE_2_D20_L20
HANSON_VELLA = HansonModel(twist_rad = twist_array, chord_m = chord_array,
                    radius_m=radius_array, B=NBLADES, nb=NBEAMS,
                    Dcylinder_m=D_CYL, Lcylinder_m=L_CYL, Omega_rads=OMEGA, rho_kgm3=RHO, c_ms=SOS, kmax=KMAX)

loadings = HANSON_VELLA.getBladeLoadingMagnitude() # shape (Nk, Nr)


# GREEN'S FUNCTION MODULE
axis_cyl = np.array([1.0, 0.0, 0.0]) # x-direction cylinder...
origin_cyl = np.array([0.0, 0.0, -L_CYL]) # ... at z=-L_CYL below the propeller
gcyl = CylinderGreen(radius=D_CYL/2, axis=axis_cyl, origin=origin_cyl, dim=3, 
                        numerics={
                    'nmax': MMAX,
                    'Nq_prop': NQ_PROP,
                    'eps_radius' : EPS_RADIUS
                })

gf = TailoredGreen(dim=3) # free-field version!

# SOURCE MODE MODULE
axis_prop = np.array([0.0, 0.0, 1.0]) # z-direction propeller...
origin_prop = np.array([0.0, 0.0, 0.0]) # ... at z=0

sourceArray = SourceModeArray(BLH=loadings, # loading per unit span
                        B = NBLADES,
                        Omega=OMEGA, gamma = twist_array,
                        axis=axis_prop, origin=origin_prop,
                        radius=radius_array,
                        green = gcyl, # pass the green's function module!
                        # green = gf,
                        numerics={'Ndipoles' : NDIPOLES}
                        )

# _____________ PLOTTING & RESULTS _______________

# 1) plot geometry

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")
# source.plotSelf(fig, ax)
sourceArray.plotSelf(fig, ax)
# ax.set_box_aspect([1, 1, 1])
ax.set_aspect('equal')
ax.set_axis_off()
plt.show()
plt.close()

# 2) plot directivity

harmonic_to_plot = 5

for m in [harmonic_to_plot]:
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    HANSON_VELLA.plotDirectivity(fig, ax, m=m, R=ROBS,
                            valmax=65, valmin=10,
                            Nphi=NPHI, Ntheta=NTHETA,
                            # mode='beam',
                            #   mode='total',
                            mode='blade'
                            )
    plt.show()
    plt.close(fig)

sourceArray.plotFarFieldPressure(m=np.array([harmonic_to_plot]), Nphi=NPHI, Ntheta=NTHETA, R=ROBS, valmax=65, valmin=10)