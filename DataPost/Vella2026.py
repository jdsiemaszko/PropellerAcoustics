from Constants.helpers import p_to_SPL, plot_BPF_peaks, spl_from_autopower, plot_directivity_contour, plot_3D_directivity, plot_3D_phase_directivity
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat
import numpy as np


casename = 'Vella2026'

# consts
Mx = 0.0
c0 = 340.0
V = Mx * c0
rho0 = 1.2

# ----------------------------------------------------------
# Blade Characteristics
# ----------------------------------------------------------
B = 2
NBEAMS = 1

D = 0.2
rT = D / 2
rR = 0.017
FA = 0
MCA = 0
pitch = 10
alpha = (90 - pitch) * np.pi / 180

TREF = 1.075
TORQUEREF = 26.7/B/1000 # Nm


# ----------------------------------------------------------

#----------- NUMERICS ----------------------

# NR = 10
# NDIPOLES = 18
# Ntheta = 18
# Nphi = 36
# numerics = {
#             'mmax': 16, # mind that increasing this increases the chance of overflows, ***should*** be handled by the safe Bessel functions, but beware
#             'Nq_prop': 32,
#             'Nq_evan': 32,
#             'eps_k' : 1e-24,
#                     }

NR = 5*2*2
NDIPOLES = 18*2*2*2 # VERY IMPORTANT, will fail at higher harmonics!
Ntheta = 18
Nphi = 36
numerics = {
'nmax': 16*2,
'Nq_prop': 64*2,
'Nq_evan': 32*2,
'eps_radius' : 1e-24, # must be lower than eps_eval!
'Nazim' : 18*2, # discretization of the boundary in the azimuth
'Nax': 64*2, # in the axial direction
'RMAX': 20, # max radius!
'mode': 'uniform', # uniform or geometric, defines the spacing of the surface panels!
'geom_factor': 1.025, # geometric stretching factor, only used if mode is 'geometric'
'eps_eval' : 1e-8 # evaluation distance from the actual surface, as a fraction of cylinder radius!
# Note: the function is currently NOT checking if the panels are compact!
}


# Radial discretization
# ----------------------------------------------------------
r0 = np.linspace(rR, rT, NR)
ddt = r0[1] - r0[0]
r_outer = np.concatenate((r0-ddt/2, [r0[-1]+ddt/2]))

RPM = 8000
f0 = RPM / 60
c = 0.025
Omega = RPM * 2 * np.pi / 60
V0 = r0 * Omega

dr = np.abs(rT - rR) / len(r0)
A = c * (rT - rR)
dA = c * dr

# ----------------------------------------------------------
# Aerodynamic load (.mat file)
# ----------------------------------------------------------
mat_data = loadmat('Data/Vella2026/Fz_Fy_aero_pale_sans_bras_tours1a5.mat')

Fz_tour_bis = mat_data['Fz_tour_bis'].flatten()
Fy_tour_bis = mat_data['Fy_tour_bis'].flatten()
x_load = mat_data['x'].flatten()

r_rT = r0 / rT

# --- 1D interpolation (MATLAB interp1 equivalent)
dT = np.interp(r_rT, x_load[:-1], Fz_tour_bis)
dQ = np.interp(r_rT, x_load[:-1], Fy_tour_bis)



# replicate MATLAB edge correction
# dT[-2:] = Fz_tour_bis[-1]
# dQ[-2:] = Fy_tour_bis[-1]

dT *= TREF / np.sum(dT)
alpha = TORQUEREF / np.sum(dQ * r0)
dQ *= alpha

dT_dr = dT / dr
dQ_dr = dQ / dr

# ----------------------------------------------------------
# Aerodynamic coefficients
# ----------------------------------------------------------
Cd = dQ / (0.5 * rho0 * V0**2 * dA)
Cl = dT / (0.5 * rho0 * V0**2 * dA)

# ----------------------------------------------------------
# Complementary Data
# ----------------------------------------------------------
MT = rT * Omega / c0
U = np.sqrt(V**2 + (Omega * r0)**2)
Mr = U / c0

# ----------------------------------------------------------
# Observer
# ----------------------------------------------------------
R = 1.62

OmegaD = Omega

t = np.arange(0, 16, 1/(51.2e3))

# ----------------------------------------------------------
# Induced velocity field
# ----------------------------------------------------------
mat_vel = loadmat('Data/Vella2026/profils_vitesse_induite_axiale_8000RPM_B2_D20.mat')

xq = mat_vel['xq'].flatten()
zz = mat_vel['zz'].flatten()
w_mean = mat_vel['w_mean']

D_bras = 0.02
g = 0.02
z = np.array([-g + D_bras/2, -g, -g - D_bras/2])
z_rT = z / rT

# --- 2D interpolation (MATLAB interp2 equivalent)
interp_func = RegularGridInterpolator(
    (xq, zz),
    w_mean,
    bounds_error=False,
    fill_value=None
)

# Build interpolation points
points = np.array([[rr, zz_i] for zz_i in z for rr in r_rT])

U_flow_interp = interp_func(points)
U_flow_interp = U_flow_interp.reshape(len(z), len(r_rT))

U_flow = -U_flow_interp[0, :]

twist = np.deg2rad(10) * np.ones_like(r_outer)
chord = 0.025 * np.ones_like(r_outer)

VMIN = 10
VMAX = 65