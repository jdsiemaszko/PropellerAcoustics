import numpy as np
import pandas as pd

datfile = 'Data/Zamponi2026/VKI_BEMT_radial_48_sections.csv'

df = pd.read_csv(datfile)

casename = 'eVTOLUTION'

rho0 = 1.2
c0 = 340.0


NR = 10
NDIPOLES = 18*4 # VERY IMPORTANT, will fail at higher harmonics!
Ntheta = 18
Nphi = 36
numerics = {
'nmax': 16,
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


r0 = df['radius_m'].to_numpy()

ddt = r0[1] - r0[0]
r_outer = np.concatenate((r0-ddt/2, [r0[-1]+ddt/2]))
dr = np.diff(r_outer)

dT_dr = df['dT_dr'].to_numpy()
dQ_dr = df['dQ_dr'].to_numpy()
dT = dT_dr * dr
dQ = dQ_dr * dr

T_cum = np.concatenate(([0], np.cumsum(dT)))
Q_cum = np.concatenate(([0], np.cumsum(dQ)))

# --- define coarse grid ---
Nr_coarse = NR  # set this
r0 = np.linspace(r0.min(), r0.max(), Nr_coarse)

# --- rebuild edges on coarse grid ---
ddt = r0[1] - r0[0]
r_outer_prime = np.concatenate((r0 - ddt/2, [r0[-1] + ddt/2]))
dr = np.diff(r_outer_prime)

# --- interpolate cumulative integrals onto coarse edges ---
T_cum_interp = np.interp(r_outer_prime, r_outer, T_cum)
Q_cum_interp = np.interp(r_outer_prime, r_outer, Q_cum)

# --- recover coarse cell-integrated values ---
dT = np.diff(T_cum_interp)
dQ = np.diff(Q_cum_interp)

# --- recover derivatives (same variable names as original) ---
dT_dr = dT / dr
dQ_dr = dQ / dr
r_outer = r_outer_prime


RPM = 8900
Omega = RPM / 60 * 2 * np.pi


# upwash from momentum theory, ignoring spin

U_flow = np.sqrt(dT / dr / r0 / rho0 / 4 / np.pi)

twist = np.deg2rad(10) * np.ones_like(r_outer)
chord = 0.025 * np.ones_like(r_outer)

D_bras = 0.02
g = 0.02
B = 2
NBEAMS = 1
R = 1.62
m = np.array([5])


VMIN = 10
VMAX = 85