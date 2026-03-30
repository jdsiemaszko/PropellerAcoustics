import h5py
# from scipy.special import hankel2, jve, jv
import numpy as np
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PotentialInteraction.beam_to_blade import BladeLoadings
from PotentialInteraction.blade_to_beam import BeamLoadings
from TailoredGreen.HalfCylinderGreen import HalfCylinderGreen, SF_FullCylinderGreen
from TailoredGreen.CylinderGreen import CylinderGreen
from TailoredGreen.TailoredGreen import TailoredGreen
from Hanson.far_field import HansonModel
from SourceMode.SourceMode import SourceModeArray
from Constants.helpers import p_to_SPL, plot_BPF_peaks, spl_from_autopower, plot_directivity_contour, plot_3D_directivity, plot_3D_phase_directivity
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat

import numpy as np

# ==========================================================
# Hanson Far-field formula for a fixed distortion case
# ==========================================================

# ----------------------------------------------------------
# Test Conditions
# ----------------------------------------------------------
Mx = 0.0
c0 = 340.0
V = Mx * c0
rho0 = 1.2
NBEAMS = 1
m = np.array([5]) # pick a harmonic with significant beam noise 

if NBEAMS == 1:
    MODE = 'half'
elif NBEAMS ==2:
    MODE = 'full'
else:
    raise ValueError('mode not recognized')

# ----------------------------------------------------------
# Blade Characteristics
# ----------------------------------------------------------
B = 2
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

NR = 5
NDIPOLES = 18*2 # VERY IMPORTANT, will fail at higher harmonics!
Ntheta = 18
Nphi = 36
numerics = {
'nmax': 16,
'Nq_prop': 64,
'Nq_evan': 32,
'eps_radius' : 1e-24, # must be lower than eps_eval!
'Nazim' : 18, # discretization of the boundary in the azimuth
'Nax': 64, # in the axial direction
'RMAX': 10, # max radius!
'mode': 'uniform', # uniform or geometric, defines the spacing of the surface panels!
'geom_factor': 1.025, # geometric stretching factor, only used if mode is 'geometric'
'eps_eval' : 1e-8 # evaluation distance from the actual surface, as a fraction of cylinder radius!
# Note: the function is currently NOT checking if the panels are compact!
}


# Radial discretization
# ----------------------------------------------------------
r0 = np.linspace(rR, rT, NR)

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

# ------------------------------------------------------------------
# Class initializations using shared instances
# ------------------------------------------------------------------
ddr = r0[1] - r0[0]
r_outer = np.concatenate((r0-ddr/2, [r0[-1]+ddr/2]))
ddr = np.diff(r_outer)
Nk = 40

twist = np.deg2rad(pitch)* np.ones(r_outer.shape)
chord = c * np.ones(r_outer.shape)

blade_l = BladeLoadings(
    twist_rad= twist,
    chord_m= chord,
    radius_m=r_outer,
    Uz0_mps=U_flow,
    Tprime_Npm=dT / dr,
    Qprime_Npm=dQ / dr,
    B=B,
    Dcylinder_m=D_bras,
    Lcylinder_m=g,
    Omega_rads=Omega,
    rho_kgm3=rho0,
    c_mps=c0,
    kmax=Nk,
    nb=NBEAMS
    # nb = 1
)

beam_l = BeamLoadings(
    twist_rad=np.deg2rad(pitch)* np.ones(r_outer.shape),
    chord_m=c* np.ones(r_outer.shape),
    radius_m=r_outer,
    Uz0_mps=U_flow,
    Tprime_Npm= dT / dr,
    Qprime_Npm= dQ / dr,
    B=B,
    Dcylinder_m=D_bras,
    Lcylinder_m=g,
    Omega_rads=Omega,
    rho_kgm3=rho0,
    c_mps=c0,
    kmax=Nk,
    nb=NBEAMS
    # nb = 2,
)

# CYLINDER GREEN'S FUNCTION
caxis = np.array([1.0, 0.0, 0.0])
D_prop = 0.2
D = 20 / 1000
L = 20 / 1000
corigin = np.array([0.0, 0.0, -L])


if NBEAMS == 2:
    cg = CylinderGreen(radius=D_bras/2, axis=caxis, origin=corigin, dim=3, 
                            numerics=numerics) # cylinder
    # cg =  SF_FullCylinderGreen(radius=D_bras/2, axis=caxis, origin=corigin, dim=3, 
    #                 numerics=numerics)

elif NBEAMS == 1:
    cg =  HalfCylinderGreen(radius=D_bras/2, axis=caxis, origin=corigin, dim=3, 
                        numerics=numerics)

gf = TailoredGreen(dim=3) # free-field version!
# SOURCE MODE MODULE
axis_prop = np.array([0.0, 0.0, 1.0]) # z-direction propeller...
origin_prop = np.array([0.0, 0.0, 0.0]) # ... at z=0


BLH = blade_l.getBladeLoadingHarmonics()
BLH_US = np.zeros_like(BLH)
BLH_US[:, 1:, :] = BLH[:, 1:, :]
BLH_S = np.zeros_like(BLH)
BLH_S[:, 0, :] = BLH[:, 0, :]
sourceArray = SourceModeArray(BLH=BLH, # loading per unit span
                        B = B,
                        Omega=Omega, gamma =twist,
                        axis=axis_prop, origin=origin_prop,
                        radius=r_outer,
                        green = cg,
                        # green = gf,
                        numerics={'Ndipoles' : NDIPOLES},
                        c = c0
                        )


# HANSON (stator only)

han = HansonModel(
radius_m=r_outer, # blade radius stations [m] of size Nr + 1
axis=axis_prop, origin=origin_prop,
# radial=np.array([1, 0, 0]), # coordinate system (not needed here)
Omega_rads=Omega, # rotation speed [rad/s]
rho_kgm3=rho0, # fluid density [kg/m^3]
c_mps= c0, # speed of sound [m/s]
nb=NBEAMS # number of beams (irrelevant)
)

if __name__ == "__main__":
    # _____________ PLOTTING & RESULTS _______________

    # 1) plot geometry

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    sourceArray.plotSelf(fig, ax)
    # ax.set_box_aspect([1, 1, 1])
    ax.set_aspect('equal')
    ax.set_axis_off()
    plt.show()
    plt.close()

    # angular coordinates
    theta = np.linspace(0.0, np.pi, Ntheta, endpoint=True)
    phi   = np.linspace(0.0, 2.0 * np.pi, Nphi, endpoint=True)

    # 2D mesh
    theta_m, phi_m = np.meshgrid(theta, phi, indexing='ij')
    # shapes: (Ntheta, Nphi)

    # flatten
    R_arr     = np.full(theta_m.size, R)
    theta_arr = theta_m.ravel()
    phi_arr   = phi_m.ravel()

    X = R_arr * np.sin(theta_arr) * np.cos(phi_arr)
    Y = R_arr * np.sin(theta_arr) * np.sin(phi_arr)
    Z = R_arr * np.cos(theta_arr)

    x_cart = np.array([X, Y, Z])


    ############## save results to a file as the computation takes some time
    # p_scattered = sourceArray.getScatteredPressure(x_cart, m)
    # np.save(f'./Data/current/NACA0012_rotor/p_scattered_{MODE}_m{int(m)}.npy', p_scattered)
    # p_direct_blade = sourceArray.getDirectPressure(x_cart, m)
    # np.save(f'./Data/current/NACA0012_rotor/p_direct_{MODE}_m{int(m)}.npy', p_direct_blade)

    p_scattered = np.load(f'./Data/current/NACA0012_rotor/p_scattered_{MODE}_m{int(m)}.npy')
    p_direct_blade = np.load(f'./Data/current/NACA0012_rotor/p_direct_{MODE}_m{int(m)}.npy')


    beam_loading = beam_l.getBeamLoadingHarmonics() 
    p_direct_beam, _ = han.getPressureStator(x_cart, m*B, beam_loading) # mind the indexing change for m
    p_direct_blade_hanson, _ = han.getPressureRotor(x_cart, m, BLH) 

    p_direct_beam *= 1 # -1 to match the phase - why?

    # Plot maps:
    # 1) 2D contours
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) =  plt.subplots(3, 2, figsize=(10, 5), sharex=True, sharey=True)

    VMIN, VMAX = 10, 65
    levels = np.linspace(VMIN, VMAX, 20, endpoint=False)
    fig, ax1 = plot_directivity_contour(
        np.rad2deg(theta_m), np.rad2deg(phi_m), p_direct_blade, title='blade direct noise (source-mode)', fig=fig, ax=ax1, cmap='jet', levels=levels, xlabel=None,
    )
    fig, ax2 = plot_directivity_contour(
        np.rad2deg(theta_m), np.rad2deg(phi_m), p_scattered, title='blade scattered noise', fig=fig, ax=ax2, cmap='jet', ylabel=None, levels=levels, xlabel=None,
    )
    fig, ax3 = plot_directivity_contour(
        np.rad2deg(theta_m), np.rad2deg(phi_m), p_direct_beam, title='beam loading noise', fig=fig, ax=ax3, cmap='jet',  levels=levels,xlabel=None, 
    )

    fig, ax4 = plot_directivity_contour(
        np.rad2deg(theta_m), np.rad2deg(phi_m), p_direct_blade_hanson,
        title='blade direct noise (Hanson)', fig=fig, ax=ax4, cmap='jet',  levels=levels, ylabel=None,  xlabel=None, 
    )

    fig, ax5 = plot_directivity_contour(
        np.rad2deg(theta_m), np.rad2deg(phi_m), p_direct_blade + p_scattered,
        title='direct blade + scattered blade', fig=fig, ax=ax5, cmap='jet',  levels=levels, 
    )
    fig, ax6 = plot_directivity_contour(
        np.rad2deg(theta_m), np.rad2deg(phi_m), p_direct_blade + p_direct_beam,
        title='direct blade + direct beam', fig=fig, ax=ax6, cmap='jet',  levels=levels, ylabel=None,

    )

    plt.show()
    plt.close()

    VMIN = 10
    # 1) 3D blobs
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(321, projection="3d")
    ax2 = fig.add_subplot(322, projection="3d")
    ax3 = fig.add_subplot(323, projection="3d")
    ax4 = fig.add_subplot(324, projection="3d")
    ax5 = fig.add_subplot(325, projection="3d")
    ax6 = fig.add_subplot(326, projection="3d")


    fig, ax1 = plot_3D_directivity(
        p_direct_blade, theta_m, phi_m, title='blade direct noise', fig=fig, ax=ax1, valmin=VMIN, valmax=VMAX,
    )
    fig, ax2 = plot_3D_directivity(
        p_scattered, theta_m, phi_m, title='blade scattered noise', fig=fig, ax=ax2, valmin=VMIN, valmax=VMAX,
    )
    fig, ax3 = plot_3D_directivity(
        p_direct_beam, theta_m, phi_m, title='beam loading noise', fig=fig, ax=ax3, valmin=VMIN, valmax=VMAX,
    )
    # fig, ax4 = plot_3D_directivity(
    #     p_direct_beam + p_direct_blade + p_scattered, theta_m, phi_m,
    #       title='total loading noise', fig=fig, ax=ax4, valmin=VMIN, valmax=VMAX,
    # )
    fig, ax4 = plot_3D_directivity(
        p_direct_blade_hanson, theta_m, phi_m,
        title='blade direct noise (Hanson)', fig=fig, ax=ax4, valmin=VMIN, valmax=VMAX,
    )

    fig, ax5 = plot_3D_directivity(
        p_direct_blade + p_scattered, theta_m, phi_m,
        title='direct blade + scattered blade', fig=fig, ax=ax5, valmin=VMIN, valmax=VMAX,
    )

    fig, ax6 = plot_3D_directivity(
        p_direct_blade + p_direct_beam, theta_m, phi_m,
        title='direct blade + direct beam', fig=fig, ax=ax6, valmin=VMIN, valmax=VMAX,
    )
    plt.show()


    # 3D phase
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(321, projection="3d")
    ax2 = fig.add_subplot(322, projection="3d")
    ax3 = fig.add_subplot(323, projection="3d")
    ax4 = fig.add_subplot(324, projection="3d")
    ax5 = fig.add_subplot(325, projection="3d")
    ax6 = fig.add_subplot(326, projection="3d")


    fig, ax1 = plot_3D_phase_directivity(
        p_direct_blade, theta_m, phi_m, title='blade direct noise', fig=fig, ax=ax1, valmin=VMIN, valmax=VMAX,
    )
    fig, ax2 = plot_3D_phase_directivity(
        p_scattered, theta_m, phi_m, title='blade scattered noise', fig=fig, ax=ax2, valmin=VMIN, valmax=VMAX,
    )
    fig, ax3 = plot_3D_phase_directivity(
        p_direct_beam, theta_m, phi_m, title='beam loading noise', fig=fig, ax=ax3, valmin=VMIN, valmax=VMAX,
    )
    # fig, ax4 = plot_3D_directivity(
    #     p_direct_beam + p_direct_blade + p_scattered, theta_m, phi_m,
    #       title='total loading noise', fig=fig, ax=ax4, valmin=VMIN, valmax=VMAX,
    # )
    fig, ax4 = plot_3D_phase_directivity(
        p_direct_blade_hanson, theta_m, phi_m,
        title='blade direct noise (Hanson)', fig=fig, ax=ax4, valmin=VMIN, valmax=VMAX,
    )

    fig, ax5 = plot_3D_phase_directivity(
        p_direct_blade + p_scattered, theta_m, phi_m,
        title='direct blade + scattered blade', fig=fig, ax=ax5, valmin=VMIN, valmax=VMAX,
    )

    fig, ax6 = plot_3D_phase_directivity(
        p_direct_blade + p_direct_beam, theta_m, phi_m,
        title='direct blade + direct beam', fig=fig, ax=ax6, valmin=VMIN, valmax=VMAX,
    )
    plt.show()


