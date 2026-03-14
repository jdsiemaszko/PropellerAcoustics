from TailoredGreen.HalfCylinderGreen import HalfCylinderGreen, SF_FullCylinderGreen
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from SourceMode.SourceMode import SourceMode, SourceModeArray
from PotentialInteraction.beam_to_blade import BladeLoadings
from PotentialInteraction.blade_to_beam import BeamLoadings
from Constants.helpers import read_airfoil_table
from Constants.helpers import p_to_SPL


# setup = True # whether to run the case again or use the stored data
setup = True

mode = 'half'
# mode = 'full'


Nax = 64
Nazim = 36
RMAX = 10

if mode == 'half':
    NBEAMS = 1
    clss = HalfCylinderGreen
elif mode == 'full':
    NBEAMS = 2
    clss = SF_FullCylinderGreen
    Nax *= 2
else:
    raise ValueError('mode not recognized')


NBLADES = 2

R0 = 0.02
R1 = 0.0889
Dcylinder_m = 0.032
Lcylinder_m = 0.014 + Dcylinder_m/2

RHO = 1.225 #kgm^-3
SOS = 340 # ms^-1
OMEGA=6390/60*2*np.pi

    # _________ NUMERICAL INPUTS __________
NSEG = 20
KMAX = 16
NDIPOLES = 36 # discterization of each source mode

radius_array = np.linspace(R0, R1, NSEG+1)


radius_c = (radius_array[1:] + radius_array[:-1]) / 2
if setup:
    # ASSUMPTIONS
    

    # load propeller data
    df = read_airfoil_table("Data/Roger2023/7x5E-PERF.PE0")

    twist_array = np.deg2rad(np.interp(radius_array, df['STATION'] * 0.0254, df["TWIST"]))# in rad, mind imperial units in input files
    chord_array = 0.0254 * np.interp(radius_array, df['STATION'] * 0.0254, df["CHORD"])



    data_roger = np.loadtxt('Data/Roger2023/RogerUzv2.csv', skiprows=1, delimiter=',').T
    Uz0_mps = np.interp(radius_c, data_roger[0], data_roger[1])


    B = NBLADES
    rho_kgm3 = RHO
    c_mps = SOS
    kmax = KMAX
    nb = NBEAMS

    # full cylinder
    axis_cg = np.array([1.0, 0.0, 0.0])
    origin_cg = np.array([0.0, 0.0, -Lcylinder_m])
    radial_cg = np.array([0.0, 0.0, 1.0]) # for consistency with potentialinteraction convention

    cg = clss(
        radius=Dcylinder_m/2, axis=axis_cg, origin=origin_cg, radial=radial_cg, dim=3, 
                            numerics={
                        'mmax': 16,
                        'Nq_prop': 128,
                        'Nq_evan': 128,
                        'eps_radius' : 1e-24, # must be lower than eps_eval!
                        'Nazim' : Nazim, # discretization of the boundary in the azimuth
                        'Nax': Nax, # in the axial direction
                        'RMAX': RMAX, # max radius!
                        'mode': 'uniform', # uniform or geometric, defines the spacing of the surface panels!
                        'geom_factor': 1.02, # geometric stretching factor, only used if mode is 'geometric'
                        'eps_eval' : 1e-12 # evaluation distance from the actual surface, as a fraction of cylinder radius!
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
    sma.plotSelf(fig, ax, plot_normals=None)
    ax.set_aspect('equal')
    ax.set_axis_off()
    plt.show()
    plt.close()

    fig, ax, data, z, th = sma.plotSurfacePressureFullCylinder(m=1, 
                                        extend_z=(0.03, 0.12)
                                        )
    plt.show()

    np.save(f'Data/Roger2023/surface_pressure_{mode}_v2.npy', data)
    np.save(f'Data/Roger2023/surface_z_{mode}_v2.npy', z)
    np.save(f'Data/Roger2023/surface_theta_{mode}_v2.npy', th)


else:
    p = np.load(f'Data/Roger2023/surface_pressure_{mode}_v2.npy')
    z = np.load(f'Data/Roger2023/surface_z_{mode}_v2.npy')
    th = np.load(f'Data/Roger2023/surface_theta_{mode}_v2.npy')


    SPL_mb = p_to_SPL(p)

    # --- reshape consistent with construction ---
    Nax, Nazim = len(z), len(th)
    sol_2d = SPL_mb.reshape(Nax, Nazim)

    # # --- shift theta from [0, 2π) → [-π, π) ---
    # th_shifted = (th + np.pi) % (2*np.pi) - np.pi

    # # --- sort theta so it increases from -π to π ---
    # sort_idx = np.argsort(th_shifted)

    # th = th_shifted[sort_idx]
    # sol_2d = sol_2d[:, sort_idx]   # reorder along azimuth axis

    # --- meshgrid ---
    Z, TH = np.meshgrid(z, th, indexing='ij')

    if mode=='half':
        vmin, vmax = 75, 105
    elif mode=='full':
        vmin, vmax = 90, 120
    levels = np.linspace(vmin, vmax, 21)
    fig, ax = plt.subplots()
    cf = ax.contourf(TH.T * Dcylinder_m/2, Z.T, sol_2d.T, levels=levels, cmap='jet', extend='both'
    )

    cbar = fig.colorbar(cf, ax=ax, extend='both')
    cbar.set_label("Directivity [dB]")

    ylabel = "$z$ [m]"
    xlabel = r"$\theta \cdot a$ [m]"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.0, 0.1)
    ax.set_ylim(0.03, 0.12)

    plt.show()