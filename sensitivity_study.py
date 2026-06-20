from TailoredGreen.HalfCylinderGreen import HalfCylinderGreen
from Constants.helpers import read_force_file
from Constants.data_assim import read_selig_airfoil, compute_camber_thickness
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D
import numpy as np
from SourceMode.SourceMode import SourceModeArray
import os

folder_name = f'./Data/current/validation/sens_v2/'
os.makedirs(folder_name, exist_ok=True)

# BEGINNING OF HEADER
FILE='TOTAL'
MODE = 'half'
from Constants.helpers import read_force_file, plot_3D_directivity, plot_3D_phase_directivity, p_to_SPL, spl_from_autopower, plot_BPF_peaks
from Constants.data_assim import getGojonData


# ------------------- Common Inputs -----------------------------
axis_prop = np.array([0.0, 0.0, 1.0]) # z-direction propeller...
origin_prop = np.array([0.0, 0.0, 0.0]) # ... at z=0
Omega_ref = 8000 / 60 * 2 * np.pi
NBLADES = 2
c0 = 340.
rho0 = 1.2
Nk = 40 # number of resolved loading harmonics, max frequency is Nk * Omega
m_surface = np.arange(1, 11, 1) # example, anything more than 10 is likely overkill and would require A LOT of dipoles to resolve!
r_inner, Fz, Fphi  = read_force_file('./Data/Zamponi2026/FS_ISAE_2_8000.txt') # reuse the radial stations from data
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

cg_midres_D20L20W00 =  HalfCylinderGreen(radius=20/1000/2, axis=caxis, origin=np.array([0.0, -0/1000, -20/1000]), dim=3, numerics= numerics_cyl_midres)

sourceArray = SourceModeArray(
                        BLH=np.zeros((3, Nk, Nr)), 
                        B = NBLADES,
                        Omega = Omega_ref, gamma = twist,
                        axis=axis_prop, origin=origin_prop,
                        radius=r_outer,
                        green = cg_midres_D20L20W00,
                        numerics={'Nsources' : 180},
                        c0 = c0,
                        dt = t_c_uniform[None, :] * chord[:, None], # Nr, Nc
                        chord = chord,
                        airfoil = 'naca0012'
                        )

DATANAME = 'LOADING_HALF'
outfile = folder_name + DATANAME + '.npz'

sourceArray.numerics['CompactnessCorrection'] = True
SUFFIX = '_D180_MR'
shape='D'
NDIPOLES = sourceArray.Nsources
D_bras = sourceArray.green.radius * 2
g = -1 * sourceArray.green.origin[2]
BLH, BLH_S, BLH_US, _ = sourceArray.getLoading(Fz, Fphi, D_bras, g, steady_only=False) # compute loading on the fly, return PIN for reuse
PIN = sourceArray.PIN

# g= 0.02
B = sourceArray.B
c = sourceArray.chord
Omega = sourceArray.Omega

c0 = sourceArray.SoS
han = sourceArray.getHanson()
# END OF HEADER

    # ind_theta = 6     # 60 to -60 in 10
    # ind_phi = 9          # 0 to 350 in 10

for (ind_theta, ind_phi) in zip([2], [9]):
    print(f'parsing case {SUFFIX}, ind_theta: {ind_theta}, ind_phi: {ind_phi}')
    datadir = './Experimental/dataverse_files'
    # casefile = f'ISAE_2_D{int(1000*D_bras)}_L{int(1000*g)}'

    data, BPF, freq, x_cart_data, theta_data, phi_data, theta_exp, phi_exp, casefile = getGojonData(datadir, D_bras, g, shape=shape, B=2, 
                                                                                                    RPM=int(Omega * 60/2/np.pi)
                                                                                                    )
    data = data[:, ind_theta, ind_phi]
    x_cart = x_cart_data[:, ind_theta, ind_phi].reshape((3, 1))
    theta = theta_data[ind_theta]
    phi = phi_data[ind_phi]

    print(theta, phi)

    Nr = len(r_inner)
    ms = np.arange(1, 11, 1) # harmonics to extract



    # pSmB_model_rotor = han.getPressureRotor(x_cart, ms, 
    #                                     #    blade_l.getBladeLoadingHarmonics()
    #                                     BLH_S
    #                                        )[0][0]

    # pUSmB_model_rotor = han.getPressureRotor(x_cart, ms, 
    #                                     #    blade_l.getBladeLoadingHarmonics()
    #                                     BLH_US
    #                                        )[0][0]
    # pLmB_model_rotor = pSmB_model_rotor + pUSmB_model_rotor
    # SPL_rotor_S = p_to_SPL(pSmB_model_rotor)
    # SPL_rotor_US = p_to_SPL(pUSmB_model_rotor)

    # BLH_S[2, :, :] = 0
    # BLH_US[2, :, :] = 0

    pLUSmB_model_rotor = han.getPressureRotor(x_cart, ms, 
                                        #    blade_l.getBladeLoadingHarmonics()
                                        BLH_US
                                        )[0][0]

    pLSmB_model_rotor = han.getPressureRotor(x_cart, ms, 
                                        #    blade_l.getBladeLoadingHarmonics()
                                        BLH_S
                                        )[0][0]



    ptmB_model_rotor = han.getThicknessNoiseRotor(x_cart, ms, sourceArray.seg_chord, 0.0822 * np.ones_like(r_inner))[0][0] # NACA0012
    # BL  =  beam_l.getBeamLoadingHarmonics(BLH=BLH)


    PIN._numerics['only_linear'] = True 
    PIN._numerics['only_nonlinear'] = False


    PIN._numerics['include_vortex_sources'] = True
    PIN._numerics['include_thickness_sources'] = False
    BL = PIN.getStrutLoadingHarmonics()
    pmB_model_beam_loading = han.getPressureStator(x_cart, ms*B, BL)[0][0]

    PIN._numerics['include_vortex_sources'] = False
    PIN._numerics['include_thickness_sources'] = True
    BL = PIN.getStrutLoadingHarmonics()
    pmB_model_beam_thickness = han.getPressureStator(x_cart, ms*B, BL)[0][0]

    PIN._numerics['only_linear'] = False
    PIN._numerics['only_nonlinear'] = False
    PIN._numerics['include_vortex_sources'] = True
    PIN._numerics['include_thickness_sources'] = True
    BL = PIN.getStrutLoadingHarmonics()
    pmB_model_beam_total = han.getPressureStator(x_cart, ms*B, BL)[0][0]


    pmB_model_rotor_total = pLSmB_model_rotor + pLUSmB_model_rotor + ptmB_model_rotor
    pmB_model_rotor_loading = pLSmB_model_rotor + pLUSmB_model_rotor
    pmB_model_total = pLSmB_model_rotor + pLUSmB_model_rotor + ptmB_model_rotor + pmB_model_beam_total # assuming coherent
    # pmB_model_total = np.sqrt(np.abs(pmB_model_rotor_total)**2 + np.abs(pmB_model_beam)**2) # assuming incoherent



    Nchildren = len(sourceArray.children)
    # SUFFIX = '

    # -------------------------------- SCATTERED LOADING NOISE ------------------------------------------
    # save gradients in the far-field (run once per observer and m)
    for index, sm in enumerate(sourceArray.children):

        gradG_surface = np.load(f'./Data/current/NACA0012_rotor/gradG_surface_sm_{index}_{MODE}{SUFFIX}.npy') # shape (3, Nm, Nz, Ny)
        # print(f'pre-computing far-field gradients {index+1}')

        gradG = sm.getScatteringGreenGradient(x_cart, ms*B * np.abs(Omega)  / c0, gradG_surface) # shape (3, Nm, Nx, Ny)
        np.save(f'./Data/current/NACA0012_rotor/gradG_sm_{index}_{MODE}_{ind_theta}_{ind_phi}_{FILE}{SUFFIX}.npy', gradG)



    # extract and rearrange

    gradG_arr = np.zeros((Nchildren, 3, ms.shape[0], x_cart.shape[1], NDIPOLES), dtype=np.complex128)
    for index, sm in enumerate(sourceArray.children):
        gradG_arr[index] = np.load(f'./Data/current/NACA0012_rotor/gradG_sm_{index}_{MODE}_{ind_theta}_{ind_phi}_{FILE}{SUFFIX}.npy')



    # p_direct_s = sourceArray.getDirectPressure(x_cart, ms, BLH=np.transpose(BLH_S, axes=[2, 0, 1]))[0]
    # p_direct_us = sourceArray.getDirectPressure(x_cart, ms, BLH=np.transpose(BLH_US, axes=[2, 0, 1]))[0]

    sourceArray.updateBLH(BLH_US)
    p_scattered_us = sourceArray.getScatteredPressure(x_cart, ms, gradG=gradG_arr)[0]
    p_direct_us = sourceArray.getDirectPressure(x_cart, ms)[0]

    sourceArray.updateBLH(BLH_S)
    p_scattered_s = sourceArray.getScatteredPressure(x_cart, ms, gradG=gradG_arr)[0]
    p_direct_s = sourceArray.getDirectPressure(x_cart, ms)[0]




    # np.save(f'./Data/current/NACA0012_rotor/p_s_spectrum_{MODE}_{ind_theta}_{ind_phi}.npy', p_scattered)
    # p_scattered = np.load(f'./Data/current/NACA0012_rotor/p_s_spectrum_{MODE}_{ind_theta}_{ind_phi}.npy')


    # -------------------------------- SCATTERED Thickness NOISE ------------------------------------------

    # save gradients in the far-field (run once per observer and m)
    for index, sm in enumerate(sourceArray.children):

        G_surface = np.load(f'./Data/current/NACA0012_rotor/G_surface_sm_{index}_{MODE}{SUFFIX}.npy') # shape (Nm, Nz, Ny)
        # print(f'pre-computing far-field gradients {index+1}')

        G = sm.getScatteringGreen(x_cart, ms*B * np.abs(Omega)  / c0, G_surface) # shape (Nm, Nx, Ny)
        np.save(f'./Data/current/NACA0012_rotor/G_sm_{index}_{MODE}_{ind_theta}_{ind_phi}_{FILE}{SUFFIX}.npy', G)


    G_arr = np.zeros((Nchildren, ms.shape[0], x_cart.shape[1], NDIPOLES), dtype=np.complex128)
    for index, sm in enumerate(sourceArray.children):
        G_arr[index] = np.load(f'./Data/current/NACA0012_rotor/G_sm_{index}_{MODE}_{ind_theta}_{ind_phi}_{FILE}{SUFFIX}.npy')


    p_scattered_thickness = sourceArray.getThicknessPressureScattered(x_cart, ms, G=G_arr)[0]
    p_direct_thickness = sourceArray.getThicknessPressureDirect(x_cart, ms)[0]

    # np.save(f'./Data/current/NACA0012_rotor/p_s_spectrum_thickness_{MODE}_{ind_theta}_{ind_phi}.npy', p_scattered_thickness)
    # p_scattered_thickness = np.load(f'./Data/current/NACA0012_rotor/p_s_spectrum_thickness_{MODE}_{ind_theta}_{ind_phi}.npy')

    for element in [p_direct_s, p_direct_us, p_scattered_s, p_scattered_us, p_direct_thickness, p_scattered_thickness]:
        element[np.isnan(element)] = 0.0

    p_total_scattering = p_direct_s + p_direct_us + p_scattered_s + p_scattered_us + p_direct_thickness + p_scattered_thickness
    p_rotor_total = p_direct_s + p_direct_us + p_direct_thickness
    # p_total_minus_scattered_thickness = p_total_scattering - p_scattered_thickness

    # SPLS PIN MODEL
    SPL_rotor_loading = p_to_SPL(pmB_model_rotor_loading)
    SPL_rotor_S = p_to_SPL(pLSmB_model_rotor)
    SPL_rotor_US = p_to_SPL(pLUSmB_model_rotor)

    SPL_rotor_total = p_to_SPL(pmB_model_rotor_total)
    SPL_rotor_thickness = p_to_SPL(ptmB_model_rotor)
    SPL_beam_loading = p_to_SPL(pmB_model_beam_loading)
    SPL_beam_thickness = p_to_SPL(pmB_model_beam_thickness)

    SPL_total_PIN = p_to_SPL(pmB_model_total)
    SPL_PIN_beam_total = p_to_SPL(pmB_model_beam_total)

    # SPLS TOTAL SCATTERING
    SPL_direct_s = p_to_SPL(p_direct_s)
    SPL_direct_us = p_to_SPL(p_direct_us)
    SPL_scattered_s = p_to_SPL(p_scattered_s)
    SPL_scattered_us = p_to_SPL(p_scattered_us)
    SPL_scattered = p_to_SPL(p_scattered_s + p_scattered_us)
    SPL_direct_thickness = p_to_SPL(p_direct_thickness)
    SPL_scattered_thickness = p_to_SPL(p_scattered_thickness)
    SPL_scattered_total = p_to_SPL(p_scattered_s + p_scattered_us + p_scattered_thickness)
    SPL_SM_rotor_total = p_to_SPL(p_rotor_total)
    SPL_total_scattering = p_to_SPL(p_total_scattering)
    # SPL_total_scattering_minus_scattered_thickness = p_to_SPL(p_total_minus_scattered_thickness)

    # SPL_total = p_to_SPL(p_rms_total) # same computation

    np.savez(
        outfile,
        SPL = SPL_total_scattering,
        m = ms,
    )
    print(f'saved SPL results to {outfile}')