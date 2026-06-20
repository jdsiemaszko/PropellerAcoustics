"""
test the PIN functionality, create a .h5 file with the inputs and outputs
"""

import h5py
import numpy as np
from Constants.helpers import read_force_file
from Constants.data_assim import getGojonData
from SourceMode.Configurations_NACA0012 import D20L20W00_D180 as sourceArray
from Hanson.far_field import HansonModel

r_inner, Fz, Fphi  = read_force_file('./Data/Zamponi2026/FS_ISAE_2_8000.txt') # reuse the radial stations from data
BLH, BLH_S, BLH_US, _ = sourceArray.getLoading(Fz, Fphi, D=0.02, L=0.02, steady_only=False) # compute loading on the fly, return PIN for reuse
PIN = sourceArray.getPIN(Fz, Fphi)
Hanson = sourceArray.getHanson()
PIN._numerics['gamma_steady'] = True

# blade_harmonics = PIN.getBladeLoadingHarmonics() # 3, Nk, Nr of np.complex128
beam_harmonics = PIN.getStrutLoadingHarmonics()  # 3, Nk, Nr of np.complex128

ms = np.arange(1, 11, 1) # pick the first 10 harmonics (arbitrary)

k = PIN.k # Nk
radius_inner = PIN.seg_radius # Nr
radius_outer = PIN.radius # Nr + 1
twist_inner = PIN.seg_twist # Nr
chord_inner = PIN.seg_chord # Nr
twist_outer = PIN.twist # Nr + 1
chord_outer = PIN.chord # Nr + 1

Omega = PIN.Omega
rho = PIN.rho
c0 = PIN.SoS

L = PIN.Lcylinder
D = PIN.Dcylinder
B = PIN.B

# parse the positions from data file!
datadir = './Experimental/dataverse_files'
data, BPF, freq, x_cart, theta, phi, theta_exp, phi_exp, casefile = getGojonData(datadir, D, L, shape='D', B=B, RPM=int(Omega * 60/2/np.pi))
t_c_effective = PIN.seg_t_c
t_c_outer = np.append(t_c_effective, t_c_effective[-1])

# 2D mesh
theta = np.sort(theta)
phi = np.sort(phi)
theta_m, phi_m = np.meshgrid(theta, phi, indexing='ij')
# shapes: (Ntheta, Nphi)

# flatten
R = np.max(np.linalg.norm(x_cart, axis=0))
R_arr     = np.full(theta_m.size, R)
theta_arr = theta_m.ravel()
phi_arr   = phi_m.ravel()

X = R_arr * np.sin(np.deg2rad(theta_arr)) * np.cos(np.deg2rad(phi_arr))
Y = R_arr * np.sin(np.deg2rad(theta_arr)) * np.sin(np.deg2rad(phi_arr))
Z = R_arr * np.cos(np.deg2rad(theta_arr))

x_cart = np.array([X, Y, Z])


#  all pressure complex of shape Ntheta*Nphi, Nm
pLSmB_model_rotor, _ = Hanson.getPressureRotor(x_cart, ms, BLH_S)

pLUSmB_model_rotor, _ = Hanson.getPressureRotor(x_cart, ms, BLH_US)

pmB_model_strut_loading, _ = Hanson.getPressureStator(x_cart, ms*B, beam_harmonics)

ptmB_model_rotor, _ = Hanson.getThicknessNoiseRotor(x_cart, ms, chord_inner, t_c_effective)

frequency = ms * B * Omega / 2 / np.pi

"""
structure:
    -test_propagator.h5
    --inputs
    ---Omega_rad_p_s
    ---rho_kg_p_m3
    ---sos_m_p_s
    ---radius_inner_m
    ---radius_outer_m
    ---twist_inner_rad
    ---twist_outer_rad
    ---chord_inner_m
    ---chord_outer_m
    ---t_c_inner
    ---t_c_outer
    ---L_cylinder_m
    ---D_cylinder_m
    ---B
    ---blade_loading_harmonics_N_p_m
    ---beam_loading_harmonics_N_p_m
    --outputs
    ---p_loading_blade_steady_Pa
    ---p_loading_blade_unsteady_Pa
    ---p_thickness_blade_Pa
    ---p_loading_strut_Pa
    ---observer_position_m
    ---harmonics_m
    ---frequency_Hz
    ---theta_rad -> shape Nx!
    ---phi_rad -> shape Nx!
"""


filename = "./tests/test_propagator.h5"

with h5py.File(filename, "w") as h5:

    # ==============================================================
    # INPUTS
    # ==============================================================
    g_in = h5.create_group("inputs")

    g_in.create_dataset("Omega_rad_p_s", data=Omega)
    g_in.create_dataset("rho_kg_p_m3", data=rho)
    g_in.create_dataset("sos_m_p_s", data=c0)

    g_in.create_dataset("radius_inner_m", data=radius_inner)
    g_in.create_dataset("radius_outer_m", data=radius_outer)

    g_in.create_dataset("twist_inner_rad", data=twist_inner)
    g_in.create_dataset("twist_outer_rad", data=twist_outer)

    g_in.create_dataset("chord_inner_m", data=chord_inner)
    g_in.create_dataset("chord_outer_m", data=chord_outer)

    g_in.create_dataset("t_c_inner", data=t_c_effective)
    g_in.create_dataset("t_c_outer", data=t_c_outer)

    g_in.create_dataset("L_cylinder_m", data=L)
    g_in.create_dataset("D_cylinder_m", data=D)

    g_in.create_dataset("B", data=B)

    # --------------------------------------------------------------
    # complex loading (blade + beam)
    # --------------------------------------------------------------
    def write_complex(group, name, arr):
        g = group.create_group(name)
        g.create_dataset("real", data=np.real(arr))
        g.create_dataset("imag", data=np.imag(arr))

    # blade + beam harmonics
    write_complex(g_in, "blade_loading_harmonics_N_p_m", BLH)
    # write_complex(g_in, "blade_loading_steady_N_p_m", BLH_S)
    # write_complex(g_in, "blade_loading_unsteady_N_p_m", BLH_US)
    write_complex(g_in, "beam_loading_harmonics_N_p_m", beam_harmonics)

    # ==============================================================
    # OUTPUTS
    # ==============================================================
    g_out = h5.create_group("outputs")

    write_complex(g_out, "p_loading_blade_steady_Pa", pLSmB_model_rotor)
    write_complex(g_out, "p_loading_blade_unsteady_Pa", pLUSmB_model_rotor)
    write_complex(g_out, "p_loading_strut_Pa", pmB_model_strut_loading)
    write_complex(g_out, "p_thickness_blade_Pa", ptmB_model_rotor)

    # observer geometry
    g_out.create_dataset("observer_positions_m", data=x_cart)

    g_out.create_dataset("harmonics_m", data=ms)
    g_out.create_dataset("frequency_mB_Hz", data=frequency)

    g_out.create_dataset("observer_radius_m", data=R_arr)
    g_out.create_dataset("observer_polar_rad", data=np.deg2rad(theta_arr))
    g_out.create_dataset("observer_azimuth_rad", data=np.deg2rad(phi_arr))
    # TODO: Robs

print(f"Wrote {filename}")