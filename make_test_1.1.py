"""
test the PIN functionality, create a .h5 file with the inputs and outputs
"""

import h5py
import numpy as np
from Constants.helpers import read_force_file
import matplotlib.pyplot as plt
from PotentialInteraction.PIN import PotentialInteraction

r_inner, Fz, Fphi  = read_force_file('./Data/Zamponi2026/FS_ISAE_2_8000.txt') # reuse the radial stations from data
dr = np.diff(r_inner)[0]
r_outer = np.hstack([r_inner-dr/2, r_inner[-1]+dr/2])
# Read flow data

# r_vella = np.load('./data/Vella2026/r.npy')
# Uinf_vella = np.load('./data/Vella2026/Uinf.npy')
# Uinf = np.interp(r_inner, r_vella, Uinf_vella) # interpolate onto our grid

PIN = PotentialInteraction(
        twist_rad = np.deg2rad(10) * np.ones_like(r_outer),
        chord_m = 0.025 * np.ones_like(r_outer),
        radius_m=r_outer,
        t_c = 0.0822 * np.ones_like(r_outer),
        Fzprime_Npm=Fz,
        Fphiprime_Npm=Fphi,
        B=2,
        Dcylinder_m= 0.02, Lcylinder_m=0.02,
        Omega_rads=8000/60 * 2 * np.pi,
        rho_kgm3=1.2,
        c_mps=340,
        kmax=40,
        nb=1,
        numerics= {'Nphi': 180, 'Nthetab': 36}
        )

PIN._numerics['gamma_steady'] = True

Uinf_0 = PIN.Ui # 2, Nr
Uinf = np.zeros_like(Uinf_0)
Uinf[1, :] = Uinf_0[1, :] # remove the x component!
# PIN.Ui = Uinf # only axial inflow
PIN.updateUi(Uinf)

blade_harmonics = PIN.getBladeLoadingHarmonics() # 3, Nk, Nr of np.complex128
strut_harmonics = PIN.getStrutLoadingHarmonics()  # 3, Nk, Nr of np.complex128

blade_downwash = PIN.getBladeDownwash() # Nr, Nphi
phi = PIN.phi # Nphi
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

frequency = k * Omega / 2 / np.pi # Nk

L = PIN.Lcylinder
D = PIN.Dcylinder
B = PIN.B

# create an h5 file with all structure added :)
"""
structure:
    -test_PIN_blade_loadings.h5
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
    ---L_cylinder_m
    ---D_cylinder_m
    ---B
    ---F_z_prime_N_p_m
    ---F_phi_prime_N_p_m
    ---F_r_prime_N_p_m
    --outputs
    ---blade_loading_harmonics_N_p_m
    ---strut_loading_harmonics_N_p_m
    ---blade_downwash_m_p_s
    ---azimuth_rad
    ---harmonic
    ---frequency_Hz
"""

# ------------------------------------------------------------------
# Write HDF5
# ------------------------------------------------------------------

filename = "./tests/test_loading_prediction.h5"

with h5py.File(filename, "w") as h5:

    # ==============================================================
    # Inputs
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

    g_in.create_dataset("L_cylinder_m", data=L)
    g_in.create_dataset("D_cylinder_m", data=D)

    g_in.create_dataset("F_z_prime_N_p_m", data=Fz)
    g_in.create_dataset("F_phi_prime_N_p_m", data=Fphi)
    g_in.create_dataset("F_r_prime_N_p_m", data=np.zeros_like(Fz))

    g_in.create_dataset("U_inf_m_p_s", data = PIN.Ui)
    g_in.create_dataset("B", data=B)

    #TODO: mean velocity profile


    # ==============================================================
    # Outputs
    # ==============================================================
    g_out = h5.create_group("outputs")
    # ---- blade loading harmonics (3, Nk, Nr) complex ----
    g_blade = g_out.create_group("blade_loading_harmonics_N_p_m")
    g_blade.create_dataset("real", data=np.real(blade_harmonics))
    g_blade.create_dataset("imag", data=np.imag(blade_harmonics))

    # ---- strut loading harmonics (3, Nk, Nr) complex ----
    g_strut = g_out.create_group("strut_loading_harmonics_N_p_m")
    g_strut.create_dataset("real", data=np.real(strut_harmonics))
    g_strut.create_dataset("imag", data=np.imag(strut_harmonics))

    # ---- downwash (Nr, Nphi) ----
    g_out.create_dataset(
        "blade_downwash_m_p_s",
        data=blade_downwash
    )

    # ---- coordinates / harmonic information ----
    g_out.create_dataset("azimuth_rad", data=phi)
    g_out.create_dataset("harmonics_k", data=k)
    g_out.create_dataset("frequency_k_Hz", data=frequency)

print(f"Wrote {filename}")






