import h5py
import numpy as np
from PotentialInteraction.PIN import PotentialInteraction
import matplotlib.pyplot as plt

filename = "./tests/test_loading_prediction.h5"

with h5py.File(filename, "r") as h5:

    # ==============================================================
    # Inputs
    # ==============================================================
    Omega = h5["inputs/Omega_rad_p_s"][()]
    rho   = h5["inputs/rho_kg_p_m3"][()]
    c0    = h5["inputs/sos_m_p_s"][()]

    r_inner = h5["inputs/radius_inner_m"][()]
    r_outer = h5["inputs/radius_outer_m"][()]

    twist= h5["inputs/twist_outer_rad"][()]
    chord       = h5["inputs/chord_outer_m"][()]
    # t_c = h5["inputs/t_c_outer"][()]

    L = h5["inputs/L_cylinder_m"][()]
    D = h5["inputs/D_cylinder_m"][()]

    Fz = h5["inputs/F_z_prime_N_p_m"][()]
    Fphi = h5["inputs/F_phi_prime_N_p_m"][()]
    Ui = h5["inputs/U_inf_m_p_s"][()]
    B = h5["inputs/B"][()]


    # ==============================================================
    # Outputs
    # ==============================================================
    blade_downwash = h5["outputs/blade_downwash_m_p_s"][()]
    phi             = h5["outputs/azimuth_rad"][()]
    k               = h5["outputs/harmonics_k"][()]
    frequency       = h5["outputs/frequency_k_Hz"][()]

    # ---- reconstruct complex blade loading harmonics ----
    blade_real = h5["outputs/blade_loading_harmonics_N_p_m/real"][()]
    blade_imag = h5["outputs/blade_loading_harmonics_N_p_m/imag"][()]
    blade_harmonics = blade_real + 1j * blade_imag

    # ---- reconstruct complex strut loading harmonics ----
    strut_real = h5["outputs/strut_loading_harmonics_N_p_m/real"][()]
    strut_imag = h5["outputs/strut_loading_harmonics_N_p_m/imag"][()]
    beam_harmonics = strut_real + 1j * strut_imag

PIN = PotentialInteraction(
twist_rad=twist,
chord_m=chord,
radius_m=r_outer,
t_c = np.zeros_like(r_outer), # ???
Fzprime_Npm=Fz,
Fphiprime_Npm=Fphi,
B=B,
Dcylinder_m=D,
Lcylinder_m=L,
Omega_rads=Omega,
rho_kgm3=rho,
c_mps=c0,
kmax=len(k)-1,
nb=1,
numerics={'Nphi': 180, 'Nthetab': 36},
U0_mps = Ui
)
PIN._numerics['gamma_steady'] = True

pass