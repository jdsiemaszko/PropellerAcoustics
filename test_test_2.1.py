import h5py
import numpy as np


filename = "./tests/test_propagator.h5"


# ==============================================================
# Helper: reconstruct complex arrays
# ==============================================================
def read_complex(group):
    real = group["real"][()]
    imag = group["imag"][()]
    return real + 1j * imag


# ==============================================================
# LOAD FILE
# ==============================================================
with h5py.File(filename, "r") as h5:

    # ----------------------------------------------------------
    # INPUTS
    # ----------------------------------------------------------
    inp = h5["inputs"]

    Omega = inp["Omega_rad_p_s"][()]
    rho   = inp["rho_kg_p_m3"][()]
    c0    = inp["sos_m_p_s"][()]

    radius_inner = inp["radius_inner_m"][()]
    radius_outer = inp["radius_outer_m"][()]

    twist_inner = inp["twist_inner_rad"][()]
    twist_outer = inp["twist_outer_rad"][()]

    chord_inner = inp["chord_inner_m"][()]
    chord_outer = inp["chord_outer_m"][()]

    L = inp["L_cylinder_m"][()]
    D = inp["D_cylinder_m"][()]
    B = inp["B"][()]

    # ----------------------------------------------------------
    # COMPLEX LOADINGS
    # ----------------------------------------------------------
    BLH   = read_complex(inp["blade_loading_harmonics_N_p_m"])
    beam_harmonics = read_complex(inp["beam_loading_harmonics_N_p_m"])

    # ----------------------------------------------------------
    # OUTPUTS
    # ----------------------------------------------------------
    out = h5["outputs"]

    p_steady   = read_complex(out["p_loading_blade_steady_Pa"])
    p_unsteady = read_complex(out["p_loading_blade_unsteady_Pa"])
    p_strut    = read_complex(out["p_loading_strut_Pa"])
    p_thickness = read_complex(out["p_thickness_blade_Pa"])

    x_cart = out["observer_positions_m"][()]

    ms = out["harmonics_m"][()]
    freq = out["frequency_mB_Hz"][()]



# ==============================================================
# RECONSTRUCTION CHECKS
# ==============================================================

print("\n========== HDF5 LOAD SUMMARY ==========")

print("Omega:", Omega)
print("rho:", rho)
print("c0:", c0)
print("B:", B)

print("\nShapes:")
print("BLH:", BLH.shape)
print("beam_harmonics:", beam_harmonics.shape)

print("\nPressure fields:")
print("p_steady:", p_steady.shape)
print("p_unsteady:", p_unsteady.shape)
print("p_strut:", p_strut.shape)
print("p_thickness:", p_thickness.shape)

print("\nObserver geometry:")
print("x_cart:", x_cart.shape)

print("\nHarmonics / frequency:")
print("ms:", ms)
print("freq:", freq)


# ==============================================================
# VALIDATION SECTION
# ==============================================================
from Hanson.far_field import HansonModel
han = HansonModel(
        radius_m=radius_outer, # blade radius stations [m] of size Nr + 1
        Omega_rads=Omega, # rotation speed [rad/s]
        rho_kgm3=rho, # fluid density [kg/m^3]
        c_mps= c0, # speed of sound [m/s]
        nb=1, # number of beams
        B=B, 
        )

BLH_s = np.zeros_like(BLH)
BLH_s[:, 0, :] = BLH[:, 0, :]
p_loading_steady, _ = han.getPressureRotor(x_cart, ms, BLH_s)

pass
