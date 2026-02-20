import numpy as np
import h5py
import matplotlib.pyplot as plt

datadir = './dataverse_files'
# ----------------------------------------------------------------------
# Global matplotlib settings (MATLAB: set(0,'defaulttextinterpreter','latex'))
# ----------------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral"
})

pref = 20e-6  # reference pressure (20 µPa)

# ----------------------------------------------------------------------
# Helper function to mimic MATLAB loadh5 structure access
# ----------------------------------------------------------------------
def load_h5(filename):
    return h5py.File(filename, "r")


def spl_from_autopower(Pa2):
    return 10 * np.log10(Pa2 / (pref * pref))


# ======================================================================
# ISAE 2 – D10 – L20 – varying RPM
# ======================================================================
with load_h5(f"{datadir}/ISAE_2_D20_L20_autopower.h5") as f:
    g = f["ISAE_2_D20_L20"]
    freq = g["frequency_Hz"]
    ap = g["Autopower"]

    plt.figure()
    for rpm in [4000, 5000, 6000, 7000, 8000]:
        BPF = 2 * rpm / 60 * 2 * np.pi
        data = ap[f"Autopower_RPM_{rpm}_Pa2"][:, 6, 9]
        plt.semilogx(freq[0] / BPF, spl_from_autopower(data), label=f"{rpm} RPM")

    plt.legend()
    # plt.title(r"ISAE 2 rotor in interaction with D10 beam, $\theta = 0^{\circ}$ and $\phi = 90^{\circ}$")
    plt.xlabel("$f^+$ (Hz)")
    plt.ylabel("SPL (dB)")
    plt.xlim(10, 25000)
    plt.ylim(0, 65)
    plt.show()


# ======================================================================
# ISAE 2 – D15 – RPM 8000 – varying distance
# ======================================================================
with load_h5(f"{datadir}/ISAE_2_D15_8000_autopower.h5") as f:
    g = f["ISAE_2_D15_8000"]
    freq = g["frequency_Hz"][:]
    ap = g["Autopower"]

    plt.figure()
    for L in [20, 30, 40, 50, 60]:
        data = ap[f"Autopower_L{L}_Pa2"][:, 6, 9]
        plt.semilogx(freq[0], spl_from_autopower(data), label=f"L={L} mm")

    plt.legend()
    plt.title(r"ISAE 2 rotor in interaction with D15 beam, $\theta = 0^{\circ}$ and $\phi = 90^{\circ}$")
    plt.xlabel("f (Hz)")
    plt.ylabel("SPL (dB)")
    plt.xlim(10, 25000)
    plt.ylim(0, 65)
    plt.show()


# ======================================================================
# ISAE 2 – D15 – L20 – varying RPM
# ======================================================================
with load_h5(f"{datadir}/ISAE_2_D15_L20_autopower.h5") as f:
    g = f["ISAE_2_D15_L20"]
    freq = g["frequency_Hz"][:]
    ap = g["Autopower"]

    plt.figure()
    for rpm in [4000, 5000, 6000, 7000, 8000]:
        data = ap[f"Autopower_RPM_{rpm}_Pa2"][:, 6, 9]
        plt.semilogx(freq[0], spl_from_autopower(data), label=f"{rpm} RPM")

    plt.legend()
    plt.title(r"ISAE 2 rotor in interaction with D15 beam, $\theta = 0^{\circ}$ and $\phi = 90^{\circ}$")
    plt.xlabel("f (Hz)")
    plt.ylabel("SPL (dB)")
    plt.xlim(10, 25000)
    plt.ylim(0, 65)
    plt.show()


# ======================================================================
# ISAE 2 – D20 – L20 – varying RPM
# ======================================================================
with load_h5(f"{datadir}/ISAE_2_D20_L20_autopower.h5") as f:
    g = f["ISAE_2_D20_L20"]
    freq = g["frequency_Hz"]
    ap = g["Autopower"]

    plt.figure()
    for rpm in [4000, 5000, 6000, 7000, 8000]:
        data = ap[f"Autopower_RPM_{rpm}_Pa2"][:, 6, 9]
        plt.semilogx(freq[0], spl_from_autopower(data), label=f"{rpm} RPM")

    plt.legend()
    plt.title(r"ISAE 2 rotor in interaction with D20 beam, $\theta = 0^{\circ}$ and $\phi = 90^{\circ}$")
    plt.xlabel("f (Hz)")
    plt.ylabel("SPL (dB)")
    plt.xlim(10, 25000)
    plt.ylim(0, 65)
    plt.show()


# ======================================================================
# ISAE 2 – S10 – L20 – varying RPM
# ======================================================================
with load_h5(f"{datadir}/ISAE_2_S10_L20_autopower.h5") as f:
    g = f["ISAE_2_S10_L20"]
    freq = g["frequency_Hz"][:]
    ap = g["Autopower"]

    plt.figure()
    for rpm in [4000, 5000, 6000, 7000, 8000]:
        data = ap[f"Autopower_RPM_{rpm}_Pa2"][:, 6, 9]
        plt.semilogx(freq[0], spl_from_autopower(data), label=f"{rpm} RPM")

    plt.legend()
    plt.title(r"ISAE 2 rotor in interaction with S10 beam, $\theta = 0^{\circ}$ and $\phi = 90^{\circ}$")
    plt.xlabel("f (Hz)")
    plt.ylabel("SPL (dB)")
    plt.xlim(10, 25000)
    plt.ylim(0, 65)
    plt.show()


# ======================================================================
# ISAE 2 – T10 – L20 – varying RPM
# ======================================================================
with load_h5(f"{datadir}/ISAE_2_T10_L20_autopower.h5") as f:
    g = f["ISAE_2_T10_L20"]
    freq = g["frequency_Hz"][:]
    ap = g["Autopower"]

    plt.figure()
    for rpm in [4000, 5000, 6000, 7000, 8000]:
        data = ap[f"Autopower_RPM_{rpm}_Pa2"][:, 6, 9]
        plt.semilogx(freq[0], spl_from_autopower(data), label=f"{rpm} RPM")

    plt.legend()
    plt.title(r"ISAE 2 rotor in interaction with T10 beam, $\theta = 0^{\circ}$ and $\phi = 90^{\circ}$")
    plt.xlabel("f (Hz)")
    plt.ylabel("SPL (dB)")
    plt.xlim(10, 25000)
    plt.ylim(0, 65)
    plt.show()


# ======================================================================
# ISAE X – isolated rotors
# ======================================================================
with load_h5(f"{datadir}/ISAE_X_Isolated_autopower.h5") as f:
    g = f["ISAE_X_Isolated"]
    freq = g["frequency_Hz"][:]
    ap = g["Autopower"]

    plt.figure()
    plt.semilogx(freq[0], spl_from_autopower(ap["Autopower_ISAE_2_Pa2"][:, 6, 4]), label="ISAE2 - 8000 RPM")
    plt.semilogx(freq[0], spl_from_autopower(ap["Autopower_ISAE_3_Pa2"][:, 6, 4]), label="ISAE3 - 8000 RPM")
    plt.semilogx(freq[0], spl_from_autopower(ap["Autopower_ISAE_4_Pa2"][:, 6, 4]), label="ISAE4 - 8000 RPM")

    plt.legend()
    plt.title(r"ISAE X rotors, $\theta = 0^{\circ}$")
    plt.xlabel("f (Hz)")
    plt.ylabel("SPL (dB)")
    plt.xlim(10, 25000)
    plt.ylim(0, 65)
    plt.show()
