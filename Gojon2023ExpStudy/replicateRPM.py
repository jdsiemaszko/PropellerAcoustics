import numpy as np
import h5py
import matplotlib.pyplot as plt

datadir = './dataverse_files'
# ----------------------------------------------------------------------
# Global matplotlib settings (MATLAB: set(0,'defaulttextinterpreter','latex'))
# ----------------------------------------------------------------------
# plt.rcParams.update({
#     "text.usetex": False,
#     "mathtext.fontset": "stix",
#     "font.family": "STIXGeneral"
# })

pref = 20e-6  # reference pressure (20 µPa)
NB = 2
# ----------------------------------------------------------------------
# Helper function to mimic MATLAB loadh5 structure access
# ----------------------------------------------------------------------
def load_h5(filename):
    return h5py.File(filename, "r")


def spl_from_autopower(Pa2):
    return 10 * np.log10(Pa2 / (pref * pref))

def plot_BPF_PEAKS(ax, freq_plus, spl, N0=2, N1=25, color='k', alpha=0.01):
    # plot a line connecting the peaks at multiples of BPF
    sol = []
    solfplus = np.arange(N0, N1 + 1)  # harmonic numbers

    for f0 in solfplus:
        # frequency window: ±1%
        fmin = (1-alpha) * f0
        fmax = (1+alpha) * f0

        # indices within the window
        mask = (freq_plus >= fmin) & (freq_plus <= fmax)

        if not np.any(mask):
            # fallback: nearest frequency if window is empty
            idx = np.argmin(np.abs(freq_plus - f0))
            sol.append(spl[idx])
        else:
            sol.append(np.max(spl[mask]))

    ax.plot(solfplus, sol, marker='s', color=color)




# ======================================================================
# ISAE 2 – D10 – L20 – varying RPM
# ======================================================================
fig, ax = plt.subplots(figsize=(10, 6))

with load_h5(f"{datadir}/ISAE_2_D20_L20_autopower.h5") as f:
    g = f["ISAE_2_D20_L20"]
    freq = g["frequency_Hz"]
    ap = g["Autopower"]
    RPMS = [4000, 5000, 6000, 7000, 8000]

    cmap = plt.cm.jet
    colors = iter(cmap(np.linspace(0, 1, len(RPMS))))

    for rpm in RPMS:
        col = next(colors)
        BPF = NB * rpm / 60
        data = ap[f"Autopower_RPM_{rpm}_Pa2"][:, 2, 9] # (freq, polar, azimuth), (aziuth=0 = > beam axis, azimuth=9 => 90 deg)
        ax.plot(freq[0] / BPF, spl_from_autopower(data), label=f"{rpm} RPM", color=col)
        plot_BPF_PEAKS(ax, freq[0] / BPF, spl_from_autopower(data), N0=2, color=col)
        
# with load_h5(f"{datadir}/ISAE_X_Isolated_autopower.h5") as f:
#     g = f["ISAE_X_Isolated"]
#     freq = g["frequency_Hz"]
#     ap = g["Autopower"]
#     RPM = g["RPM"]
#     # plt.figure()
#     RPMS = RPM[0]
#     cmap = plt.cm.jet
#     colors = iter(cmap(np.linspace(0, 1, len(RPMS))))

#     for rpm in RPMS:
#         col = next(colors)
#         BPF = NB * rpm / 60
#         data = ap[f"Autopower_RPM_{rpm}_Pa2"][:, 6, 9] # (freq, polar, azimuth), (aziuth=0 = > beam axis, azimuth=9 => 90 deg)
#         ax.plot(freq[0] / BPF, spl_from_autopower(data), label=f"{rpm} RPM", color=col)
#         plot_BPF_PEAKS(ax, freq[0] / BPF, spl_from_autopower(data), N0=2, N1=10, color=col)
        

ax.legend()
# plt.title(r"ISAE 2 rotor in interaction with D10 beam, $\theta = 0^{\circ}$ and $\phi = 90^{\circ}$")
ax.set_xlabel("$f^+$ (Hz)")
ax.set_ylabel("SPL (dB)")
ax.set_xscale('log')
# ax.set_yscale('log')

ax.grid(visible=True, which='major', color='k', linestyle='-')
ax.grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.5)


plt.xlim(0.1, 100)
plt.ylim(0, 65)
plt.tight_layout()
plt.show()