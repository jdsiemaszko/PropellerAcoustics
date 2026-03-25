import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from R_Gargiulo_test_01 import read_tecplot_block_dat

# ---------------------------------------------
# PARAMETERS
# ---------------------------------------------

angle = 20
time_indices = np.arange(1, 121, 1)
Nt = time_indices.shape[0]
Nr = 41

# for Nk in [0, 1, 2, 5, 10, 15, 20, 25]:
    # print(Nk)
Nk = 10
k  = np.arange(0, Nk+1, 1)
force_time = np.zeros((3, Nt, Nr))

# ---------------------------------------------
# ROTOR / ACOUSTIC PARAMETERS
# ---------------------------------------------

B        = 8
timestep = 5.26e-5
RPM      = 9505
Omega    = 2 * np.pi / 60 * RPM
time     = time_indices * timestep
period   = 2 * np.pi / Omega
BPF      = B * Omega
T_ref_avg = {5: 31.73, 10: 33.00, 15: 35.18, 20: 37.43}

c        = 340
lam      = c / (BPF / 2 / np.pi)
ROBS     = 10 * lam * 2 * np.pi

# ---------------------------------------------
# LOAD FORCE DATA FROM TECPLOT FILES
# ---------------------------------------------

for ind, t in enumerate(time_indices):
    data = read_tecplot_block_dat(
        r"Data/Gargiulo2026/run2/acoustics_data_AoA_20.0deg/acoustics_data_AoA_20.0deg/fR/"+
        f"traction_Blade_RPM9505.0_46_elts_pitch_28.0_t{t:03d}_AoA_{angle}.0.dat"
    )

    r = data["R_middle"][:Nr]

    dr = np.diff(r)
    dr = np.append(dr, dr[-1])

    r_outer = np.append(r-dr/2, r[-1]+dr[-1]/2)

    F_radial = data["Fr_fR"][:Nr]
    F_tan    = data["Ft_fR"][:Nr]
    F_ax     = data["Tz_fR"][:Nr]

    force_time[0, ind, :] = F_radial
    force_time[1, ind, :] = F_ax
    force_time[2, ind, :] = F_tan

# ---------------------------------------------
# FOURIER DECOMPOSITION IN TIME (EXPLICIT DFT)
# ---------------------------------------------

Nk = k.shape[0]

force_freq = 1/period * np.sum(force_time[:, None, :, :] * np.exp(+1j *
                k[None, :, None, None] * 2 * np.pi / period *
                time[None, None, :, None]) * timestep , axis=2)

# TRUNCATE FORCES BELOW A THRESHOLD

threshold = 1e-4 # play around with cutoff value
fmax = np.max(np.abs(force_freq))

force_freq[np.where(np.abs(force_freq) < threshold * fmax)] = 0 # set all noise to zero

#force_freq[:,5:,:]=0

# ---------------------------------------------
# FOURIER AMPLITUDE OF THRUST (Z COMPONENT)
# ---------------------------------------------

freq    = k * Omega / (2*np.pi)
fshaft  = Omega / (2*np.pi)
amp_z   = np.abs(np.sum(force_freq[:,:,:], axis=-1))

plt.figure(figsize=(6,4))
plt.plot(freq/fshaft, amp_z[0]/T_ref_avg[angle], marker='o', color='r')
plt.plot(freq/fshaft, amp_z[1]/T_ref_avg[angle], marker='x', color='b')
plt.plot(freq/fshaft, amp_z[2]/T_ref_avg[angle], marker='s', color='g')

plt.xlabel("f/fshaft [-]", fontsize=14)
plt.ylabel(r"Fourier amplitude $\left|\frac{\widehat{T_z^k}}{\overline{T_z}}\right|$ [-]", fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.grid(True)
# plt.xscale('log')
plt.yscale('log')

plt.tight_layout()
plt.show()

# ---------------------------------------------
# BUILD SPANWISE LOADING ARRAY (force per unit span)
# ---------------------------------------------

Fblade = np.zeros((3, Nk, len(r)), dtype=np.complex128)
Fblade[0, :, :] =  force_freq[0, :, :] / dr
Fblade[1, :, :] = -force_freq[1, :, :] / dr
Fblade[2, :, :] =  force_freq[2, :, :] / dr


# ---------------------------------------------
# HANSON ACOUSTIC MODEL
# ---------------------------------------------

from Hanson.far_field import HansonModel

kmax = k[-1]
BPF  = B * Omega
c    = 340
lam  = c / (BPF / 2 / np.pi)
ROBS = 10 * lam * 2 * np.pi

hm = HansonModel(
                radius_m   = r_outer,
                axis       = np.array([0, 0, 1]),
                origin     = np.array([0, 0, 0]),
                radial     = np.array([1, 0, 0]),
                B          = B,
                Omega_rads = 9505 / 60 * 2 * np.pi,
                rho_kgm3   = 1.2,
                c_mps      = c,
                nb         = 0)

# ---------------------------------------------
# PLOTS
# ---------------------------------------------

# 1) 3D directivity plots for harmonics m = 1 ... 5
# for m in [1]:
#     fig = plt.figure(figsize=(7, 7))
#     ax1 = fig.add_subplot(111, projection="3d")
#     hm.plot3Ddirectivity(
#         fig=fig,
#         ax=ax1,
#         m=m,
#         R=ROBS,
#         Nphi=36,
#         Ntheta=18,
#         title='far-field',
#         mode='rotor',
#         loadings=Fblade,
#         valmin=10,
#         valmax=80,
#     )
#     ax1.tick_params(axis='both', labelsize=12)
#     ax1.set_xlabel(ax1.get_xlabel(), fontsize=14)
#     ax1.set_ylabel(ax1.get_ylabel(), fontsize=14)
#     ax1.set_zlabel(ax1.get_zlabel(), fontsize=14)
#     plt.tight_layout()
#     plt.show()

# 2) Plot directivity (2D polar plot)
# hm.plot2Ddirectivity(
#     m=1, # harmonic to plot
#     R=ROBS, # observation radius
#     Ntheta=360,
#     mode='rotor', # 'rotor' or 'stator' or 'total'
#     loadings=Fblade, # blade loading harmonics
#     # plane='xy'
#     plane='xz'

# )
# plt.tight_layout()
# plt.show()

# 3) plot 2D contours
for m in [3]:
    fig, ax = plt.subplots()
    hm.plotDirectivityContour(
        fig=fig,
        ax=ax,
        m=m,
        R=ROBS,
        Nphi=36,
        Ntheta=18,
        title='far-field',
        mode='rotor',
        loadings=Fblade,
        valmin=10,
        valmax=80,
    )
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel(ax.get_xlabel(), fontsize=14)
    ax.set_ylabel(ax.get_ylabel(), fontsize=14)
    plt.tight_layout()
    plt.show()


# 4) plot pressure spectrum at a single observer point
fig, ax = plt.subplots()
hm.plotPressureSpectrum(
    fig=fig, ax=ax,
    x        = np.array([1.0, 0.0, 0.0]).T,
    m        = np.arange(1, 25, 1),
    loadings = Fblade
)
ax.set_xlabel(ax.get_xlabel(), fontsize=14)
ax.set_ylabel(ax.get_ylabel(), fontsize=14)
ax.tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.show()