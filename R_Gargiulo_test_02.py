import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from R_Gargiulo_test_01 import read_tecplot_block_dat

# data = read_tecplot_block_dat("Data/Gargiulo2026/raw_01.dat")

time_indices = np.arange(1,121,1)
Nt = time_indices.shape[0]
Nr = 40
force_time = np.zeros((3, Nt, Nr))
B = 8
timestep = 5.26e-5 # s
RPM = 9505
Omega = 2 * np.pi / 60 * RPM
time = time_indices * timestep
period = 2 * np.pi / Omega # seconds


# read time data
for ind, t in enumerate(time_indices):
    data = read_tecplot_block_dat(f"Data/Gargiulo2026/traction_Blade_RPM9505.0_46_elts_pitch_28.0_t{t:03d}_AoA_5.0.dat")

    # print(data.keys())

    r = data["R_middle"][:Nr] # segment centers, in units meter

    dr = np.diff(r)
    dr = np.append(dr, dr[-1]) # segment lengths

    r_outer = np.append(r-dr/2, r[-1]+dr[-1]/2) # segment edges

    F_radial = data["Fr_fR"][:Nr] # in units Newton # of shape (Nr, )
    F_tan = data["Ft_fR"][:Nr]
    F_ax = data["Tz_fR"][:Nr]

    # store loading data
    force_time[0, ind, :] = F_radial
    force_time[0, ind, :13] = 0 # ignore the erreneous loading near the root
    force_time[1, ind, :] = F_ax
    force_time[2, ind, :] = F_tan

# Fourier transform
k = np.arange(1, 21, 1) # example
Nk = k.shape[0]
force_freq = 1/period * np.sum(force_time[:, None, :, :] * np.exp(+1j *
                 k[None, :, None, None] * 2 * np.pi / period * 
                 time[None, None, :, None]) , axis=2) # explicitly doing fft , result of shape (3, Nk, Nr)





# fill the loading-per-unit-span array
Fblade = np.zeros((3, Nk, len(r)), dtype=np.complex128)
Fblade[0, :, :] = force_freq[0, :, :] / dr # not used
Fblade[1, :, :] = -force_freq[1, :, :] / dr # unit Newton per meter
Fblade[2, :, :] = force_freq[2, :, :]  / dr # force oriented backwards -> positive in our sign convention

from Hanson.far_field import HansonModel # my module

kmax = k[-1]
BPF = B * Omega
c = 340.0 #m/s
lam = c/(BPF/2/np.pi) # m

ROBS = 10 * lam * 2 * np.pi # observation radius, in meters

# Initialize Module
hm = HansonModel(
                radius_m = r_outer, # blade radius stations [m] of size Nr + 1\
                axis=np.array([0, 0, 1]), origin=np.array([0, 0, 0]), radial=np.array([1, 0, 0]), # coordinate system (not needed here)
                B=B, # number of blades
                Omega_rads = 9505 / 60 * 2 * np.pi, # rotation speed [rad/s]
                rho_kgm3 = 1.2, # fluid density [kg/m^3]
                c_mps = 340., # speed of sound [m/s]
                nb = 0 # number of beams (irrelevant)
                )


# 1) Plot directivity (3D)

for m in [1, 2, 3, 4, 5]:
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111, projection="3d")
    hm.plot3Ddirectivity(
        fig=fig,
        ax=ax1,
        m=m, # harmonic to plot
        R=ROBS, # observation radius
        Nphi=36*3, # plotting params
        Ntheta=18*3,
        # valmin=40,
        # valmax=65,
        title='far-field',
        mode='rotor', # 'rotor' or 'stator' or 'total'
        loadings=Fblade # blade loading harmonics
    )
    plt.tight_layout()
    plt.show()


# 2) Plot spectrum at a point
fig, ax = plt.subplots()
hm.plotPressureSpectrum(fig=fig, ax=ax, 
                        x =np.array([1.0, 0.0, 0.0]).T, # position to plot the spectrum at
                        m = np.arange(1, 11, 1), # modes to compute
                        loadings=Fblade # blade loading harmonics
                            )
plt.tight_layout()
plt.show()
