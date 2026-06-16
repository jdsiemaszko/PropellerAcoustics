
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from Constants.data_assim import getGojonData, getHarmonicsFromData
from PotentialInteraction.PIN import PotentialInteraction
from Constants.helpers import read_force_file, plot_3D_directivity, plot_3D_phase_directivity, plot_beam_azimuth, plot_rotation_arrow

r_inner, Fz, Fphi  = read_force_file('./Data/Zamponi2026/FS_ISAE_2_8000.txt') # reuse the radial stations from data

rt, t =  np.loadtxt('./Data/Parrot2024/thrust_Npm.csv', skiprows=1, delimiter=',').T # radius/r1, thrust in Npm
rq, q =  np.loadtxt('./Data/Parrot2024/torque_Nmpm.csv', skiprows=1, delimiter=',').T # radius/r1, torque in Nmpm

q /= 1.125 # correct the torque to aero data


r1 = 0.1
Fz_p = np.interp(r_inner/r1, rt, t) # same radial array
Q = np.interp(r_inner/r1, rq, q) 
Fphi_p = Q / r_inner

TTARGET = 2.15 / 2 # Newtons
QTARGET = 25 / 1000 / 2 # Newton-radian-meters
Fz_p *= TTARGET / np.trapezoid(Fz, r_inner)  # rescale to target
Fphi_p *= QTARGET / np.trapezoid(Fphi * r_inner, r_inner) # rescale to target

c0 = 340
B = 2
D=0.02
R=D/2
t_c = 0.0822
chord = 0.025 * np.ones_like(r_inner)
Omega = 8000/60*2*np.pi
CL = Fz / 0.5 / 1.2 / (Omega * r_inner)**2 / chord

F1 = CL / 4 / np.pi * chord / r_inner
F2 = CL / 4 * R / r_inner * R / chord * 1/t_c
F3 = 1/2/np.pi * t_c * (chord / R)**2

Omega_p = 7250 / 60 * 2 * np.pi
rc, c = np.loadtxt('./Data/Parrot2024/chord.csv', skiprows=1, delimiter=',').T # radius, chord in meters
rp, p = np.loadtxt('./Data/Parrot2024/pitch.csv', skiprows=1, delimiter=',').T # radius, pitch in degrees

chord_parrot = np.interp(r_inner, rc, c)
pitch_parrot = np.interp(r_inner, rp, p)

CL_p = Fz / 0.5 / 1.2 / (Omega_p * r_inner)**2 / chord_parrot
F_p = CL / 4 / np.pi * chord_parrot / r_inner


lambda0 = c0 / Omega * 2 * np.pi * B
Mach_r  = Omega * r_inner / c0
He_Mr = chord / lambda0 * 2 * np.pi / Mach_r

fig, ax = plt.subplots(figsize=(8, 3))

# --- primary axis ---
l1 = ax.plot(r_inner / r1, F1, color='r', label='$F_1 = C_L c/4\pi r $')[0]
# l1 = ax.plot(r_inner / r1, F_p, color='c', label='$F_1 = C_L c/4\pi r $')[0]

l2 = ax.plot(r_inner / r1, F2, color='g', label='$F_2 = C_L R^2/4rt$')[0]
l3 = ax.plot(r_inner / r1, F3, color='b', label='$F_3 = tc/2\pi R^2$')[0]
l4 = ax.plot(r_inner / r1, He_Mr, color='m', label='$He_0 / M_r$')[0]


ax.set_xlabel('$r/r_t$')
# ax.set_ylabel(r'$F = C_l c / 4\pi r$')
ax.grid()
ax.set_yscale('log')
# --- secondary axis ---
# ax2 = ax.twinx()
# l2 = ax2.plot(r_inner / r1, He_Mr, color='b', label=r'$He_0 / M_r$')[0]
# ax2.set_ylabel(r'$He_0 / M_r = B c / r$')

# --- legend (IMPORTANT FIX) ---
ax.legend(
    handles=[l1, l2, l3, l4],
    loc='best',          # keeps it inside automatically
    frameon=True
)

plt.tight_layout()
plt.show()
# plt.savefig('./Figures/blade_params_single.pdf')