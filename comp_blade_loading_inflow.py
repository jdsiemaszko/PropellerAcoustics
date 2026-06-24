
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

U_z = np.sqrt(Fz * 2 /  4 / np.pi / 1.2 / r_inner)
U_phi = Fphi * 2 / 4 / np.pi / 1.2 / r_inner / np.abs(U_z)


lambda0 = c0 / Omega * 2 * np.pi * B
Mach_r  = Omega * r_inner / c0
He_Mr = chord / lambda0 * 2 * np.pi / Mach_r

fig, ax = plt.subplots(figsize=(8, 3))

# --- primary axis ---
l1 = ax.plot(r_inner / r1, Fz, color='r', label='$F_z$')[0]
l2 = ax.plot(r_inner / r1, Fphi, color='r', linestyle='--', label='$F_\phi$')[0]
ax2 = ax.twinx()
l3 = ax2.plot(r_inner / r1, U_z, color='b', label='$U_z$')[0]
l4 = ax2.plot(r_inner / r1, U_phi, color='b', linestyle='--', label='$U_\phi$')[0]


ax.set_xlabel('$r/r_t$')
ax.set_ylabel(r'$F [N/m]$')
ax2.set_ylabel(r'$U [m/s]$')

ax.grid()
# --- secondary axis ---
# ax2 = ax.twinx()
# l2 = ax2.plot(r_inner / r1, He_Mr, color='b', label=r'$He_0 / M_r$')[0]
# ax2.set_ylabel(r'$He_0 / M_r = B c / r$')

# --- legend (IMPORTANT FIX) ---
ax.legend(
    handles=[l1, l2, l3, l4],
    loc='best',          # keeps it inside automatically
    frameon=True, fontsize = 8
)

plt.tight_layout()
plt.show()
fig.savefig('./Figures/blade_loading_inflow.pdf')