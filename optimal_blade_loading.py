
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from Constants.data_assim import getGojonData, getHarmonicsFromData
from PotentialInteraction.PIN import PotentialInteraction
from Constants.helpers import read_force_file, plot_3D_directivity, plot_3D_phase_directivity, plot_beam_azimuth, plot_rotation_arrow
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

r_inner, Fz, Fphi  = read_force_file('./Data/Zamponi2026/FS_ISAE_2_8000.txt') # reuse the radial stations from data

r = np.linspace(0.016, 0.1, 100)
Fz = np.interp(r, r_inner, Fz)
Fphi = np.interp(r, r_inner, Fphi)
r_inner = r

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
rho0 = 1.2
B = 2
D=0.02
R=D/2
L = 0.02
t_c = 0.0822
chord = 0.025 * np.ones_like(r_inner)
Omega = 8000/60*2*np.pi
CL = Fz / 0.5 / 1.2 / (Omega * r_inner)**2 / chord

# updated
F1 = np.sqrt(B * CL / 8 / np.pi * chord / r_inner)
F2 = CL / 4  / np.pi * chord / L
F3 = 1/2/np.pi * t_c * (chord / L)**2

Omega_p = 7250 / 60 * 2 * np.pi
rc, c = np.loadtxt('./Data/Parrot2024/chord.csv', skiprows=1, delimiter=',').T # radius, chord in meters
rp, p = np.loadtxt('./Data/Parrot2024/pitch.csv', skiprows=1, delimiter=',').T # radius, pitch in degrees

chord_parrot = np.interp(r_inner, rc, c)
pitch_parrot = np.interp(r_inner, rp, p)

CL_p = Fz / 0.5 / 1.2 / (Omega_p * r_inner)**2 / chord_parrot
F_p = CL / 4 / np.pi * chord_parrot / r_inner


lambda0 = c0 / Omega * 2 * np.pi / B
Mach_r  = Omega * r_inner / c0
He_Mr = chord / lambda0 * 2 * np.pi / Mach_r


fig, ax = plt.subplots(figsize=(5, 3))

factor_opt = 1/2
# CL_opt = t_c * chord / L / factor_opt
# Lift_opt = 0.5 * CL_opt * rho0 * (Omega * r_inner)**2 * chord

L_opt = t_c * chord / CL /  factor_opt

# l6 = ax.plot(r_inner / r1, Lift_opt, color='r', label=r'$L_{\mathrm{opt}}$', linestyle=(5, (10, 3)))[0]
# l1 = ax.plot(r_inner / r1, CL_opt, color='r', label=r'$C_{L_{\mathrm{opt}}}$', linestyle=(5, (10, 3)))[0]
l6 = ax.plot(r_inner / r1, L_opt * 1000, color='r', label=r'$L_{\mathrm{opt}}$', linestyle=(5, (10, 3)))[0]

ax.set_xlabel(r'$r/r_{\mathrm{tip}}$')
ax.set_ylabel(r'$L$ [mm]')
# ax.set_yticks(np.linspace(-np.pi, np.pi, 17), minor=True)

plt.minorticks_on()
# Grid
ax.grid(which='major', axis='both', linestyle='-')
ax.grid(which='minor', linestyle='--', alpha=0.5)
# ax.set_yscale('log')
# ax.set_ylim(5e-4, 1e0)
# ax.set_ylim(1e-2, 5e1)
ax.set_xlim(r_inner[0]/r_inner[-1], 1)
# ax.set_ylim(0, 0.8)

# --- secondary axis ---
# ax2 = ax.twinx()
# l2 = ax2.plot(r_inner / r1, He_Mr, color='b', label=r'$He_0 / M_r$')[0]
# ax2.set_ylabel(r'$He_0 / M_r = B c / r$')

# --- legend (IMPORTANT FIX) ---
# ax.legend(
#     handles=[l1, l2, l3, l4, l6, l5],
#     # handles=[l1, l3, l5, l2, l4],

#     # loc='lower left',          # keeps it inside automatically
#     loc='upper right',
#     frameon=True, fontsize = 10, ncol=2
# )
np.savetxt(r'C:\Users\ThinkPad\Desktop\delft temp\opt_lift\opt_L.csv', L_opt* 1000)

plt.tight_layout()
plt.show()
fig.savefig('./Figures/L_opt.pdf')