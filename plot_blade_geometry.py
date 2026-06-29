
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from Constants.data_assim import getGojonData, getHarmonicsFromData
from PotentialInteraction.PIN import PotentialInteraction
from Constants.helpers import read_force_file, plot_3D_directivity, plot_3D_phase_directivity, plot_beam_azimuth, plot_rotation_arrow
import matplotlib.colors as colors
from matplotlib.lines import Line2D

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
Fz_p *= TTARGET / np.trapezoid(Fz_p, r_inner)  # rescale to target
Fphi_p *= QTARGET / np.trapezoid(Fphi_p * r_inner, r_inner) # rescale to target

Omega_p = 7250 / 60 * 2 * np.pi
rc, c = np.loadtxt('./Data/Parrot2024/chord.csv', skiprows=1, delimiter=',').T # radius, chord in meters
rp, p = np.loadtxt('./Data/Parrot2024/pitch.csv', skiprows=1, delimiter=',').T # radius, pitch in degrees

chord_parrot = np.interp(r_inner, rc, c)
pitch_parrot = np.interp(r_inner, rp, p)
chord = 0.025 * np.ones_like(r_inner)
Omega = 8000/60*2*np.pi

CASES = [
    # baseline
    {
        'D' : 0.02,
        'chord' : chord,
        'Omega' : Omega,
        'twist' : 10 * np.ones_like(chord),
        'marker' : 's',
        'linestyle' : '-',
        'Fz' : Fz
    },
        {
        'D' : 0.02,
        'chord' : chord_parrot,
        'Omega' : Omega_p,
        # 'twist' : np.rad2deg(pitch_parrot),
        'twist' : pitch_parrot,
        'marker' : 'o',
        'linestyle' : '--',
        'Fz' : Fz_p
    },
]

fig, ax = plt.subplots(figsize=(8, 4))
ax2 = ax.twinx()
for case in CASES:
    c0 = 340
    B = 2
    D=case['D']
    R=D/2
    t_c = 0.0822
    chor = case['chord']
    Omeg = case['Omega']
    twis = case['twist']

    Fzz = case['Fz']
    CL = Fzz / 0.5 / 1.2 / (Omeg * r_inner)**2 / chor

    F1 = CL / 4 / np.pi * chor / r_inner
    F2 = CL / 4 * R / r_inner * R / chor * 1/t_c
    F3 = 1/2/np.pi * t_c * (chor / R)**2

    lambda0 = c0 / Omeg * 2 * np.pi * B
    Mach_r  = Omeg * r_inner / c0
    He_Mr = chor / lambda0 * 2 * np.pi / Mach_r


    # --- primary axis ---
    ax.plot(r_inner[::2] / r1, chor[::2], color='r', marker=case['marker'], linestyle=case['linestyle'])[0]
    ax2.plot(r_inner[::2] / r1, twis[::2], color='b', marker=case['marker'], linestyle=case['linestyle'])[0]


ax.set_xlabel('$r/r_t$')
ax.set_ylabel('chord $[m]$')
ax2.set_ylabel('twist angle $[^\circ]$')
# ax.set_ylabel(r'$F = C_l c / 4\pi r$')
ax.grid()
# --- secondary axis ---
# ax2 = ax.twinx()
# l2 = ax2.plot(r_inner / r1, He_Mr, color='b', label=r'$He_0 / M_r$')[0]
# ax2.set_ylabel(r'$He_0 / M_r = B c / r$')
case_handles = [
    Line2D([0], [0], color='k', marker='s', linestyle='-',
           label='academic'),
    Line2D([0], [0], color='k', marker='o', linestyle = '--',
           label='optimized'),
]

param_handles = [
    Line2D([0], [0], color='r', label='chord $[m]$'),
    Line2D([0], [0], color='b', label='twist angle $[^\circ]$'),
]
leg2 = ax.legend(
    handles=param_handles,
    loc='lower center',          # keeps it inside automatically
    frameon=True, fontsize=8

)

leg3 = ax.legend(
    handles=case_handles,
    loc='lower left',          # keeps it inside automatically
    frameon=True, fontsize=8
)

ax.add_artist(leg2)
ax.add_artist(leg3)

plt.tight_layout()
plt.show()
fig.savefig('./Figures/blade_geometry.pdf')