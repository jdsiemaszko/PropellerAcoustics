
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
        'marker' : 's',
        'linestyle' : '-',
        'Fz' : Fz
    },
    # 6000 RPM
        {
        'D' : 0.02,
        'chord' : chord,
        'Omega' : 6000/60*2*np.pi,
        'marker' : '^',
        'linestyle' : '--',
        'Fz' : Fz * (6000 / 8000)**2
    },
# D10L20
        {
        'D' : 0.01,
        'chord' : chord,
        'Omega' : Omega,
        'marker' : 'o',
        'linestyle' : ':',
        'Fz' : Fz
    },
#D15L20
        {
        'D' : 0.015,
        'chord' : chord,
        'Omega' : Omega,
        'marker' : 'p',
        'linestyle' : 'dashdot',
        'Fz' : Fz
    },

        {
        'D' : 0.02,
        'chord' : chord_parrot,
        'Omega' : Omega_p,
        'marker' : '*',
        'linestyle' : (5, (10, 3)),
        'Fz' : Fz_p
    },
]

fig, ax = plt.subplots(figsize=(8, 6))

for case in CASES:
    c0 = 340
    B = 2
    D=case['D']
    R=D/2
    t_c = 0.0822
    chor = case['chord']
    Omeg = case['Omega']

    Fzz = case['Fz']
    CL = Fzz / 0.5 / 1.2 / (Omeg * r_inner)**2 / chor

    F1 = CL / 4 / np.pi * chor / r_inner
    F2 = CL / 4 * R / r_inner * R / chor * 1/t_c
    F3 = 1/2/np.pi * t_c * (chor / R)**2

    lambda0 = c0 / Omeg * 2 * np.pi * B
    Mach_r  = Omeg * r_inner / c0
    He_Mr = chor / lambda0 * 2 * np.pi / Mach_r


    # --- primary axis ---
    ax.plot(r_inner / r1, F1, color='r', marker=case['marker'], linestyle=case['linestyle'])[0]
    ax.plot(r_inner / r1, F2, color='g', marker=case['marker'], linestyle=case['linestyle'])[0]
    ax.plot(r_inner / r1, F3, color='b', marker=case['marker'], linestyle=case['linestyle'])[0]
    ax.plot(r_inner / r1, He_Mr, color='m', marker=case['marker'], linestyle=case['linestyle'])[0]


ax.set_xlabel('$r/r_t$')
# ax.set_ylabel(r'$F = C_l c / 4\pi r$')
ax.grid()
ax.set_yscale('log')
# --- secondary axis ---
# ax2 = ax.twinx()
# l2 = ax2.plot(r_inner / r1, He_Mr, color='b', label=r'$He_0 / M_r$')[0]
# ax2.set_ylabel(r'$He_0 / M_r = B c / r$')
case_handles = [
    Line2D([0], [0], color='k', marker='s', linestyle='-',
           label='academic, D20L20, 8000 RPM'),
    Line2D([0], [0], color='k', marker='^', linestyle='--',
           label='academic, D20L20, 6000 RPM'),
    Line2D([0], [0], color='k', marker='o', linestyle=':',
           label='academic, D10L20, 8000 RPM'),
    Line2D([0], [0], color='k', marker='p', linestyle='dashdot',
           label='academic, D15L20, 8000 RPM'),
    Line2D([0], [0], color='k', marker='*', linestyle = (5, (10, 3)),
           label='optimized, D20L20, 7250 RPM'),
]

param_handles = [
    Line2D([0], [0], color='r', label='$F_1 = C_L c/4\pi r $'),
    Line2D([0], [0], color='g', label='$F_2 = C_L R^2/4rt$'),
    Line2D([0], [0], color='b', label='$F_3 = tc/2\pi R^2$'),
    Line2D([0], [0], color='m', label='$He_0 / M_r$'),
]
leg2 = ax.legend(
    handles=param_handles,
    loc='upper left',          # keeps it inside automatically
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
# plt.show()
plt.savefig('./Figures/blade_params_ALL.pdf')