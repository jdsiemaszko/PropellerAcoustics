import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

source = './Data/current/surface_pressure/'
Fms_NC = np.load(source + 'loading_harmonics_ALL.npy') # 3, Nm, Nrq, Ncomp

Fz_iLES = np.load(source + 'Fzfreq_iLES.npy')[:, :] # Nfreq, Nrq
Fphi_iLES = np.load(source + 'Fphifreq_iLES.npy')[:, :] # Nfreq, Nrq
freqs_iLES = np.load(source + 'freqs_iLES.npy') # Nfreq

rs = np.load(source + 'r_query.npy') # Nrquery
ms = np.load(source + 'm_query.npy') #Nm
B = 2
ks = ms * B
Omega = 8000/60 * 2 * np.pi

# indices of frequencies matching ks
idx = np.array([
    np.argmin(np.abs(freqs_iLES / Omega * 2 * np.pi - k))
    for k in ks
])

# optional sanity check
if not np.allclose(freqs_iLES[idx]/ Omega * 2 * np.pi, ks):
    print("Warning: some ks do not exactly exist in freqs_iLES")

Fz_iLES = Fz_iLES[idx, :]
Fphi_iLES = Fphi_iLES[idx, :]
freqs_iLES = freqs_iLES[idx]


RTIP = 0.1
RROOT = 0.016


data_vella = np.zeros((3, 10))
for ind, st in enumerate(['0.5', '0.8', '0.9']):
    buffer = np.loadtxt(f'./Data/Vella2026/Fz_beam_r{st}.csv', delimiter=',', skiprows=1)

    data_vella[ind, :] = buffer[:, 1]

# D = 0.2
# rT = D / 2
# rR = 0.017
# r0 = np.linspace(rR, rT, 30)
# dr = np.abs(rT - rR) / len(r0)
dr = (RTIP - RROOT) / 20
data_vella /= dr

components = [
    # ==========================================================
    # 1) Loading contributions
    # ==========================================================
    {
        "name": "loading_scattering",
        "title": "Loading contribution",
        'color' : 'r',
        'linestyle': 'dashed',
        'marker' : 's',
        # 'label' : ''
    },

    # ==========================================================
    # 2) Thickness contributions
    # ==========================================================
    {
        "name": "thickness_scattering",
        "title": "Thickness contribution",
        'color' : 'b',
        'linestyle': 'dashed',
        'marker' : 's',
    },
    {
        "name": "total_scattering",
        'color' : 'r',
        'linestyle': 'dashed',
        'marker' : 's',
    },

            # ==========================================================
    # 1) Loading contributions
    # ==========================================================
    {
        "name": "loading_scattering_cp",
        'color' : 'r',
        'linestyle': 'dotted',
        'marker' : 's',
        # 'label' : ''
    },

    # ==========================================================
    # 2) Thickness contributions
    # ==========================================================
    {
        "name": "thickness_scattering_cp",
        'color' : 'b',
        'linestyle': 'dotted',
        'marker' : 's',
    },

    {
        "name": "total_scattering_cp",
        'color' : 'g',
        'linestyle': 'dashed',
        'marker' : 's',
    },
        {
        "name": "loading_PIN_total",
    'color' : 'm',
        'linestyle': 'solid',
        'marker' : '^',
    },        {
        "name": "thickness_PIN",
                'color' : 'b',
        'linestyle': 'dotted',
        'marker' : '^',
    },
    {
        "name": "nonlinear_PIN",
            'color' : 'm',
        'linestyle': 'solid',
        'marker' : '^',
    },
    {
        "name": "tota_PIN",
                'color' : 'c',
        'linestyle': 'dotted',
        'marker' : '^',
    },

        {
        "name": "total_plus_nolinear_PIN",
        'color' : 'b',
        'linestyle': 'dotted',
        'marker' : '^',
    },



]

for index_r, r in enumerate(rs):
    for index_dir, F_iLES, DIRECTION in zip([1, 2], [Fz_iLES, Fphi_iLES], ['Z', 'PHI']):

        R_RT = r / RTIP

        fig, ax = plt.subplots(figsize=(4, 3))

        for index_comp, comp in enumerate(components):
            if 'total' not in comp['name']:
                continue
            ax.plot(ks, abs(Fms_NC[index_dir, :, index_r, index_comp]), label=comp['name'], color=comp['color'], marker=comp['marker'], linestyle=comp['linestyle'])

            # if comp['marker'] == 's':
            #     ax.plot(ks, abs(Fms_NC[1, :, index_r, index_comp]), label=comp['name'], color='g', marker='s', linestyle='--')
    
        # ax.plot(ks, data_vella[index_r, :], color='k', linewidth=2, marker='o')
        ax.plot(ks, abs(F_iLES[:, index_r]), color='k', linewidth=2, marker='o')


        from matplotlib.lines import Line2D
        # component_handles = [
        #     Line2D([0], [0], color='r', lw=2, label='L'),
        #     # Line2D([0], [0], color='g', lw=2, label='Unsteady Loading'),
        #     Line2D([0], [0], color='b', lw=2, label='T'),
        #     # Line2D([0], [0], color='y', lw=2, label='Rotor Total'),
        #     Line2D([0], [0], color='m', lw=2, label='NL'),
        #     Line2D([0], [0], color='c', lw=2, label='L+T+NL'),

        #     # Line2D([0], [0], color='c', lw=2, label='Beam Noise due to Thickness'),
        #     Line2D([0], [0], color='k', lw=2, label='L+T'),
        # ]

        model_handles = [
        Line2D([0], [0], color='b',  linestyle=':', marker='^',
            label='PIN (current)'),
        Line2D([0], [0], color='m',  linestyle=':', marker='^',
            label='PIN (Vella et al. 2026)'),
        # Line2D([0], [0], color='m',  linestyle=(0, (1, 1)), marker='^',
        #        label='PIN (incl. nonlinear)'),
        Line2D([0], [0], color='r',  linestyle='--', marker='s',
            label='SM (compact)'),
            Line2D([0], [0], color='g',  linestyle='--', marker='s',
            label='SM (non-compact)'),
        Line2D([0], [0], color='k',  linestyle='-', marker='o',
            label='iLES'),
        ]

        # leg = ax.legend(handles=component_handles, loc='upper right', fontsize=8)
        leg2 = ax.legend(handles=model_handles,
                    #  title='Model',
                    loc='upper right', fontsize=8)
        # ax.add_artist(leg)
        ax.add_artist(leg2)

        ax.set_xlabel(fr'$k = f / \Omega$')
        ax.set_ylabel(fr'$|\hat{{F}}_k| [N/m]$')
        plt.tight_layout()
        ax.set_xticks(ks)

        ax.grid()
        plt.savefig(f'./Figures/iLES/harmonics_beam_{DIRECTION}_{r:.4f}.pdf')

        plt.show()