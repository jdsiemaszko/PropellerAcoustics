import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

source = './Data/current/surface_pressure/'
Fms = np.load(source + 'loading_harmonics.npy') # 3, Nm, Nr, Ncomp
rs = np.load(source + 'r_query.npy') # Nrquery
ms = np.load(source + 'm_query.npy') #Nm
RTIP = 0.1
RROOT = 0.016
B = 2
ks = ms * B

data_vella = np.zeros((3, 10))
for ind, st in enumerate(['0.5', '0.8', '0.9']):
    buffer = np.loadtxt(f'./Data/Vella2026/Fz_beam_r{st}.csv', delimiter=',', skiprows=1)

    data_vella[ind, :] = buffer[:, 1]

# D = 0.2
# rT = D / 2
# rR = 0.017
# r0 = np.linspace(rR, rT, 30)
# dr = np.abs(rT - rR) / len(r0)
dr = (RTIP - RROOT) / 40
data_vella /= dr

components = [
    {
        "name": "loading_scattering",
        "title": "Loading contribution",
        'color' : 'r',
        'linestyle': 'dashed',
        'marker' : 's',
        # 'label' : ''
    },
    {
        "name": "thickness_scattering",
        "title": "Thickness contribution",
        'color' : 'b',
        'linestyle': 'dashed',
        'marker' : 's',
    },

    {
        "name": "total_scattering",
        "title": "Total model",
        'color' : 'k',
        'linestyle': 'dashed',
        'marker' : 's',
    },

        {
        "name": "loading_PIN",
    'color' : 'r',
        'linestyle': ':',
        'marker' : '^',
    },        {
        "name": "thickness_PIN",
                'color' : 'b',
        'linestyle': ':',
        'marker' : '^',
    },
    {
        "name": "nonlinear_PIN",
            'color' : 'm',
        'linestyle': ':',
        'marker' : '^',
    },
    {
        "name": "total_PIN",
                'color' : 'k',
        'linestyle': ':',
        'marker' : '^',
    },

        {
        "name": "total_plus_nolinear_PIN",
        'color' : 'c',
        'linestyle': ':',
        'marker' : '^',
    },
]

for index_r, r in enumerate(rs):

    R_RT = r / RTIP

    fig, ax = plt.subplots(figsize=(4, 3))

    for index_comp, comp in enumerate(components):

       ax.plot(ks, abs(Fms[1, :, index_r, index_comp]), label=comp['name'], color=comp['color'], marker=comp['marker'], linestyle=comp['linestyle'])
 
    ax.plot(ks, data_vella[index_r, :], color='k', linewidth=2, marker='o')


    from matplotlib.lines import Line2D
    component_handles = [
        Line2D([0], [0], color='r', lw=2, label='L'),
        # Line2D([0], [0], color='g', lw=2, label='Unsteady Loading'),
        Line2D([0], [0], color='b', lw=2, label='T'),
        # Line2D([0], [0], color='y', lw=2, label='Rotor Total'),
        Line2D([0], [0], color='m', lw=2, label='NL'),
        Line2D([0], [0], color='c', lw=2, label='L+T+NL'),

        # Line2D([0], [0], color='c', lw=2, label='Beam Noise due to Thickness'),
        Line2D([0], [0], color='k', lw=2, label='L+T'),
    ]

    model_handles = [
    Line2D([0], [0], color='k',  linestyle=':', marker='^',
           label='PIN'),
    Line2D([0], [0], color='k',  linestyle='--', marker='s',
           label='SM'),
    Line2D([0], [0], color='k',  linestyle='-', marker='o',
           label='iLES'),
    ]

    leg = ax.legend(handles=component_handles, loc='upper right', fontsize=8)
    leg2 = ax.legend(handles=model_handles,
                #  title='Model',
                 loc='upper center', fontsize=8)
    ax.add_artist(leg)
    ax.add_artist(leg2)

    ax.set_xlabel(fr'$k = f / \Omega$')
    ax.set_ylabel(fr'$|\hat{{F}}_k| [N/m]$')
    plt.tight_layout()

    ax.grid()
    plt.savefig(f'./Figures/iLES/harmonics_beam_{r:.4f}.pdf')

    plt.show()