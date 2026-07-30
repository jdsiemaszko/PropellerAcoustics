import numpy as np
import matplotlib.pyplot as plt

MODE = 'half'
SUFFIX = '_D180_MR'
folder = './Data/current/phase_curves/'
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
m_values = [1, 2, 3, 4, 5]
phi_values = [50, 90, 130]

labels = [
    'Scattering',
    'PIN (current)',
    'PIN (Vella et al. 2026)',
    # 'Direct Rotor Radiation'
    # 'Direct Only'
]

markers = ['s', '^', '*', 'p']
colors = ['b', 'r', 'g', 'm']


for m in m_values:

    for phi_plot in phi_values:

        fig, ax = plt.subplots(figsize=(4, 2.75))

        # Load data
        Pxx = np.load(
            folder + f'pxx_exp_{m}_{phi_plot}_{MODE}{SUFFIX}.npy'
        )
        phase = np.load(
            folder + f'phase_exp_{m}_{phi_plot}_{MODE}{SUFFIX}.npy'
        )

        p_total_scattering = np.load(
            folder + f'p_total_scattering_{m}_{phi_plot}_{MODE}{SUFFIX}.npy'
        )
        p_total_pin = np.load(
            folder + f'p_total_pin_{m}_{phi_plot}_{MODE}{SUFFIX}.npy'
        )
        p_total_pin_loading = np.load(
            folder + f'p_total_pin_loading_{m}_{phi_plot}_{MODE}{SUFFIX}.npy'
        )
        p_direct_total = np.load(
            folder + f'p_total_direct_{m}_{phi_plot}_{MODE}{SUFFIX}.npy'
        )

        theta = np.load(
            folder + f'theta_{m}_{phi_plot}_{MODE}{SUFFIX}.npy'
        )

        mic_index = np.arange(len(theta)) + 1


        # shitty fix
        # if phase[-1] > 0 and phase[-2] < -np.pi/2:
        #     phase[-1] -= 2*np.pi
        phase = np.unwrap(phase)

        # Experiment
        ax.plot(
            mic_index,
            phase,
            marker='o',
            color='k',
            label='Experiment'
        )


        # Models
        for dataset, label, marker, color in zip(
            [
                p_total_scattering,
                p_total_pin,
                p_total_pin_loading,
                p_direct_total
            ],
            labels,
            markers,
            colors
        ):

            phase_data = np.angle(
                dataset[:, 0] *
                np.exp(-1j*np.angle(dataset[0, 0]))
            )

            # remove +/- pi jumps
            phase_data = np.unwrap(phase_data)



            ax.plot(
                mic_index,
                phase_data,
                marker=marker,
                linestyle='-',
                color=color,
                alpha=0.8,
                label=label
            )


        # ax.set_title(rf'$m={m},\ \phi={phi_plot}^\circ$')

        ax.set_xticks(mic_index)


        # Phase ticks modulo 2*pi
        # yticks = np.linspace(0, 2*np.pi, 5)
        yticks = np.linspace(-np.pi, np.pi, 5)


        # yticklabels = [
        #     r'$0$',
        #     r'$\pi/2$',
        #     r'$\pi$',
        #     r'$3\pi/2$',
        #     r'$2\pi$'
        # ]
        yticklabels = [
            r'$-\pi$',
            r'$-\pi/2$',
            r'$0$',
            r'$\pi/2$',
            r'$\pi$'
        ]


        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

        # Minor ticks (e.g. every π/4)
        ax.set_yticks(np.linspace(-np.pi, np.pi, 17), minor=True)

        # Grid
        ax.grid(which='major', axis='both', linestyle='-')
        ax.grid(which='minor', axis='y', linestyle='--', alpha=0.5)

        ax.set_ylim(-np.pi * 1.2, np.pi * 1.2)
        ax.set_xlim(0, 14)
        # ax.minorticks_on()
        # ax.grid(which='major', linestyle='-')
        # ax.grid(which='minor', linestyle='--', alpha=0.5)


        # Only show legend for reference case
        if m == 1 and phi_plot == 50:
            ax.legend(
                fontsize=10,
                ncol=1, loc='lower right'
            )

        if m == 5:
            ax.set_xlabel('Mic. index')
        if phi_plot==50: 
            ax.set_ylabel('Phase w.r.t. mic. 1. [rad]')

        fig.tight_layout()
        plt.show()
        fig.savefig(f'./Figures/phase_curves/phase_plot_{SUFFIX}_M{m}_phi{phi_plot}.pdf')