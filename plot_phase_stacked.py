import numpy as np
import matplotlib.pyplot as plt

MODE = 'half'
SUFFIX = '_D180_MR'
folder = './Data/current/phase_curves/'

m_values = [1, 2, 3, 4, 5]
phi_values = [50, 90, 130]

datasets = [
    'p_total_scattering',
    'p_total_pin',
    'p_total_pin_loading',
    'p_total_direct'
]

labels = [
    'Scattering',
    'PIN (current)',
    'PIN (Vella et al. 2026)',
    'Direct Rotor Radiation Only'
]

markers = ['s', '^', '*', 'p']
colors = ['b', 'r', 'g', 'm']

m_colors = plt.cm.tab10(np.arange(len(m_values)))


for phi_plot in phi_values:

    fig, ax = plt.subplots(figsize=(4, 8))

    for m, m_color in zip(m_values, m_colors):

        # Load data
        Pxx = np.load(folder + f'pxx_exp_{m}_{phi_plot}_{MODE}{SUFFIX}.npy')
        phase = np.load(folder + f'phase_exp_{m}_{phi_plot}_{MODE}{SUFFIX}.npy')

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

        # phase offset for harmonic number
        offset = 2*np.pi*(m-1)


        # Plot experiment
        exp_phase = phase
        ax.plot(
            mic_index,
            exp_phase + offset,
            marker='o',
            # linestyle='none',
            # color=m_color,
            color='k',
            label=f'Exp. m={m}'
        )


        # Plot all model components
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

            phase_data = np.unwrap(phase_data)

            ax.plot(
                mic_index,
                phase_data + offset,
                marker=marker,
                linestyle='-',
                color=color,
                alpha=0.8,
                label=f'{label}, m={m}'
            )


        # m annotation
        ax.text(
            mic_index[-1] + 0.3,
            offset,
            rf'$m={m}$',
            color=m_color,
            fontsize=10,
            va='center'
        )


    ax.set_xlabel('Microphone Index [-]')
    ax.set_ylabel('Phase w.r.t. Mic. 1 + $2\pi(m-1)$ [rad]')

    ax.set_xticks(np.arange(1, len(mic_index)+1))

    # phase ticks repeated modulo 2*pi for each harmonic
    n_periods = len(m_values)

    base_ticks = np.linspace(0, 2*np.pi, 5)
    yticks = np.concatenate(
        [base_ticks + 2*np.pi*i for i in range(n_periods)]
    )

    yticklabels = [
        r'$0$',
        r'$\pi/2$',
        r'$\pi$',
        r'$3\pi/2$',
        r'$2\pi$'
    ]

    ylabels = yticklabels * n_periods

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)

    ax.grid()

    # # avoid huge legend duplication
    # handles, labels_legend = ax.get_legend_handles_labels()
    # by_label = dict(zip(labels_legend, handles))
    # ax.legend(
    #     by_label.values(),
    #     by_label.keys(),
    #     fontsize=8,
    #     ncol=2
    # )

    plt.tight_layout()
    plt.show()
