import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from Constants.helpers import read_force_file, plot_3D_directivity, plot_3D_phase_directivity, p_to_SPL, spl_from_autopower, plot_BPF_peaks
from Constants.data_assim import getGojonData

# BEGINNING OF HEADER
FILE='TOTAL'
MODE = 'half'
SUFFIX = '_D180_MR'
shape='D'
RPM = 8000

# SUFFIX = 'PARROT_D20L20_D180_NQ160'
# shape = 'PARROT'
# RPM = -7250
ms = np.arange(1,11,1)


for (ind_theta, ind_phi) in zip([6, 10, 6, 2], [9, 9, 0, 18]):


    # ind_theta = 2
    # ind_phi = 18

    # List of variable names used when saving


    datadir = './Experimental/dataverse_files'

    data, BPF, freq, x_cart_data, theta_data, phi_data, theta_exp, phi_exp, casefile = getGojonData(datadir, 0.02, 0.02, shape=shape, B=2, 
                                                                                                    RPM=RPM
                                                                                                    )
    data = data[:, ind_theta, ind_phi]
    x_cart = x_cart_data[:, ind_theta, ind_phi].reshape((3, 1))
    theta = theta_data[ind_theta]
    phi = phi_data[ind_phi]

    print(theta, phi)

    # RIGHT PLOT: totals + exp
    fig, ax = plt.subplots(figsize=(6, 4))

    # --- plotting ---
    # ax.plot(ms, SPL_total_PIN, color='r', marker='^')

    # ax.plot(ms, SPL_total_scattering, color='b', marker='s', linestyle='--')


    ax.plot(freq[0]/BPF,
            spl_from_autopower(data),
            color='0.3',
            linewidth=2)

    fig, ax = plot_BPF_peaks(fig, ax, freq[0] / BPF, spl_from_autopower(data), N0=1, N1= 25, range=0.01, 
                            plot_kwargs={
                                'color':'k',
                                'linestyle':'dashed',
                                'alpha':1.0,
                                'linewidth': 2
                            })

    model_handles = [
        Line2D([0], [0], color='k', marker='^', linestyle=':',
            label='PIN'),
        Line2D([0], [0], color='k', marker='s', linestyle='--',
            label='Scattering'),
        Line2D([0], [0], color='0.3', lw=3,
            label='Experiment'),
    ]
    component_handles = [
        Line2D([0], [0], color='r', lw=2, label='Rotor Steady Loading Noise'),
        Line2D([0], [0], color='b', lw=2, label='Rotor Thickness Noise'),
        Line2D([0], [0], color='g', lw=2, label='Rotor Unsteady Loading Noise'),
        Line2D([0], [0], color='m', lw=2, label='Strut-Scattered Noise'),
        Line2D([0], [0], color='k', lw=2, label='Model Total'),
        # Line2D([0], [0], color='c', lw=2, label='Non-linear'),
        # Line2D([0], [0], color='k', lw=2, label='Total'),

        # Line2D([0], [0], color='c', lw=2, label='Beam Noise due to Thickness'),
        # Line2D([0], [0], color='k', lw=2, label='L+T'),
    ]

    leg2 = ax.legend(handles=model_handles,
                    #  title='Model',
                    loc='upper right' if ind_phi != 0 else 'lower right', fontsize=10)
    leg1 = ax.legend(handles=component_handles,
                    #  title='Model',
                    loc='lower right' if ind_phi != 0 else 'upper right', fontsize=10)
    ax.add_artist(leg1)
    ax.add_artist(leg2)

    ax.set_xticks(ms)


    # ax.legend(ncol=2, loc='upper left', fontsize=10)
    ax.set_xlabel("$m = f/B/\Omega$ (Hz)")
    ax.set_ylabel("SPL (dB) w.r.t. 20e-6 Pa")
    ax.set_xscale('log')

    ax.grid(visible=True, which='major', color='k', linestyle='-')
    ax.grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.5)
    # ax.set_title(f'Theta = {theta} deg, Phi = {phi} deg')
    # plt.xlim(0.03333, 100)
    # plt.xlim(0.1, 100)
    # plt.xlim(0, 11)
    # plt.xlim(0.8, 14)
    plt.xlim(0.8, 50)


    plt.ylim(0, 70)

    plt.tight_layout()
    plt.show()
    import os
    folder_name = f'./Figures/Spectra'
    fig.savefig(
        os.path.join(folder_name, f"spectrum_EXP_ONLY_{ind_theta}_{ind_phi}{SUFFIX}.pdf"),
        dpi=300,
        bbox_inches="tight",
    )