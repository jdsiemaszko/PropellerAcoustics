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
ms = np.arange(1,11,1)

for (ind_theta, ind_phi) in zip([6, 10, 6, 2], [9, 9, 0, 18]):


    # ind_theta = 2
    # ind_phi = 18

    # List of variable names used when saving
    variable_names = [
        'p_direct_s',
        'p_direct_us',
        'p_scattered_s',
        'p_scattered_us',
        'p_direct_thickness',
        'p_scattered_thickness',
        'p_scattered_s_nc', 'p_scattered_us_nc', 'p_scattered_thickness_nc',
        'ptmB_model_rotor',
        'pLSmB_model_rotor',
        'pLUSmB_model_rotor',
        'pmB_model_beam_thickness',
        'pmB_model_beam_loading',
        'pmB_model_beam_nonlinear',
        'pmB_model_beam_total'
    ]

    # Load each array
    for name in variable_names:
        filename = f'./Data/current/PRESSURE/{name}_{MODE}_{ind_theta}_{ind_phi}_{FILE}{SUFFIX}.npy'
        globals()[name] = np.load(filename)


    datadir = './Experimental/dataverse_files'

    data, BPF, freq, x_cart_data, theta_data, phi_data, theta_exp, phi_exp, casefile = getGojonData(datadir, 0.02, 0.02, shape=shape, B=2, 
                                                                                                    RPM=8000
                                                                                                    )
    data = data[:, ind_theta, ind_phi]
    x_cart = x_cart_data[:, ind_theta, ind_phi].reshape((3, 1))
    theta = theta_data[ind_theta]
    phi = phi_data[ind_phi]

    print(theta, phi)


    # CENTER PLOT: STRUT ONLY
    fig, ax = plt.subplots(figsize=(4, 3))

    # ax.plot(ms, SPL_rotor_S, label=f"Steady Loading Noise (PIN)", color='r', marker='^')
    # ax.plot(ms, SPL_rotor_US, label=f"Unsteady Loading Noise (PIN)", color='g', marker='^')

    # ax.plot(ms, SPL_rotor_S, label=f"Steady Loading Noise (PIN)", color='r', marker='^')
    # ax.plot(ms, SPL_rotor_US, label=f"Unsteady Loading Noise (PIN)", color='g', marker='^')

    # ax.plot(ms, SPL_rotor_thickness, label=f"Thickness Noise (PIN)", color='b', marker='^')
    # ax.plot(ms, SPL_rotor_total, label=f"Rotor Total (PIN)", color='y', marker='^')
    ax.plot(ms, p_to_SPL(pmB_model_beam_loading), label=f"Beam Loading due to Blade Loading", color='r', marker='^', linestyle=':')
    ax.plot(ms, p_to_SPL(pmB_model_beam_thickness), label=f"Beam Loading due to Blade Thickness", color='b', marker='^', linestyle=':')
    ax.plot(ms, p_to_SPL(pmB_model_beam_nonlinear), label=f"Non-linear", color='c', marker='^', linestyle=':')
    ax.plot(ms, p_to_SPL(pmB_model_beam_total), label=f"Strut Total", color='k', marker='^', linestyle=':')


    # ax.plot(ms, SPL_total_PIN, label=f"Total (PIN)", color='k', marker='^')

    # ax.plot(freq[0] / BPF, spl_from_autopower(data), label=f"Experimental, total", color='k', alpha=0.75)
    # fig, ax = plot_BPF_peaks(fig, ax, freq[0] / BPF, spl_from_autopower(data), N0=1, N1= 25, range=0.01, 
    #                          plot_kwargs={
    #                              'color':'k',
    #                              'linestyle':'dashed',
    #                              'alpha':0.75
    #                          })

    ax.plot(ms, p_to_SPL(p_scattered_s_nc), label=f"Scattered Steady Loading Noise", color='r', marker='*', linestyle='dashed')
    ax.plot(ms, p_to_SPL(p_scattered_us_nc), label=f"Scattered Unsteady Loading Noise", color='m', marker='*', linestyle='dashed')
    # ax.plot(ms, SPL_scattered, label=f"Scattered Loading Noise", color='m', marker='s', linestyle='dashed')
    ax.plot(ms, p_to_SPL(p_scattered_thickness_nc), label=f"Scattered Thickness Noise", color='b', marker='*', linestyle='dashed')
    ax.plot(ms, p_to_SPL(p_scattered_thickness_nc+p_scattered_s_nc+p_scattered_us_nc), label=f"Total (Scattering)", color='k', marker='*', linestyle='dashed')


    # --- plotting ---
    # ax.plot(ms, SPL_total_PIN, color='r', marker='^')

    # ax.plot(ms, SPL_total_scattering, color='b', marker='s', linestyle='--')


    model_handles = [
        Line2D([0], [0], color='k', marker='^', linestyle=':',
            label='PIN'),
        Line2D([0], [0], color='k', marker='s', linestyle='--',
            label='Scattering'),
        # Line2D([0], [0], color='0.3', lw=3,
        #     label='Experiment'),
    ]
    component_handles = [
        Line2D([0], [0], color='r', lw=2, label='Steady Loading'),
        Line2D([0], [0], color='b', lw=2, label='Thickness'),
        Line2D([0], [0], color='m', lw=2, label='Unsteady Loading'),
        Line2D([0], [0], color='c', lw=2, label='Non-linear'),
        Line2D([0], [0], color='k', lw=2, label='Total'),

        # Line2D([0], [0], color='c', lw=2, label='Beam Noise due to Thickness'),
        # Line2D([0], [0], color='k', lw=2, label='L+T'),
    ]

    leg2 = ax.legend(handles=model_handles,
                    #  title='Model',
                    loc='lower left' if ind_phi != 0 else 'upper left', fontsize=10)
    leg1 = ax.legend(handles=component_handles,
                    #  title='Model',
                    loc='lower right' if ind_phi != 0 else 'lower left', fontsize=10)
    ax.add_artist(leg1)
    ax.add_artist(leg2)

    ax.set_xticks(ms)


    # ax.legend(ncol=2, loc='upper left', fontsize=10)
    ax.set_xlabel("$m = f/B/\Omega$ (Hz)")
    ax.set_ylabel("SPL (dB)")
    ax.set_xscale('log')

    ax.grid(visible=True, which='major', color='k', linestyle='-')
    ax.grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.5)
    # ax.set_title(f'Theta = {theta} deg, Phi = {phi} deg')
    # plt.xlim(0.03333, 100)
    # plt.xlim(0.1, 100)
    plt.xlim(0.8, 14)


    plt.ylim(20, 70)
    # plt.ylim(15, 65)

    plt.tight_layout()
    # plt.show()
    import os
    folder_name = f'./Figures/Spectra'
    fig.savefig(
        os.path.join(folder_name, f"spectrum_TAXONOMY_CENTER_{ind_theta}_{ind_phi}{SUFFIX}.pdf"),
        dpi=300,
        bbox_inches="tight",
    )

    # LEFT PLOT: rotor ONLY
    fig, ax = plt.subplots(figsize=(4, 3))

    ax.plot(ms, p_to_SPL(p_direct_s),  color='r', marker='s', linestyle='dashed')
    ax.plot(ms, p_to_SPL(p_direct_us), color='g', marker='s', linestyle='dashed')
    ax.plot(ms, p_to_SPL(p_direct_thickness),  color='b', marker='s', linestyle='dashed')

    component_handles = [
        Line2D([0], [0], color='r', lw=2, label='Direct Rotor Steady Loading Noise'),
        Line2D([0], [0], color='b', lw=2, label='Direct Rotor Thickness Noise'),
        Line2D([0], [0], color='g', lw=2, label='Direct Rotor Steady Loading Noise'),

        # Line2D([0], [0], color='c', lw=2, label='Beam Noise due to Thickness'),
        # Line2D([0], [0], color='k', lw=2, label='L+T'),
    ]

    leg1 = ax.legend(handles=component_handles,
                    #  title='Model',
                    loc='upper right', fontsize=10)
    ax.add_artist(leg1)

    ax.set_xticks(ms)


    # ax.legend(ncol=2, loc='upper left', fontsize=10)
    ax.set_xlabel("$m = f/B/\Omega$ (Hz)")
    ax.set_ylabel("SPL (dB)")
    ax.set_xscale('log')

    ax.grid(visible=True, which='major', color='k', linestyle='-')
    ax.grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.5)
    # ax.set_title(f'Theta = {theta} deg, Phi = {phi} deg')
    # plt.xlim(0.03333, 100)
    # plt.xlim(0.1, 100)
    plt.xlim(0.8, 14)


    plt.ylim(20, 70)

    # plt.ylim(15, 65)

    plt.tight_layout()
    # plt.show()
    import os
    folder_name = f'./Figures/Spectra'
    fig.savefig(
        os.path.join(folder_name, f"spectrum_TAXONOMY_LEFT_{ind_theta}_{ind_phi}{SUFFIX}.pdf"),
        dpi=300,
        bbox_inches="tight",
    )

    # RIGHT PLOT: totals + exp
    fig, ax = plt.subplots(figsize=(6, 4))

    # ax.plot(ms, p_to_SPL(p_direct_s+p_direct_us+p_direct_thickness), label=f"Rotor Total", color='r', marker='s')
    ax.plot(ms, p_to_SPL(p_direct_s),  color='r', marker='s', linestyle='dashed')
    ax.plot(ms, p_to_SPL(p_direct_us), color='g', marker='s', linestyle='dashed')
    ax.plot(ms, p_to_SPL(p_direct_thickness),  color='b', marker='s', linestyle='dashed')

    ax.plot(ms, p_to_SPL(pLSmB_model_rotor),  color='r', marker='^', linestyle=':')
    ax.plot(ms, p_to_SPL(pLUSmB_model_rotor), color='g', marker='^', linestyle=':')
    ax.plot(ms, p_to_SPL(ptmB_model_rotor),  color='b', marker='^', linestyle=':')
    ax.plot(ms, p_to_SPL(pmB_model_beam_total), label=f"Strut Total (PIN)", color='m', marker='^', linestyle=':')
    ax.plot(ms, p_to_SPL(pmB_model_beam_total+p_direct_s+p_direct_us+p_direct_thickness), label=f"Total (PIN)", color='k', marker='^', linestyle=':')

    ax.plot(ms, p_to_SPL(p_scattered_thickness_nc+p_scattered_s_nc+p_scattered_us_nc), label=f"Strut Total (Scattering)", color='m', marker='s', linestyle='dashed')
    ax.plot(ms, p_to_SPL(p_scattered_thickness_nc+p_scattered_s_nc+p_scattered_us_nc+p_direct_s+p_direct_us+p_direct_thickness), label=f"Total (Scattering)", color='k', marker='s', linestyle='dashed')


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
        os.path.join(folder_name, f"spectrum_TAXONOMY_RIGHT_{ind_theta}_{ind_phi}{SUFFIX}.pdf"),
        dpi=300,
        bbox_inches="tight",
    )