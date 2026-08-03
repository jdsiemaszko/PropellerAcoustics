import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from Constants.helpers import read_force_file, plot_3D_directivity, plot_3D_phase_directivity, p_to_SPL, spl_from_autopower, plot_BPF_peaks
from Constants.data_assim import getGojonData
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# BEGINNING OF HEADER
FILE='TOTAL'
MODE = 'half'
SUFFIX = 'D20L20_D180_v2'
shape='D'
RPM = 8000

# SUFFIX = 'PARROT_D20L20_D180_NQ160'
# shape = 'PARROT'
# RPM = -7250
ms = np.arange(1,11,1)

# LBM_rotor = np.loadtxt('./Data/Vella2026/LBM_fine_rotor.csv', delimiter=',', skiprows=1).T[1]
# LBM_strut = np.loadtxt('./Data/Vella2026/LBM_fine_strut.csv', delimiter=',', skiprows=1).T[1]
# LBM_total = np.loadtxt('./Data/Vella2026/LBM_fine_total.csv', delimiter=',', skiprows=1).T[1]
LBM_rotor = np.loadtxt('./Data/Vella2026/NS_rotor.csv', delimiter=',', skiprows=1).T[1]
LBM_strut = np.loadtxt('./Data/Vella2026/NS_strut.csv', delimiter=',', skiprows=1).T[1]
mLBM = np.arange(1, len(LBM_rotor)+1, 1)


for index, (ind_theta, ind_phi, y1, y2) in enumerate(zip([6, 10, 6, 2], [9, 
                                                                        #  9, 0, 18
                                                                         ], [42, 51, None, 46], [63, 66, None, 61])):


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
                                                                                                    RPM=RPM
                                                                                                    )
    data = data[:, ind_theta, ind_phi]
    x_cart = x_cart_data[:, ind_theta, ind_phi].reshape((3, 1))
    theta = theta_data[ind_theta]
    phi = phi_data[ind_phi]

    print(theta, phi)

    # PAPER PLOT: only totals
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(ms, p_to_SPL(p_direct_s+p_direct_us+p_direct_thickness), label=f"Total (PIN)", color='r', marker='^', linestyle=':')
    ax.plot(ms, p_to_SPL(pmB_model_beam_total), label=f"Total (PIN)", color='g', marker='^', linestyle=':')

    # ax.plot(ms, p_to_SPL(pmB_model_beam_total+p_direct_thickness), label=f"Total (PIN)", color='b', marker='^', linestyle=':')

    ax.plot(ms, p_to_SPL(p_direct_s+p_direct_us+p_direct_thickness), label=f"Total (Scattering)", color='r', marker='s', linestyle='dashed')
    ax.plot(ms, p_to_SPL(p_scattered_thickness_nc+p_scattered_s_nc+p_scattered_us_nc), label=f"Total (Scattering)", color='g', marker='s', linestyle='dashed')

    # ax.plot(ms, p_to_SPL(p_scattered_thickness_nc+p_scattered_s_nc+p_scattered_us_nc+p_direct_s+p_direct_us+p_direct_thickness),
    #          label=f"Total (Scattering)", color='b', marker='s', linestyle='dashed')

    # LBM
    ax.plot(mLBM, LBM_rotor, color='r', marker='o', linestyle='dashdot')
    ax.plot(mLBM, LBM_strut, color='g', marker='o', linestyle='dashdot')
    # ax.plot(mLBM, LBM_total, color='b', marker='o', linestyle='dashdot')

    if y1 is not None and False:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

        # --- inset zoom ---
        x1, x2 = 1.01, 11      # example x-range of inset

        axins = inset_axes(ax, width="40%", height="45%", loc="upper right")

        # replot all curves in inset


        axins.plot(ms, p_to_SPL(p_direct_s+p_direct_us+p_direct_thickness),
                color='r', marker='^', linestyle=':')

        axins.plot(ms, p_to_SPL(pmB_model_beam_total),
                color='g', marker='^', linestyle=':')

        # axins.plot(ms, p_to_SPL(pmB_model_beam_total+p_direct_thickness), label=f"Total (PIN)", color='b', marker='^', linestyle=':')


        # axins.plot(ms, p_to_SPL(p_scattered_thickness_nc+p_scattered_s_nc+p_scattered_us_nc+
        #                         p_direct_s+p_direct_us+p_direct_thickness),
        #         color='b', marker='s', linestyle='dashed')
        axins.plot(ms, p_to_SPL(p_direct_s+p_direct_us+p_direct_thickness), label=f"Total (Scattering)", color='r', marker='s', linestyle='dashed')
        axins.plot(ms, p_to_SPL(p_scattered_thickness_nc+p_scattered_s_nc+p_scattered_us_nc), label=f"Total (Scattering)", color='g', marker='s', linestyle='dashed')
        # axins.plot(ms, p_to_SPL(p_scattered_thickness_nc+p_scattered_s_nc+p_scattered_us_nc+p_direct_s+p_direct_us+p_direct_thickness),
            #  label=f"Total (Scattering)", color='b', marker='s', linestyle='dashed')
        
        axins.plot(mLBM, LBM_rotor, color='r', marker='o', linestyle='dashdot')
        axins.plot(mLBM, LBM_strut, color='g', marker='o', linestyle='dashdot')
        # axins.plot(mLBM, LBM_total, color='b', marker='o', linestyle='dashdot')

        # axins.plot(freq[0]/BPF,
        #         spl_from_autopower(data),
        #         color='0.3',
        #         linewidth=2)
        

        fig, axins = plot_BPF_peaks(fig, axins, freq[0] / BPF, spl_from_autopower(data), N0=1, N1= 25, range=0.01, 
                            plot_kwargs={
                                'color':'k',
                                'linestyle':'solid',
                                'alpha':1.0,
                                'linewidth': 2
                            })

        # inset limits


        # axins.set_xscale('log')
        axins.set_xticks(ms[1::2])
        axins.set_yticks(np.arange(35, 80, 5))  
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)    

        plt.minorticks_on()
        # Grid
        axins.grid(which='major', axis='both', linestyle='-')
        axins.grid(which='minor', linestyle='--', alpha=0.5)

        # optional: draw rectangle showing zoom region
        mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")



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
                                'linestyle':'solid',
                                'alpha':1.0,
                                'linewidth': 2
                            })

    model_handles = [
        Line2D([0], [0], color='0.3', lw=3,
            label='Experiment'),
        Line2D([0], [0], color='k', marker='s', linestyle='--',
            label='Scattering'),
        Line2D([0], [0], color='k', marker='^', linestyle=':',
            label='PIN'),
        Line2D([0], [0], color='k', marker='o', linestyle='dashdot',
        #     label='PIN (Vella et al. 2026)'),
            # label='LBM',)
            label='iLES',)
    ]

    component_handles = [
        # Line2D([0], [0], color='r', lw=2, label='Rotor Steady Loading Noise'),
        # Line2D([0], [0], color='b', lw=2, label='Rotor Thickness Noise'),
        # Line2D([0], [0], color='g', lw=2, label='Rotor Unsteady Loading Noise'),
        Line2D([0], [0], color='r', lw=2, label='Rotor'),
        Line2D([0], [0], color='g', lw=2, label='Strut'),
        # Line2D([0], [0], color='b', lw=2, label='Total'),
        # Line2D([0], [0], color='c', lw=2, label='Non-linear'),
        # Line2D([0], [0], color='k', lw=2, label='Total'),

        # Line2D([0], [0], color='c', lw=2, label='Beam Noise due to Thickness'),
        # Line2D([0], [0], color='k', lw=2, label='L+T'),
    ]



    if index == 0:

        leg1 = ax.legend(handles=model_handles,
                # loc='upper center',
                # loc='lower left' if y1 is not None else 'upper right',
                loc='upper right',
                # bbox_to_anchor=(0.5, -0.18),
                ncol=1,
                fontsize=10)

        leg2 = ax.legend(handles=component_handles,
                        # loc='upper center',
                        loc='lower right',
                        # bbox_to_anchor=(0.5, -0.28),
                        ncol=1,
                        fontsize=10)
        ax.add_artist(leg1)
        ax.add_artist(leg2)



    # ax.legend(ncol=2, loc='upper left', fontsize=10)
    ax.set_xlabel(r"$m = f/B\Omega$")
    ax.set_ylabel("SPL [dB w.r.t. 20e-6 Pa]")
    ax.set_xscale('log')
    # ax.set_xticks(10.**np.arange(-1, 3, 1))
    # ax.set_xticks(10.**np.arange(-1, 3, 1), minor=True)


    plt.minorticks_on()
    ax.set_yticks(np.arange(0, 80, 10))

    ax.set_yticks(np.arange(0, 80, 2), minor=True)
    # Grid
    ax.grid(which='major', axis='both', linestyle='-')
    ax.grid(which='minor', linestyle='--', alpha=0.5)
    # ax.set_title(f'Theta = {theta} deg, Phi = {phi} deg')
    # plt.xlim(0.03333, 100)
    # plt.xlim(0.1, 100)
    # plt.xlim(0, 11)
    # plt.xlim(0.8, 14)


    # ax.set_xlim(0.8, 200 if y1 is not None else 50)
    ax.set_xlim(0.8, 50)



    ax.set_ylim(0, 75)

    # plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.tight_layout()
    plt.show()
    import os
    folder_name = f'./Figures/Spectra'
    fig.savefig(
        os.path.join(folder_name, f"spectrum_TAXONOMY_PAPER_TOTAL_vs_LBM_{ind_theta}_{ind_phi}{SUFFIX}.pdf"),
        dpi=300,
        bbox_inches="tight",
    )