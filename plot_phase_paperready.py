
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from Constants.data_assim import getGojonData, getHarmonicsFromData
from PotentialInteraction.PIN import PotentialInteraction
from Constants.helpers import read_force_file, plot_3D_directivity, plot_3D_phase_directivity, p_to_SPL, spl_from_autopower, plot_complex_curve
MODE = 'half'
FILE = 'COMP_PHASE'
SUFFIX = '_D180_MR'
folder = './Data/current/phase_curves/'
m=4
phi_plot = 180

Pxx = np.load(folder + f'pxx_exp_{m}_{phi_plot}_{MODE}{SUFFIX}.npy')
phase = np.load(folder + f'phase_exp_{m}_{phi_plot}_{MODE}{SUFFIX}.npy')
p_total_scattering = np.load(folder + f'p_total_scattering_{m}_{phi_plot}_{MODE}{SUFFIX}.npy')
p_total_pin = np.load(folder + f'p_total_pin_{m}_{phi_plot}_{MODE}{SUFFIX}.npy')
p_total_pin_loading = np.load(folder + f'p_total_pin_loading_{m}_{phi_plot}_{MODE}{SUFFIX}.npy')
p_direct_total = np.load(folder + f'p_total_direct_{m}_{phi_plot}_{MODE}{SUFFIX}.npy')
theta = np.load(folder + f'theta_{m}_{phi_plot}_{MODE}{SUFFIX}.npy')

fig, ax = plt.subplots(figsize=(4, 3))
mic_index = np.arange(len(theta))+1
for dataset, label, color, marker in zip(
    [Pxx, p_total_scattering, p_total_pin, p_total_pin_loading, p_direct_total],
    ['Experiment', 'Scattering', 'PIN (current)', 'PIN (Vella et al. 2026)', 'Direct Rotor Radiation Only'],
    ['k', 'b', 'r', 'g', 'm'],
    ['o', 's', '^', '*', 'p']
):
    if label == 'Experiment':
        spl = spl_from_autopower(dataset)
        phase_data = phase
    else:
        spl = p_to_SPL(dataset[:, 0])
        phase_data = np.angle(dataset[:, 0] * np.exp(-1j * np.angle(dataset[0, 0]))) # angle w.r.t x_cart[0] - i.e., the first microphone

    # ax.plot(mic_index, spl, label=label, marker='s')
    ax.plot(mic_index, phase_data, label=label, marker=marker, color=color)

    # ax.set_xlabel('Theta (degrees)')
    ax.set_xlabel('Microphone Index [-]')
    # ax.set_ylabel('SPL (dB w.r.t. 20e-6 Pa)')
    ax.set_ylabel('Phase w.r.t. Mic. 1 [rad]')

ax.grid()
ax.legend()
ax.set_xticks(mic_index)

plt.tight_layout()
plt.show()