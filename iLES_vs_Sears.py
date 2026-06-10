"""
Example script to parse a MATLAB .mat file with known variable names.

Supports:
- Standard MATLAB .mat files (v7 and earlier) via scipy.io.loadmat
- Basic inspection and extraction of arrays/scalars

Install dependencies:
    pip install scipy numpy
"""

from pathlib import Path
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import welch

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

MAT_FILE = "./Data/current/iLES/ISAE2_L20_D20_Loading_20_20.mat"

# -------------------------------------------------------------------
# Load .mat file
# -------------------------------------------------------------------

mat_path = Path(MAT_FILE)

if not mat_path.exists():
    raise FileNotFoundError(f"MAT file not found: {mat_path}")

data = loadmat(mat_path)

# -------------------------------------------------------------------
# Print available keys
# -------------------------------------------------------------------

print("Available keys in MAT file:")
for key in data.keys():
    if not key.startswith("__"):
        print(f"  - {key}")

"""
dtype=[('Fz', 'O'), ('Fr', 'O'), ('Ftheta', 'O'), ('dSi', 'O'), ('Fz_span', 'O'), ('Fr_span', 'O'), ('Ftheta_span', 'O'), ('Fz_area', 'O'), ('Fr_area', 'O'), ('Ftheta_area', 'O'), ('dA', 'O'), ('ri', 'O'), ('dStot', 'O'), ('Ttot', 'O'), ('dt', 'O'), ('Fs', 'O'), ('BPF', 'O'), ('Fr_fft', 'O'), ('Fz_fft', 'O'), ('Ftheta_fft', 'O')])
"""
dFzdA = data['ISAE2_L20_D20'][0, 0][7] # shape Nt, Nr, Nc
dFrdA = data['ISAE2_L20_D20'][0, 0][8]
dFphidA = data['ISAE2_L20_D20'][0, 0][9]
dA = data['ISAE2_L20_D20'][0, 0][10]

Nt, Nr, Nc = dFzdA.shape

chord = 0.025
rt = 0.1
rr = 0.15 * rt
for r_query in [0.5 * rt, 0.8 * rt, 0.9 * rt]:

    dr = (rt - rr) / Nr
    dc = chord / Nc

    r = np.arange(rr + dr/2, rt, dr) # Nr
    c = np.arange(-chord/2+dc/2, chord/2, dc) # Nc

    Fz_reconstruct = np.sum(np.mean(dFzdA, axis=0)[:, :] * dA[0, :, :], axis=-1) / dr
    Fphi_reconstruct = np.sum(np.mean(dFphidA, axis=0)[:, :] * dA[0, :, :], axis=-1) / dr


    Fz_in_time_reconstruct = np.sum(dFzdA* dA[:, :, :], axis=-1) / dr
    rot_freq = 8000 / 60 # Hz

    period = 1/rot_freq
    dt = 1/(rot_freq)/400 # 8000 RPM, 400 timesteps per rev

    time_reconstruct = np.arange(0, 2000 * dt, dt)
    BPF = 2 * rot_freq # blade passing frequency (2 blades)
    harmonics = np.arange(0, 41, 1) # 0-40 harmonics of rot_freq to plot

    def extract_bpf_harmonics(signal, dt, f_fundamental, harmonics):
        """
        Extract amplitudes of harmonics using an explicit Fourier projection
        at the requested frequencies.

        Assumes the sampled signal contains an integer number of periods of
        the fundamental frequency, so the harmonics are exactly orthogonal
        over the sampled window.

        Parameters
        ----------
        signal : ndarray
            Shape (Nt, Nr, Nc)

        dt : float
            Time step [s]

        f_fundamental : float
            Fundamental frequency [Hz]

        harmonics : array-like
            Harmonic indices to extract

        Returns
        -------
        harmonic_amplitudes : ndarray
            Shape (Nh, Nr, Nc)

        harmonic_freqs : ndarray
            Frequencies corresponding to harmonics
        """

        Nt = signal.shape[0]

        # time vector
        t = np.arange(Nt) * dt

        harmonics = np.asarray(harmonics)
        harmonic_freqs = harmonics * f_fundamental

        Nh = len(harmonics)

        harmonics = np.zeros(
            (Nh, signal.shape[1], signal.shape[2]),
            dtype=np.complex128
        )

        for i, f in enumerate(harmonic_freqs):

            # complex exponential for explicit Fourier coefficient
            basis = np.exp(2j * np.pi * f * t)

            # projection onto Fourier mode
            coeff = np.tensordot(
                basis,
                signal,
                axes=(0, 0)
            ) / Nt


            harmonics[i] = coeff

        return harmonics, harmonic_freqs

    def extract_bpf_harmonics_welch(signal, dt, f_fundamental, harmonics,
                                    nperseg=None, detrend='constant'):
        """
        Estimate harmonic/modal amplitudes using Welch PSD estimation.

        This approach computes the power spectral density (PSD) using Welch's
        method and then extracts amplitudes at the harmonic frequencies.

        Parameters
        ----------
        signal : ndarray
            Shape (Nt, Nr, Nc)

        dt : float
            Time step [s]

        f_fundamental : float
            Fundamental frequency [Hz]

        harmonics : array-like
            Harmonic indices to extract (e.g. [1,2,3,...])

        nperseg : int, optional
            Segment length for Welch. If None, scipy default is used.

        detrend : str or function, optional
            Detrending method passed to scipy.signal.welch

        Returns
        -------
        harmonic_amplitudes : ndarray
            Shape (Nh, Nr, Nc), amplitude estimated from PSD

        harmonic_freqs : ndarray
            Frequencies corresponding to harmonics
        """

        Nt = signal.shape[0]
        fs = 1.0 / dt

        harmonics = np.asarray(harmonics)
        harmonic_freqs = harmonics * f_fundamental
        Nh = len(harmonics)

        Nr, Nc = signal.shape[1], signal.shape[2]

        harmonic_amplitudes = np.zeros((Nh, Nr, Nc), dtype=np.float64)

        # loop over spatial points
        for r in range(Nr):
            for c in range(Nc):

                x = signal[:, r, c]

                # compute PSD via Welch
                f, Pxx = welch(
                    x,
                    fs=fs,
                    nperseg=nperseg,
                    detrend=detrend, scaling='spectrum'
                )

                # interpolate PSD at harmonic frequencies
                for i, fh in enumerate(harmonic_freqs):

                    # nearest frequency bin (simple + robust)
                    idx = np.argmin(np.abs(f - fh))

                    # convert PSD -> amplitude (single-sided assumption)
                    modal_energy = Pxx[idx]
                    if fh > 0:
                        modal_energy /= 2 # take two-sided spectrum!
                    harmonic_amplitudes[i, r, c] = np.sqrt(modal_energy)

        harmonic_amplitudes[0, :, :] = signal.mean(axis=0)
        return harmonic_amplitudes, harmonic_freqs

    def plot_harmonic_surface(
        harmonic_field,
        r,
        c,
        harmonic_number,
        harmonic_frequency,
        quantity_name=r"$|dF_z/dA|$"
    ):
        """
        Plot a 3D surface of one harmonic amplitude.
        """

        C, R = np.meshgrid(c, r)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(
            C,
            R,
            harmonic_field,
            cmap='viridis',
            edgecolor='none'
        )

        ax.set_xlabel('Chordwise coordinate c [m]')
        ax.set_ylabel('Radial coordinate r [m]')
        ax.set_zlabel(quantity_name)

        ax.set_title(
            f'Harmonic {harmonic_number} '
            f'({harmonic_frequency:.1f} Hz)'
        )

        cbar = fig.colorbar(surf, ax=ax, shrink=0.7)
        cbar.set_label(quantity_name)

        ax.view_init(elev=30, azim=-135)

        ax.set_xlim(-r.max()/2, r.max()/2)
        ax.set_ylim(0, r.max())

        plt.tight_layout()
        plt.show()

    def extract_chordwise_distribution(
        harmonic_field,
        r,
        c,
        r_query,
        interpolation_kind='linear'
    ):
        """
        Extract chordwise distribution at arbitrary radial location
        using interpolation in the radial direction.

        Parameters
        ----------
        harmonic_field : ndarray
            Shape (Nr, Nc)

        r : ndarray
            Radial coordinates, shape (Nr,)

        c : ndarray
            Chordwise coordinates, shape (Nc,)

        r_query : float
            Desired radial location

        interpolation_kind : str
            scipy interp1d interpolation kind

        Returns
        -------
        c : ndarray
            Chordwise coordinates

        distribution : ndarray
            Interpolated chordwise distribution, shape (Nc,)
        """

        if r_query < r.min() or r_query > r.max():
            raise ValueError(
                f"r_query={r_query:.4f} outside range "
                f"[{r.min():.4f}, {r.max():.4f}]"
            )

        # interpolate each chordwise station in radial direction
        interpolator = interp1d(
            r,
            harmonic_field,
            axis=0,
            kind=interpolation_kind
        )

        distribution = interpolator(r_query)

        return c, distribution

    def plot_chordwise_distribution(
        c,
        distribution,
        r_query,
        harmonic_number=None,
        harmonic_frequency=None,
        quantity_name=r"$|dF_z/dA|$",
        fig = None, ax = None,
        color=None
    ):
        """
        Plot chordwise distribution at a given radial station.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            c,
            distribution,
            linewidth=2,
            label=quantity_name,
            # marker='x',
            # marker='s',
            color=color
        )

        ax.set_xlabel(r'Chordwise coordinate $\overline{x} = \frac{2x}{c}$')
        # ax.set_ylabel('$\Delta p$ [$N/m^2$]')
        ax.set_ylabel('$c \Delta p / L\prime = C_p / C_l$')


        ax.grid(True)
        plt.tight_layout()
        # plt.show()
        return fig, ax

    def getBLHatRadius(BLH, r, r_query):
        """
        BLH of shape Nk, Nr
        return: BLH_at_r of shape Nk
        """
        interpolator = interp1d(
            r,
            BLH,
            axis=1,
            kind='linear'
        )
        return interpolator(r_query)

    # harmonic_amplitudes, harmonic_freqs = extract_bpf_harmonics(
    #     dFzdA,
    #     dt,
    #     # BPF,
    #     rot_freq,
    #     harmonics
    # )

    harmonic_amplitudes, harmonic_freqs = extract_bpf_harmonics_welch(
        dFzdA,
        dt,
        # BPF,
        rot_freq,
        harmonics, nperseg=400
    )

    Fz_harmonics_reconstruct = np.sum(harmonic_amplitudes * dA[0, :, :], axis=-1) / dr # shape Nk, Nr


    # ------------------------------------------------------------------
    # Optional: plot all harmonics
    # ------------------------------------------------------------------

    # for i in range(0, 11, 1):

    #     plot_harmonic_surface(
    #         harmonic_amplitudes[i],
    #         r,
    #         c,
    #         harmonic_number=harmonics[i],
    #         harmonic_frequency=harmonic_freqs[i],
    #         quantity_name=r"$|dF_z/dA|$"
    #     )


    # get the loadings
    from SourceMode.Configurations_NACA0012 import D20L20W00_D180
    from Constants.helpers import read_force_file
    r_inner, Fz, Fphi  = read_force_file('./Data/Zamponi2026/FS_ISAE_2_8000.txt') # reuse the radial stations from data

    Fz = np.interp(r_inner, r, Fz_reconstruct)
    Fphi = np.interp(r_inner, r, Fphi_reconstruct)

    PIN = D20L20W00_D180.getPIN(Fz, Fphi, D=0.02, L=0.02)

    rv = np.load('./Data/Vella2026/r.npy')
    Uinf = np.load('./Data/Vella2026/Uinf.npy')

    Uinf = np.interp(r_inner, rv, Uinf)
    Ui = np.zeros((2, len(r_inner)))
    Ui[1, :] = -abs(Uinf) # only axial flow, at Uinf downwards
    PIN.updateUi(Ui)


    BLH = PIN.getBladeLoadingHarmonics() # 3, Nk, Nr
    BL, time = PIN.getBladeLoading() # 3, Nt, Nr
    r_inner = D20L20W00_D180.seg_radius
    ks = PIN.k
    ks = ks[ks<=10]
    # ks = ks[::3]
    colors = plt.cm.viridis(np.linspace(0, 1, len(ks)))

    # choose radial location


    from matplotlib.lines import Line2D
    model_handles = [
        Line2D([0], [0], color='k', linestyle='--',
            label='Model'),
        Line2D([0], [0], color='k',  linestyle='-',
            label='iLES'),
        # Line2D([0], [0], color='0.3', lw=3,
        #        label='Experiment'),
    ]



    # plot
    fig, ax = plt.subplots(figsize=(4, 3))
    for i, (k, color) in enumerate(zip(ks, colors)):

        # extract interpolated chordwise distribution
        c_dist, harmonic_dist = extract_chordwise_distribution(
            harmonic_amplitudes[i],
            r,
            c,
            r_query
        )

        BLH_abs_reconstruct = np.abs(np.interp(r_query, r, Fz_harmonics_reconstruct[i, :]))

        fig, ax = plot_chordwise_distribution(
            c_dist / chord * 2,
            (np.abs(harmonic_dist) * dA[0, -2, :] / dr / dc) / BLH_abs_reconstruct * chord, # convert to right scale: area varies per chord, but not per radius
            r_query,
            harmonic_number=harmonics[i],
            harmonic_frequency=harmonic_freqs[i],
            quantity_name=f"{i}",
            fig=fig, ax=ax,
            color=color
        )

        # plot sears dist on the same plot
        ybar = np.linspace(-1, 1, 100)
        ybar = (ybar[1:] + ybar[:-1]) / 2 # midpoints
        f_sears = np.sqrt((1-ybar) / (1 + ybar)) / np.pi / chord * 2 * getBLHatRadius(BLH[1, :, :], r_inner, r_query)[i]
        c_sears = ybar * chord / 2
        # ax.plot(c_sears, np.abs(f_sears), color=color, linestyle='--')

        # ax.plot(ybar, np.abs(f_sears / getBLHatRadius(BLH[1, :, :], r_inner, r_query)[i]), color=color, linestyle='--')
        ax.plot(ybar, np.abs(f_sears / getBLHatRadius(BLH[1, :, :], r_inner, r_query)[i]) * chord, color='k', linestyle='--', linewidth=2)

    leg1 = ax.legend(title=f'Hamonic number', ncols=2, loc='upper right', fontsize='8')

    leg2 = ax.legend(handles=model_handles,
                    #  title='Model',
                    loc='upper left', fontsize='8')

    # ax.axvline(-chord/2, color='k', linestyle='--')
    # ax.axvline(+chord/2, color='k', linestyle='--')
    ax.axvline(-1, color='k', linestyle='-')
    ax.axvline(+1, color='k', linestyle='-')

    ax.add_artist(leg1)
    ax.add_artist(leg2)

    ax.set_xlim(-0.99, 0.99)
    ax.set_ylim(0, 5)
    # ax.set_yscale('log')
    plt.savefig(f'./Figures/iLES/chordwise_{r_query:.4f}.pdf')




    fig, ax = plt.subplots(figsize=(4, 3))

    Fz_at_r_PIN = np.array([
        np.interp(r_query, r_inner, BL[1, k, :])
        for k in range(len(time))
    ])

    Fz_at_r_iLES = np.array([
        np.interp(r_query, r, 
Fz_in_time_reconstruct[k, :])
        for k in range(len(time_reconstruct))
    ])

    Fz_at_r_iLES_periodic_mean = np.array([
        np.mean(Fz_at_r_iLES[i::400]) for i in range(400)
    ])

    shift = 1.0
    mid = 200
    time_mean_nondim = time_reconstruct[:400]/period - shift + 0.5
    time_mean = time_mean_nondim * period
    Fz_mean = np.roll(Fz_at_r_iLES_periodic_mean, -mid)
    ax.plot(time_mean_nondim, Fz_mean, color='b', label='iLES', linewidth=2)
    for i in range(0, 5):
        ax.plot(time_reconstruct/period - shift  - i, Fz_at_r_iLES, color='b', alpha=0.3)

    ax.plot(time/period, Fz_at_r_PIN, color='r', linestyle='--',  label='Model')
    ax.legend()
    # ax.set_ylim(0, 1.1 * max(np.max(abs(Fz_at_r[1:len(ks)])), np.max(abs(getBLHatRadius(BLH[1, :, :], r_inner, r_query)[1:len(ks)]))))
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 35)

    ax.grid()

    # ax.set_xticks(ks)
    ax.set_xlabel(fr'$t\Omega / 2 / \pi$')
    ax.set_ylabel(fr'${{F}} [N/m]$')
    plt.tight_layout()
    plt.savefig(f'./Figures/iLES/net_time_{r_query:.4f}.pdf')




    ks = PIN.k
    ks = ks[ks<=20]

    F_k_explicit = rot_freq * np.sum(
        Fz_mean[None, :] * np.exp(1j * ks[:, None] * 2 * np.pi * rot_freq * time_mean[None, :]) * dt 
        , axis=1
    )

    fig, ax = plt.subplots(figsize=(4, 3))

    Fz_at_r = np.array([
        np.interp(r_query, r, Fz_harmonics_reconstruct[k, :])
        for k in range(Fz_harmonics_reconstruct.shape[0])
    ])

    ax.plot(ks, abs(F_k_explicit), color='b', marker='s', label='iLES')

    # ax.plot(ks, abs(Fz_at_r[:len(ks)]), color='b', marker='s', label='iLES')
    ax.plot(ks, abs(getBLHatRadius(BLH[1, :, :], r_inner, r_query)[:len(ks)]), color='r', linestyle='--', marker='s', label='Model')

    ax.legend()
    # ax.set_ylim(0, 1.1 * max(np.max(abs(Fz_at_r[1:len(ks)])), np.max(abs(getBLHatRadius(BLH[1, :, :], r_inner, r_query)[1:len(ks)]))))
    ax.set_ylim(0, 1.1)
    ax.grid()

    ax.set_xticks(ks[::2])
    ax.set_xlabel(fr'$k = f / B \Omega$')
    ax.set_ylabel(fr'$|\hat{{F}}_k| [N/m]$')
    plt.tight_layout()
    plt.savefig(f'./Figures/iLES/net_{r_query:.4f}.pdf')

    # plt.show()
