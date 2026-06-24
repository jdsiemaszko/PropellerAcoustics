import numpy as np
import matplotlib.pyplot as plt
from Constants.helpers import plot_directivity_contour, p_to_SPL
from scipy.interpolate import interp1d

def compute_frequency_map(pressure_maps, dt, target_frequency):
    """
    Compute complex pressure field at a given frequency.

    Parameters
    ----------
    pressure_maps : ndarray
        Shape (Nt, Nr, Ntheta)
    dt : float
        Time step size [s]
    target_frequency : float
        Frequency of interest [Hz]

    Returns
    -------
    P_f : ndarray
        Complex pressure field, shape (Nr, Ntheta)
    freq : float
        Actual FFT frequency selected
    """

    Nt = pressure_maps.shape[0]

    # remove mean
    fluctuations = pressure_maps - pressure_maps.mean(axis=0)

    # FFT along time axis
    P_hat = np.fft.rfft(fluctuations, axis=0) / Nt

    freqs = np.fft.rfftfreq(Nt, dt)

    idx = np.argmin(np.abs(freqs - target_frequency))

    # P_hat = np.nan_to_num(P_hat)
    return P_hat[idx], freqs[idx]

def plot_frequency_contour(
        pressure_maps,
        r_grid,
        theta_grid,
        dt,
        target_frequency,
):
    """
    Plot |P(f,r,theta)| at a selected frequency.

    pressure_maps.shape = (Nt, Nr, Ntheta)
    """

    P_f, actual_freq = compute_frequency_map(
        pressure_maps,
        dt,
        target_frequency
    )

    # # Shape: (Nr, Ntheta)
    # magnitudes = p_to_SPL(P_f)

    # # Shape: (Nr, Ntheta)
    # Phi, Theta = np.meshgrid(
    #     r_grid,
    #     np.rad2deg(theta_grid),
    #     indexing="ij"
    # )

    VMIN, VMAX = 80, 130
    levels = np.linspace(VMIN, VMAX, 21)
    levels_phase = np.linspace(-np.pi, np.pi, 21)

    # Shape: (Nr, Ntheta)
    magnitudes = P_f
    magnitudes=magnitudes.reshape((r_grid.shape[0], theta_grid.shape[0]))

    theta_deg = np.rad2deg(theta_grid)            # currently -180 .. 180
    theta_deg_0_360 = np.mod(theta_deg, 360)      # map negatives -> 180..360
    order = np.argsort(theta_deg_0_360)           # indices that rearrange to 0 -> 360

    theta_deg_sorted = theta_deg_0_360[order]
    magnitudes = magnitudes[:, order]             # keep data aligned with reordered theta

    Phi, Theta = np.meshgrid(
        r_grid, theta_deg_sorted, indexing="ij"
    )


    period = 360
    Theta = np.concatenate(
    [Theta[:, -1:] - period, Theta, Theta[:, :1] + period],
    axis=1
    )

    Phi = np.concatenate(
        [Phi[:, -1:], Phi, Phi[:, :1]],
        axis=1
    )

    magnitudes = np.concatenate(
        [magnitudes[:, -1:], magnitudes, magnitudes[:, :1]],
        axis=1
    )

    print(f'max SPL: {p_to_SPL(magnitudes).max()}')
    rtip = 0.1
    rroot = 0.1 * 0.16

    reference_xrange = rtip - rroot
    reference_width = 6.0
    reference_height = 4.0
    reference_width_long = reference_width * 2 * rtip / reference_xrange
    HEIGHT = reference_height

    xrange = Phi.max() - Phi.min()

    WIDTH = reference_width * xrange / reference_xrange

    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))

    fig, ax, cf = plot_directivity_contour(
        Theta=Theta,
        Phi=Phi,
        magnitudes=magnitudes,
        # title=f"|P(f)| @ {actual_freq:.1f} Hz",
        ylabel=r"$\theta$ [deg]",
        xlabel="z [m]",
        fig=fig, ax=ax,
        levels=levels,
            cmap='jet'


    )

    # ax.set_xlim(rroot, rtip)

    RADIUS = 0.01
    ax.set_aspect((360 / (rtip - rroot))**(-1))
    if Phi.max() > rtip:
        ax.axvline(rroot, color='white', linestyle='dashed')
        ax.axvline(rtip, color='white', linestyle='dashed')
    ax.set_ylim(0, 360)
    ax.set_xlim(right=2*rtip-1e-6)

    # plt.colorbar(cf, ax=ax, label="SPL [dB]")

    plt.tight_layout()
    return fig, ax

def get_sectional_loading(pressure_maps, dt, r_grid, theta_grid, r_query = None, suffix=''):
    """
        Compute complex pressure field at a given frequency.

        Parameters
        ----------
        pressure_maps : ndarray
            Shape (Nt, Nr, Ntheta)
        dt : float
            Time step size [s]
        r_grid: shape Nr, Ntheta
        theta_grid: shape Nr, Ntheta


        Returns
        -------
        P_f : ndarray
            Complex pressure field, shape (Nr, Ntheta)
        freq : float
            Actual FFT frequency selected
    """
    Nt, Nrth = pressure_maps.shape
    Nr= r_grid.shape[0]
    Ntheta = theta_grid.shape[0]
    pressure_maps = np.reshape(pressure_maps, (Nt, Nr, Ntheta))
    RADIUS = 0.01
    # DTHETA = 2 * np.pi / Ntheta
    DTHETA = theta_grid[1] - theta_grid[0]
    Omega = 8000/60 * 2 * np.pi
    period = 2 * np.pi/ Omega

    net_loading_z = -RADIUS * np.sum(pressure_maps * np.sin(theta_grid[None, None, :]) * DTHETA, axis=-1) # Nt, Nr
    net_loading_phi = RADIUS * np.sum(pressure_maps * np.cos(theta_grid[None, None, :])* DTHETA, axis=-1)

    # loading_harmonics_z = np.conj(np.fft.rfft(net_loading_z, axis=0)) / Nt
    # loading_harmonics_phi = np.conj(np.fft.rfft(net_loading_phi, axis=0)) / Nt
    # freqs = np.fft.rfftfreq(Nt, dt)

    ks = np.arange(1, 81, 1)
    freqs = ks / period 
    time = np.arange(0, 400, 1) * dt

    loading_harmonics_z = 1/period * np.sum(
        dt * net_loading_z[None, :, :] * np.exp(1j * Omega * ks[:, None, None] * time[None, :, None]), axis=1
    )

    loading_harmonics_phi = 1/period * np.sum(
        dt * net_loading_phi[None, :, :] * np.exp(1j * Omega * ks[:, None, None] * time[None, :, None]), axis=1
    )

    for component, name in zip([net_loading_z, net_loading_phi, loading_harmonics_z, loading_harmonics_phi],
                               ['Fztime', 'Fphitime', 'Fzfreq', 'Fphifreq']):
        interp_func = interp1d(
            r_grid,
            component,
            axis=1,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        interp = interp_func(r_query) # N0, Nrquery

        destination = f'./Data/current/surface_pressure/{name}_iLES{suffix}.npy'

        print(f'saving results of sectional loading at r_query={r_query} to {destination}')
        np.save(destination, interp)

    destination_freq = f'./Data/current/surface_pressure/freqs_iLES{suffix}.npy'
    np.save(destination_freq, freqs)

    destination_time = f'./Data/current/surface_pressure/times_iLES{suffix}.npy'
    np.save(destination_time, time)

    destination_rq = f'./Data/current/surface_pressure/rquery_iLES{suffix}.npy'
    np.save(destination_rq, r_query)

    return net_loading_z, net_loading_phi, loading_harmonics_z, loading_harmonics_phi, freqs, r_grid

def plot_loading_harmonics(
    loading_harmonics,
    freqs,
    r_grid,
    r_query,
    component_name="z",
    plot_phase=True,
):
    """
    Plot harmonic loading spectrum at an arbitrary radius.

    Parameters
    ----------
    loading_harmonics : ndarray
        Complex loading harmonics, shape (Nf, Nr)
    freqs : ndarray
        Frequency vector, shape (Nf,)
    r_grid : ndarray
        Radial station locations, shape (Nr,)
    r_query : float
        Radius at which to extract loading.
    component_name : str
        Label for plot.
    plot_phase : bool
        If True, also plot phase.
    """

    # interpolate real and imaginary parts separately
    interp_fun = interp1d(
        r_grid,
        loading_harmonics,
        axis=1,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )

    harmonic_interp = interp_fun(r_query) 
    amplitude = np.abs(harmonic_interp)
    phase = np.angle(harmonic_interp)

    if plot_phase:
        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        ax[0].plot(freqs, amplitude)
        ax[0].set_ylabel("Amplitude")
        ax[0].set_title(
            f"{component_name}-loading harmonics at r={r_query:.4f}"
        )
        ax[0].grid(True)

        ax[1].plot(freqs, phase)
        ax[1].set_ylabel("Phase [rad]")
        ax[1].set_xlabel("Frequency [Hz]")
        ax[1].grid(True)

    else:
        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(freqs, amplitude)
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_title(
            f"{component_name}-loading harmonics at r={r_query:.4f}"
        )
        ax.grid(True)

    plt.tight_layout()
    return harmonic_interp

if __name__ == "__main__":
    filename =  "./Data/current/pressure_maps_strut_iLES.npy"
    data = np.load(filename)
    filename_r =  "./Data/current/rs_strut_iLES.npy"
    r_grid = np.load(filename_r)
    filename_theta = "./Data/current/thetas_maps_strut_iLES.npy"
    theta_grid = np.load(filename_theta)

    Omega = 8000 / 60 * 2 * np.pi
    B = 2
    print(r_grid.max())
    net_loading_z, net_loading_phi, loading_harmonics_z, loading_harmonics_phi, freqs, r_stations = get_sectional_loading(data,
                             2 * np.pi / Omega / 400,  r_grid[0], theta_grid[0],
                            #    r_query=np.array([0.5, 0.8, 0.9]) * 0.1,
                            # r_query = r_grid[0]
                            r_query = np.linspace(0.016, 0.1, 37),
                            suffix='TEST'
                               )
    # harm_z = plot_loading_harmonics(
    #     loading_harmonics_z,
    #     freqs,
    #     r_stations,
    #     r_query=0.8,
    #     component_name="z",
    # )

    # harm_phi = plot_loading_harmonics(
    #     loading_harmonics_phi,
    #     freqs,
    #     r_stations,
    #     r_query=0.8,
    #     component_name="phi",
    # )

    # import os
    for m in np.arange(1, 11, 1):
        print(f'saving plots for m={m}')
        folder_name = f"./Figures/SurfacePressureComponents_M{m}_RdBu"

        fig, ax = plot_frequency_contour(data, r_grid[0], theta_grid[0], 2 * np.pi / Omega / 400, Omega * B / 2 / np.pi * m)
        plt.show()
        fig.savefig(
        os.path.join(folder_name, f"p_surface_iLES.pdf"),
        dpi=300,
        bbox_inches="tight",
        )
        plt.close(fig)