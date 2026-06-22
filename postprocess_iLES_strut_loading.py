import numpy as np
import matplotlib.pyplot as plt
from Constants.helpers import plot_directivity_contour, p_to_SPL

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
        xlabel="r [m]",
        fig=fig, ax=ax,
        levels=levels,

    )

    # ax.set_xlim(rroot, rtip)

    RADIUS = 0.01
    ax.set_aspect((360 / (rtip - rroot))**(-1))
    if Phi.max() > rtip:
        ax.axvline(rroot, color='k', alpha=0.7)
        ax.axvline(rtip, color='k', alpha=0.7)

    # plt.colorbar(cf, ax=ax, label="SPL [dB]")

    plt.tight_layout()
    return fig, ax

if __name__ == "__main__":
    filename =  "./Data/current/pressure_maps_strut_iLES.npy"
    data = np.load(filename)
    filename_r =  "./Data/current/rs_strut_iLES.npy"
    r_grid = np.load(filename_r)
    filename_theta = "./Data/current/thetas_maps_strut_iLES.npy"
    theta_grid = np.load(filename_theta)

    Omega = 8000 / 60 * 2 * np.pi
    B = 2

    plot_frequency_contour(data, r_grid[0], theta_grid[0], 2 * np.pi / Omega / 400, Omega * B / 2 / np.pi * 5)
    plt.show()