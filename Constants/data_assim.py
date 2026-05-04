import numpy as np
import h5py

def getGojonData(datadir, D, L, shape='D', B=2, RPM=8000):
    """
    parse the h5 files from the Gojon et al. 2023 dataset
    """
    if shape != 'A':
        casefile = f'ISAE_2_{shape}{int(1000*D)}_L{int(1000*L)}'
    else:
        casefile = f'ISAE_2_airfoil_8000'
    def load_h5(filename):
        return h5py.File(filename, "r")

    with load_h5(f"{datadir}/{casefile}_autopower.h5") as f:
        g = f[casefile]
        freq = np.array(g["frequency_Hz"])
        ap = g["Autopower"]

        theta_exp = np.array(g["theta_deg"])[0] # polar angle array
        radius = g["radius_m"][0][0] # float

        BPF = B * RPM / 60
        if shape != 'A':
            phi_exp = np.array(g["phi_deg"])[0]# azimuth
            data = np.array(ap[f"Autopower_RPM_{RPM}_Pa2"]) # (freq, polar, azimuth), (aziuth=0 = > beam axis, azimuth=9 => 90 deg)
        else: # different labels for airfoil data :(
            phi_exp = np.array(g["phi_L20_deg"])[0] # azimuth
            data = np.array(ap[f"Autopower_arfoil20_Pa2"])


    theta = 90 - theta_exp
    phi = 180 - phi_exp
    print(f'Theta_exp = {theta_exp} deg, Phi_exp = {phi_exp} deg')
    print(f'Theta = {theta} deg, Phi = {phi} deg')
    x_cart = np.array([
        radius * np.cos(np.deg2rad(phi[None, :])) * np.sin(np.deg2rad(theta[:, None])),
        radius * np.sin(np.deg2rad(phi[None, :])) * np.sin(np.deg2rad(theta[:, None])),
        radius * np.cos(np.deg2rad(theta[:, None])) * np.ones_like(phi[None, :]),
    ]) # shape Ntheta, Nphi

    return data, BPF, freq, x_cart, theta, phi, theta_exp, phi_exp, casefile

def getHarmonicsFromData(data, frequency, BPF, range=0.02):
    """
    Extract modal amplitudes at multiples of BPF.
    
    Parameters
    ----------
    data      : array of shape (Nfreq, Ntheta, Nphi)
    frequency : array of shape (Nfreq,)
    BPF       : float, fundamental frequency
    range     : float, fractional half-width of frequency window (default ±2%)
    
    Returns
    -------
    data_modal : array of shape (Nharmonics, Ntheta, Nphi)
    ms         : array of shape (Nharmonics,), harmonic numbers 1, 2, ..., Nharmonics
    """
    Nfreq = frequency.shape[0]
    frequency = frequency.ravel()
    freq_max = np.max(frequency)  # or np.max(frequency)
    Nharmonics = int(np.floor(freq_max / BPF))
    ms = np.arange(1, Nharmonics + 1)  # harmonic numbers: 1, 2, ..., Nharmonics

    data_modal = []

    for m in ms:
        f0 = m * BPF  # target frequency for this harmonic

        # frequency window: ±range around f0
        fmin = (1 - range) * f0
        fmax = (1 + range) * f0

        mask = (frequency >= fmin) & (frequency <= fmax)
        mask.reshape((Nfreq, ))
        if not np.any(mask):
            # fallback: nearest frequency if window is empty
            idx = np.argmin(np.abs(frequency - f0))
            data_modal.append(data[idx, :, :])  # shape (Ntheta, Nphi)
        else:
            # peak (max) across the masked frequencies, axis=0 → shape (Ntheta, Nphi)
            data_modal.append(np.max(data[mask, :, :], axis=0))

    data_modal = np.array(data_modal)  # shape (Nharmonics, Ntheta, Nphi)

    return data_modal, ms