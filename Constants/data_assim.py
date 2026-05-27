import numpy as np
import h5py

def getGojonData(datadir, D, L, shape='D', B=2, RPM=8000):
    """
    parse the h5 files from the Gojon et al. 2023 dataset
    """
    match shape:
        case 'PARROT':
            casefile = f'ISAE_PARROT_D{int(1000*D)}_L{int(1000*L)}'
        case 'A':
            casefile = f'ISAE_2_airfoil_8000'
        case _:
            casefile = f'ISAE_2_{shape}{int(1000*D)}_L{int(1000*L)}'

    def load_h5(filename):
        return h5py.File(filename, "r")

    with load_h5(f"{datadir}/{casefile}_autopower.h5") as f:
        g = f[casefile]
        freq = np.array(g["frequency_Hz"])
        ap = g["Autopower"]

        theta_exp = np.array(g["theta_deg"])[0] # polar angle array
        radius = g["radius_m"][0][0] # float

        BPF = np.abs(B * RPM / 60)
        if shape != 'A':
            phi_exp = np.array(g["phi_deg"])[0]# azimuth
            data = np.array(ap[f"Autopower_RPM_{RPM}_Pa2"]) # (freq, polar, azimuth), (aziuth=0 = > beam axis, azimuth=9 => 90 deg)
        else: # different labels for airfoil data :(
            phi_exp = np.array(g["phi_L20_deg"])[0] # azimuth
            data = np.array(ap[f"Autopower_arfoil20_Pa2"])

    theta_exp = -theta_exp # wrong arrangement in the dataset!
    theta = 90 - theta_exp
    phi = 180 - phi_exp
    print(f'Theta_exp = {theta_exp} deg, Phi_exp = {phi_exp} deg')
    print(f'Theta = {theta} deg, Phi = {phi} deg')
    x_cart = np.array([
        radius * np.cos(np.deg2rad(phi[None, :])) * np.sin(np.deg2rad(theta[:, None])),
        radius * np.sin(np.deg2rad(phi[None, :])) * np.sin(np.deg2rad(theta[:, None])),
        radius * np.cos(np.deg2rad(theta[:, None])) * np.ones_like(phi[None, :]),
    ]) # shape Ntheta, Nphi

    # switch the phi array if the rotor rotates the other way (CCW)
    # all else should be unaffected
    # TODO: check if correct
    if RPM < 0:
        phi = - phi

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


def read_selig_airfoil(filename):
    """
    Reads an airfoil in Selig format.
    
    Returns:
        name (str): airfoil name (first line)
        x (ndarray): x coordinates
        y (ndarray): y coordinates
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # First line is the airfoil name
    name = lines[0].strip()

    coords = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue  # skip empty lines
        
        parts = line.split()
        if len(parts) >= 2:
            try:
                x, y = float(parts[0]), float(parts[1])
                coords.append((x, y))
            except ValueError:
                pass  # skip malformed lines

    coords = np.array(coords)
    x = coords[:, 0]
    y = coords[:, 1]

    return name, x, y


def split_surfaces(x, y):
    i_le = np.argmin(x)
    xu, yu = x[:i_le+1], y[:i_le+1]   # upper surface
    xl, yl = x[i_le:], y[i_le:]       # lower surface
    return xu, yu, xl, yl


def compute_camber_thickness(x, y, n_points=200):
    """
    Compute camber line and thickness distribution.

    Returns:
        xc      : common x stations (0 → 1)
        camber  : camber line
        thickness : thickness distribution
    """
    xu, yu, xl, yl = split_surfaces(x, y)

    # Ensure monotonic increasing x for interpolation
    xu, yu = xu[::-1], yu[::-1]  # flip upper: LE → TE
    # lower already LE → TE typically

    # Common chordwise stations (cosine spacing optional)
    beta = np.linspace(0, np.pi, n_points)
    xc = 0.5 * (1 - np.cos(beta))  # cosine spacing (better near LE)

    # Interpolate
    yu_interp = np.interp(xc, xu, yu)
    yl_interp = np.interp(xc, xl, yl)

    # Compute camber & thickness
    camber = 0.5 * (yu_interp + yl_interp)
    thickness = yu_interp - yl_interp

    return xc, camber, thickness

