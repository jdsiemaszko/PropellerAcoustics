from .const import PREF, SPLSHIFT
import numpy as np
from scipy.special import hankel2, jve, jv
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import interp1d
import pandas as pd
import re


def p_to_SPL(p, pref=PREF, upper=200, lower=-200):
    SPLdB = 20 * np.log10(np.abs(p) / pref) + SPLSHIFT

    SPLdB = np.minimum(SPLdB, upper)
    SPLdB = np.maximum(SPLdB, lower)

    return SPLdB

def getCylindricalCoordinates(x, axis:np.ndarray, origin:np.ndarray, radial:np.ndarray, normal:np.ndarray):
    """
    Convert Cartesian coordinates to cylindrical coordinates
    relative to a system defined by its axis, origin, and radial direction.

    x - points of size (3, N)
    axis - cylinder axis vector of size (3,)
    origin - point on the cylinder axis of size (3,)
    radial - radial direction vector of size (3,)

    returns:
    r - radial distances of size (N,)
    phi - radial angles of size (N,)
    z - axial distances of size (N,)
    """

    # Normalize axis and radial vectors
    axis = axis / np.linalg.norm(axis)
    radial = radial / np.linalg.norm(radial)
    # Compute the normal vector
    normal = np.cross(axis, radial)

    # Shift points by origin
    x_shifted = x - origin[:, np.newaxis]

    # Project onto the cylinder coordinate system
    z = np.dot(axis, x_shifted) # size (N,)
    r_vec = x_shifted - axis[:, np.newaxis] * z[np.newaxis, :]  # size (3, N)
    r = np.linalg.norm(r_vec, axis=0)

    # Compute radial angle
    cos_phi = np.dot(radial, r_vec) / r
    sin_phi = np.dot(normal, r_vec) / r
    phi = np.arctan2(sin_phi, cos_phi)

    # replace nans with zeros (happens when r=0)
    phi = np.nan_to_num(phi)

    return r, phi, z

def getPolarCoordinates(x, origin:np.ndarray, radial:np.ndarray, normal:np.ndarray):
    """
    Convert Cartesian coordinates to polar coordinates (2D)
    relative to a system defined by its axis, origin, and radial direction.

    x - points of size (2, N), all subsequent axes are ignored
    origin - point on the cylinder axis of size (2,)
    radial - radial direction vector of size (2,), defining the zero-azimuth direction
    normal = - normal direction, defining the direction of positive azimuth

    returns:
    r - radial distances of size (N,)
    phi - radial angles of size (N,)
    """

    # Normalize axis and radial vectors
    radial = radial / np.linalg.norm(radial)
    # Compute the normal vector
    normal = normal / np.linalg.norm(normal)

    # Shift points by origin
    x_shifted = x - origin[:, None]

    # Project onto the cylinder coordinate system
    r_vec = x_shifted
    r = np.linalg.norm(r_vec, axis=0)

    # Compute radial angle
    cos_phi = np.dot(radial, r_vec) / r
    sin_phi = np.dot(normal, r_vec) / r
    phi = np.arctan2(sin_phi, cos_phi)

    # replace nans with zeros (happens when r=0)
    phi = np.nan_to_num(phi)

    return r, phi

def getSphericalCoordinates(x, axis: np.ndarray, origin: np.ndarray, radial: np.ndarray, normal:np.ndarray):
    """
    Convert Cartesian coordinates to spherical coordinates
    relative to a system defined by its axis, origin, and radial direction.

    x - points of size (3, N)
    axis - reference axis vector (3,) → defines polar direction (θ = 0)
    origin - origin of spherical system (3,)
    radial - reference radial direction (3,) → defines φ = 0 direction

    returns:
    R     - radial distances (N,)
    theta - polar angles (N,)      [0, π]
    phi   - azimuth angles (N,)    [-π, π]
    """

    if len(x.shape) ==1 :
        x=x.reshape(3, 1)

    # # Normalize axis and radial vectors
    axis = axis / np.linalg.norm(axis)
    radial = radial / np.linalg.norm(radial)

    # # Ensure radial is orthogonal to axis
    radial = radial - axis * np.dot(axis, radial)
    radial = radial / np.linalg.norm(radial)


    # Shift points by origin
    x_shifted = x - origin[:, None]  # (3, N)

    # Total radial distance
    R = np.linalg.norm(x_shifted, axis=0)

    # Projection onto axis
    z = np.dot(axis, x_shifted)  # (N,)

    # Polar angle θ (angle from axis)
    theta = np.arccos(np.clip(z / R, -1.0, 1.0))

    # Radial component in transverse plane
    r_vec = x_shifted - axis[:, None] * z[None, :]

    # Azimuth angle φ
    cos_phi = np.dot(radial, r_vec)
    sin_phi = np.dot(normal, r_vec)
    phi = np.arctan2(sin_phi, cos_phi)

    # Handle singularity at R = 0
    theta = np.nan_to_num(theta)
    phi = np.nan_to_num(phi)

    return np.array([R, theta, phi])

def distance_in_polar(x_polar, y_polar):
    """
    x: array of shape (3, Nx1, Nx2, ...)
    y: array of shape (3, Ny1, Ny2, ...)
    
    returns:
        distances of shape (Nx1, Nx2, ..., Ny1, Ny2, ...)
    """
    R1, theta1, phi1 = x_polar
    R2, theta2, phi2 = y_polar

    # number of spatial dimensions (excluding the coordinate axis)
    nx = R1.ndim
    ny = R2.ndim

    # reshape for broadcasting
    R1 = R1.reshape(R1.shape + (1,) * ny)
    theta1 = theta1.reshape(theta1.shape + (1,) * ny)
    phi1 = phi1.reshape(phi1.shape + (1,) * ny)

    R2 = R2.reshape((1,) * nx + R2.shape)
    theta2 = theta2.reshape((1,) * nx + theta2.shape)
    phi2 = phi2.reshape((1,) * nx + phi2.shape)

    distance = np.sqrt(
        R1**2 + R2**2
        - 2 * R1 * R2 * (
            np.cos(theta1) * np.cos(theta2)
            + np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2)
        )
    )

    return distance 

def theodorsen(sigma):
    res = hankel2(1, sigma) / (hankel2(1, sigma) + 1j * hankel2(0, sigma))

    res[np.where(np.abs(sigma) < 1e-12)] = 1.0 # low freq limit

    return res
    # return 1.0

#TODO: fix
def periodic_sum(array, period, time):
    """
    Perform a discrete periodic sum (modal sum) of a signal onto one period.

    Parameters
    ----------
    array : array-like, shape (..., Nt, Nr)
        Signal sampled at 'time'.
        The **last-but-one axis** must correspond to time.
    period : float
        Period of the signal.
    time : 1D array, shape (Nt,)
        Time samples covering a large window, e.g., (-tmax-period/2, tmax+period/2)

    Returns
    -------
    t_mod : 1D array, shape (Np,)
        Time samples in [0, period), evenly spaced with spacing dt.
    psum : array, shape (..., Np, Nr)
        Periodic sum of the signal onto one period.
    """

    array = np.asarray(array)
    time = np.asarray(time)

    # --- assume uniform time spacing
    dt = np.mean(np.diff(time))
    n_per = int(np.round(period / dt))
    Np = n_per

    # --- output time grid
    t_mod = np.arange(Np) * dt
    t_mod = t_mod % period - period/2  # ensure [-period/2, period/2)

    # --- prepare output array
    psum_shape = list(array.shape)
    psum_shape[-2] = Np  # replace time axis
    psum = np.zeros(psum_shape, dtype=array.dtype)

    # --- fold each time sample onto the correct bin
    # last-but-one axis is assumed to be time
    time_idx = np.floor((time % period) / dt).astype(int)
    time_idx = np.clip(time_idx, 0, Np-1)  # safety

    # iterate over time samples (vectorized over all other axes)
    for i, k in enumerate(time_idx):
        # np.take along time axis
        psum[..., k, :] += array[..., i, :]

    return t_mod, psum


def periodic_sum_interpolated(array, period, time, t_new=None, kind='cubic'):
    """
    Compute the periodic sum of a signal onto one period using interpolation.
    
    f_sum(t) = sum_n f(t + n*period) over all integer n where data exists.
    
    Parameters
    ----------
    array : array-like, shape (..., Nt, Nr)
        Signal sampled at 'time'.
        Last-but-one axis corresponds to time.
    period : float
        Period of the signal.
    time : 1D array, shape (Nt,)
        Original time samples (may span multiple periods).
    t_new : 1D array, optional
        Target time array in [-period/2, period/2). If None, uniform grid is built.
    kind : str
        Interpolation kind ('linear', 'cubic', etc.)
    
    Returns
    -------
    t_new : 1D array
        Target times in [-period/2, period/2).
    psum : array, shape (..., len(t_new), Nr)
        Periodic sum interpolated onto t_new.
    """

    array = np.asarray(array)
    time = np.asarray(time)

    # --- set up output time grid
    dt = np.mean(np.diff(time))
    if t_new is None:
        Np = int(np.round(period/dt))
        t_new = np.linspace(-period/2, period/2, Np, endpoint=False)
    else:
        t_new = np.asarray(t_new)

    # --- create single interpolator over original time
    interp_func = interp1d(
        time,
        array,
        kind=kind,
        axis=-2,
        bounds_error=False,
        fill_value=0.0,
        assume_sorted=True
    )

    # --- determine which integer shifts are needed
    t_min, t_max = time[0], time[-1]
    n_min = int(np.floor((t_min - t_new[-1]) / period))
    n_max = int(np.ceil((t_max - t_new[0]) / period))
    n_range = np.arange(n_min-1, n_max +1 + 1)

    # --- sum over all periodic shifts
    psum = np.zeros(array.shape[:-2] + (len(t_new), array.shape[-1]), dtype=array.dtype)
    for n in n_range:
        psum += interp_func(t_new + n*period)

    return t_new, psum

def compute_distance_matrix(x, y):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    # Ensure input shapes are (3, Nx) and (3, Ny)
    # if x.shape[0] != 3:
    #     x = x.T
    # if y.shape[0] != 3:
    #     y = y.T

    Nx = x.shape[1]
    Ny = y.shape[1]

    # Compute pairwise distances
    diff = x[:, :, None] - y[:, None, :]
    r = np.linalg.norm(diff, axis=0)  # shape (Nx, Ny)
    return r

def plot_3D_directivity(vector_to_plot, Theta, Phi, 
    extra_script=lambda fig, ax: None,
    blending=0.1, title=None,
    valmin = None, valmax=None, fig=None, ax=None):

    Ntheta = Theta.shape[0]
    Nphi = Theta.shape[1]

    G = vector_to_plot

    for label, g in zip(
        [
            # 'real', 'imag',
        'abs'],
        [
            # np.real(G), np.imag(G),
            np.abs(G)]
    ):

        mag = np.abs(g) # take square of magnitude as measure



        mag_db = p_to_SPL(mag)
        if valmax is None:
            valmax = min(mag_db.max(), 200)
        if valmin is None:
            valmin = max(mag_db.min(), -120)
            
        print(f'maximum magnitude: {np.max(mag_db)} [dB]')

        # --- normalize radius ---
        r0 = (mag_db - valmin) / (valmax - valmin) * (1 - blending) + blending
        mag_db0 = mag_db.reshape(Ntheta, Nphi)
        r0 = r0.reshape(Ntheta, Nphi)

        r_c = np.maximum(r0, blending)
        Theta_c = Theta
        Phi_c = Phi
        mag_db_c = mag_db0

        # --- spherical to Cartesian ---
        X = r_c * np.sin(Theta_c) * np.cos(Phi_c)
        Y = r_c * np.sin(Theta_c) * np.sin(Phi_c)
        Z = r_c * np.cos(Theta_c)

        if fig is None or ax is None:   
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection="3d")

        # --- color normalization ---
        norm = colors.Normalize(vmin=valmin, vmax=valmax)
        facecolors = plt.cm.viridis(norm(mag_db_c))

        # --- build quad faces ---
        faces = []
        face_colors = []

        for i in range(Ntheta - 1):
            for j in range(Nphi - 1):
                verts = [
                    [X[i, j],     Y[i, j],     Z[i, j]],
                    [X[i+1, j],   Y[i+1, j],   Z[i+1, j]],
                    [X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1]],
                    [X[i, j+1],   Y[i, j+1],   Z[i, j+1]],
                ]
                faces.append(verts)
                face_colors.append(facecolors[i, j])

        poly = Poly3DCollection(
            faces,
            facecolors=face_colors,
            edgecolors='k',
            linewidths=0.5,
            alpha=0.75
        )

        ax.add_collection3d(poly)
        # --- colorbar ---
        mappable = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        mappable.set_array(mag_db_c)
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.1)
        cbar.set_label("Directivity [dB]")

        # --- axes ---
        if title is not None:
            ax.set_title(title)
        ax.set_aspect('equal')
        ax.set_box_aspect([1, 1, 1])
        RR = np.max(r0) * 1.1
        ax.set_xlim([-RR, RR])
        ax.set_ylim([-RR, RR])    
        ax.set_zlim([-RR, RR])
        # ax.set_axis_off()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        extra_script(fig, ax)
        ax.grid()


    return fig, ax


def plot_directivity_contour(theta, phi, magnitudes, levels=20, cmap='viridis', title=None, xlabel=None, ylabel=None, fig=None, ax=None,):
    """
    Plot a 2D contour map of directivity (in dB) vs theta and phi.

    Parameters
    ----------
    theta : array_like
        Array of theta angles (radians) of shape (Ntheta,)
    phi : array_like
        Array of phi angles (radians) of shape (Nphi,)
    magnitudes_db : array_like
        2D array of directivity in dB, shape (Ntheta, Nphi)
    levels : int
        Number of contour levels
    cmap : str
        Colormap name
    """

    if fig is None or ax is None:   
        fig, ax = plt.subplots(figsize=(7, 5))

    Theta, Phi = np.meshgrid(theta, phi, indexing='ij')  # shape (Ntheta, Nphi)
    magnitudes_db = p_to_SPL(magnitudes).reshape(Theta.shape)  # ensure shape matches Theta and Phi
    # 2D filled contour plot
    cf = ax.contourf(Phi, Theta, magnitudes_db, levels=levels, cmap=cmap)
    # cf = ax.imshow(magnitudes_db, extent=(phi.min(), phi.max(), theta.min(), theta.max()), origin='lower', aspect='auto', cmap=cmap)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Directivity [dB]")

    if xlabel is None:
        xlabel = "Phi [deg]"
    if ylabel is None:
        ylabel = "Theta [deg]"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    return fig, ax


def read_airfoil_table(filename, skip_lines=None):
    """
    Reads the AIRFOIL SUMMARY DATA numeric table into a pandas DataFrame.

    Parameters
    ----------
    filename : str
        Path to the file.
    skip_lines : int, optional
        If provided, start reading numeric data after this many lines.
        If None, the table is automatically detected.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing only the numeric table data.
    """

    with open(filename, "r") as f:
        lines = f.readlines()

    header = None
    data_start = None

    # ------------------------------------------------------------
    # Automatic detection
    # ------------------------------------------------------------
    if skip_lines is None:

        # 1️⃣ Find header line
        for i, line in enumerate(lines):
            if "STATION" in line and "CHORD" in line:
                header = line.strip().split()
                header_line_index = i
                break

        if header is None:
            raise RuntimeError("Could not locate table header.")

        ncols = len(header)

        # 2️⃣ Find first numeric row after header
        for i in range(header_line_index + 1, len(lines)):
            stripped = lines[i].strip()
            if not stripped:
                continue

            parts = stripped.split()

            if len(parts) != ncols:
                continue

            try:
                [float(x) for x in parts]
                data_start = i
                break
            except ValueError:
                continue

        if data_start is None:
            raise RuntimeError("Could not locate start of numeric table.")

    # ------------------------------------------------------------
    # Manual skip mode
    # ------------------------------------------------------------
    else:
        data_start = skip_lines
        header = None
        ncols = None

    # ------------------------------------------------------------
    # Read numeric rows
    # ------------------------------------------------------------
    data_rows = []

    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped:
            break

        parts = stripped.split()

        # Stop if row length changes
        if ncols is not None and len(parts) != ncols:
            break

        try:
            row = [float(x) for x in parts]
            data_rows.append(row)
        except ValueError:
            break

    df = pd.DataFrame(data_rows)

    # ------------------------------------------------------------
    # Assign clean column names
    # ------------------------------------------------------------
    if header is not None and len(header) == df.shape[1]:

        # Make duplicate column names unique (e.g. PITCH_1, PITCH_2)
        counts = {}
        clean_cols = []

        for col in header:
            if col not in counts:
                counts[col] = 0
                clean_cols.append(col)
            else:
                counts[col] += 1
                clean_cols.append(f"{col}_{counts[col]}")

        df.columns = clean_cols

    return df