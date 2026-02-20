from .const import PREF, SPLSHIFT
import numpy as np
from scipy.special import hankel2, jve, jv
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def p_to_SPL(p, pref=PREF):

    return 20 * np.log10(np.abs(p) / pref) + SPLSHIFT

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

    # # Normalize axis and radial vectors
    axis = axis / np.linalg.norm(axis)
    radial = radial / np.linalg.norm(radial)

    # # Ensure radial is orthogonal to axis
    radial = radial - axis * np.dot(axis, radial)
    radial = radial / np.linalg.norm(radial)


    # Shift points by origin
    x_shifted = x - origin[:, np.newaxis]  # (3, N)

    # Total radial distance
    R = np.linalg.norm(x_shifted, axis=0)

    # Projection onto axis
    z = np.dot(axis, x_shifted)  # (N,)

    # Polar angle θ (angle from axis)
    theta = np.arccos(np.clip(z / R, -1.0, 1.0))

    # Radial component in transverse plane
    r_vec = x_shifted - axis[:, np.newaxis] * z[np.newaxis, :]

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
    n_per = int(np.ceil(period / dt))
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

def plot3DDirectivity(vector_to_plot, Theta, Phi, 
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
            valmax = mag_db.max()
        if valmin is None:
            valmin = mag_db.min()
            
        print(f'maximum magnitude: {np.max(mag_db)} [dB]')

        # --- normalize radius ---
        r0 = (mag_db - valmin) / (valmax - valmin) * (1 - blending) + blending
        mag_db0 = mag_db.reshape(Ntheta, Nphi)
        r0 = r0.reshape(Ntheta, Nphi)

        r_c = r0
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

        plt.show()
        plt.close(fig)


    return fig, ax
