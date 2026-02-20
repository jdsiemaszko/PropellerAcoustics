from .const import PREF, SPLSHIFT
import numpy as np

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
