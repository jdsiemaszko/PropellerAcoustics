import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

datadir = './dataverse_files'
datafile = 'ISAE_2_D20_L20'

# datafile = 'ISAE_2_T10_L20'

pref = 20e-6  # reference pressure (20 µPa)
NB = 2
RPM = 8000
NBPF = 2 # harmonic to plot directivity for

def load_h5(filename):
    return h5py.File(filename, "r")

def spl_from_autopower(Pa2):
    return 10 * np.log10(Pa2 / (pref * pref))


def get_spl_at_harmonic(fplus, spls, harmonic: int, alpha=0.01):
    """
    Extract SPL at a given BPF harmonic.

    Parameters
    ----------
    fplus : array_like, shape (Nfreq,)
        Reduced frequencies (f / BPF - Nfreq)
    spls : array_like, shape (Nfreq, Ntheta, Nphi)
        SPL data
    harmonic : int
        Harmonic number to extract
    alpha : float
        Fractional window around harmonic, i.e., pick frequencies in
        [harmonic*(1-alpha), harmonic*(1+alpha)]

    Returns
    -------
    spl_harmonic : ndarray, shape (Ntheta, Nphi)
        SPL at the given harmonic (max within window)
    """

    # define frequency window
    fmin = harmonic * (1 - alpha)
    fmax = harmonic * (1 + alpha)

    # boolean mask of frequencies inside window
    mask = (fplus >= fmin) & (fplus <= fmax)

    if not np.any(mask):
        # fallback: nearest frequency
        idx = np.argmin(np.abs(fplus - harmonic))
        spl_harmonic = spls[idx]
    else:
        # take max over frequencies in the window
        spl_harmonic = np.max(spls[mask], axis=0)  # shape (Ntheta, Nphi)

    return spl_harmonic


def plot_directivity(fig, ax, units, magnitudes, valmax=None, valmin=None):
    """
    units:      (3, Ntheta, Nphi) unit direction vectors
    magnitudes: (Ntheta, Nphi) array of directivity values
    """
    _, Ntheta, Nphi = units.shape
    if valmax is None:
        valmax = magnitudes.max()
    if valmin is None:
        valmin = magnitudes.min()
    r = (magnitudes - valmin) / (valmax - valmin) / 2 + 0.5

    # --- normalize radius ---
    unitsnorm = np.linalg.norm(units, axis=0)

    # --- scale unit vectors ---
    X = units[0] / unitsnorm * r
    Y = units[1] / unitsnorm * r
    Z = units[2] / unitsnorm * r

    # --- color normalization ---
    norm = colors.Normalize(vmin=valmin, vmax=valmax)
    facecolors = plt.cm.viridis(norm(magnitudes))

    # --- plot surface without closing phi ---
    # ax.plot_surface(
    #     X, Y, Z,
    #     facecolors=facecolors,
    #     rstride=1,
    #     cstride=1,
    #     linewidth=1.,        # line width of cell edges
    #     edgecolors='k',        # black edges
    #     # antialiased=True,
    #     shade=False,
    #     alpha=0.75

    # )

    # flatten the grid into quads
    faces = []
    face_colors = []

    Ntheta, Nphi = X.shape

    for i in range(Ntheta-1):
        for j in range(Nphi-1):
            # vertices of one quad
            verts = [
                [X[i,j],     Y[i,j],     Z[i,j]],
                [X[i+1,j],   Y[i+1,j],   Z[i+1,j]],
                [X[i+1,j+1], Y[i+1,j+1], Z[i+1,j+1]],
                [X[i,j+1],   Y[i,j+1],   Z[i,j+1]],
            ]
            faces.append(verts)
            # face color from your facecolors array
            face_colors.append(facecolors[i,j])

    # create Poly3DCollection
    poly = Poly3DCollection(
        faces,
        facecolors=face_colors,
        edgecolors='k',  # black edges
        linewidths=0.5,
        alpha=0.75
    )

    ax.add_collection3d(poly)

    # --- colorbar ---
    mappable = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    mappable.set_array(magnitudes)
    fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.1, label="Directivity [dB]")


    rmax = np.max(r) * 0.65
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)
    ax.set_zlim(-rmax, rmax)

    # --- axes ---
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.set_title("Far-field directivity")

        # set viewing angle
    ax.view_init(elev=20, azim=-45)



def plot_directivity_contour(fig, ax, theta, phi, magnitudes_db, levels=20, cmap='viridis'):
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
    Theta, Phi = np.meshgrid(theta, phi, indexing='ij')  # shape (Ntheta, Nphi)

    # 2D filled contour plot
    cf = ax.contourf(np.rad2deg(Phi), np.rad2deg(Theta), magnitudes_db, levels=levels, cmap=cmap)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Directivity [dB]")

    ax.set_xlabel("Phi [deg]")
    ax.set_ylabel("Theta [deg]")
    ax.set_title("Far-field Directivity")

with load_h5(f"{datadir}/{datafile}_autopower.h5") as f:
    g = f[f"{datafile}"]
    freq = np.array(g["frequency_Hz"])[0]
    ap = g["Autopower"]

    phi = np.array(g["phi_deg"])[0] # azimuth
    theta = np.array(g["theta_deg"])[0] # polar
    radius = g["radius_m"][0][0] # float

    autopow = np.array(ap[f"Autopower_RPM_{RPM}_Pa2"])  # (freq, polar, azimuth)

    spl = spl_from_autopower(autopow)


phi, theta = np.deg2rad(phi), np.deg2rad(theta)
Nphi, Ntheta = np.shape(phi)[0], np.shape(theta)[0]

fplus = freq / NB / RPM * 60
spl_ = get_spl_at_harmonic(fplus, spl, NBPF, alpha=0.01)
# explicitly build each component with shape (Ntheta, Nphi)
# x = radius * np.cos(theta[:, None]) * np.cos(phi[None, :])
# y = radius * np.sin(theta[:, None]) * np.cos(phi[None, :])
# z = radius * np.ones((Ntheta, 1)) * np.sin(phi[None, :])

x = radius * np.cos(theta[:, None]) * np.cos(phi[None, :])
y = radius * np.cos(theta[:, None]) * np.sin(phi[None, :])
z = radius * np.ones((1, Nphi)) * np.sin(theta[:, None])

vectors = np.stack((x, y, z), axis=0)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")
plot_directivity(fig, ax, vectors, spl_, valmax=50, valmin=20)
plt.show()
plt.close(fig)

fig, ax = plt.subplots()
plot_directivity_contour(fig, ax, theta, phi, spl_)
plt.show()
plt.close(fig)

print(f'maximum SPL recorded: {np.max(spl_)} dB')