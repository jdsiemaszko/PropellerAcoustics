import numpy as np
from scipy.special import hankel1, jv, kv
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# from numpy.polynomial.legendre import leggauss
# from numpy.polynomial.laguerre import laggauss

"""
Taylored Green's function on the Helmholtz equation.

"""

def getSphericalCoordinates(x):
    x = np.atleast_2d(x)
    if x.shape[0] != 3:
        x = x.T
    R = np.linalg.norm(x, axis=0)
    phi = np.arccos(x[2, :] / R)  # angle to the z axis
    theta = np.arctan2(x[1, :], x[0, :])  # angle to the x axis in the xy plane

    xpolar = np.array([R, phi, theta])
    return xpolar

def getSphericalCoordinatesWRTx(x):
    x = np.atleast_2d(x)
    if x.shape[0] != 3:
        x = x.T
    R = np.linalg.norm(x, axis=0)
    phi = np.arccos(x[0, :] / R)  # angle to the X axis
    theta = np.arctan2(x[1, :], x[2, :])  # angle to the X axis in the ZY plane

    xpolar = np.array([R, phi, theta])
    return xpolar

def getCylindricalCoordinatesWRTx(x):
    x = np.atleast_2d(x)
    if x.shape[0] != 3:
        x = x.T
    R = np.linalg.norm(x[1:3, :], axis=0)
    phi = np.arctan2(x[2, :], x[1, :])  # angle to the X axis in the ZY plane
    xx = x[0, :]
    xpolar = np.array([R, phi, xx])
    return xpolar

def compute_distance_matrix(x, y):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    # Ensure input shapes are (3, Nx) and (3, Ny)
    if x.shape[0] != 3:
        x = x.T
    if y.shape[0] != 3:
        y = y.T

    Nx = x.shape[1]
    Ny = y.shape[1]

    # Compute pairwise distances
    diff = x[:, :, np.newaxis] - y[:, np.newaxis, :]
    r = np.linalg.norm(diff, axis=0)  # shape (Nx, Ny)
    return r

def FreeSpaceGreenFunction(x, y, k, dim=3):
    """
    x - observer position of size (3, Nx)
    y - source position of size (3, Ny)
    k - wavenumber
    dim - dimension (default is 3)

    returns:
    G - Green's function matrix of size (Nx, Ny)
    """

    r = compute_distance_matrix(x, y)

    if np.any(r == 0):
        raise ValueError("Source and field points cannot be the same.")

    if dim == 3:
        G = np.exp(1j * k * r) / (4 * np.pi * r)
        return G  # shape (Nx, Ny)
    else:
        raise NotImplementedError("Only 3D Green's function is implemented as of now.")
    
# def CylinderGreenFunction(x, y, k, a, dim=3):
#     """
#     x - observer
#     y - source
#     k - wavenumber
#     a - cylinder radius
#     dim - dimension (default is 3)
#     """
#     r = compute_distance_matrix(x, y)

#     if np.any(r == 0):
#         raise ValueError("Source and field points cannot be the same.")

#     if dim==3:
#         G = np.zeros_like(r, dtype=complex)
#         kint = np.linspace(0, 10 * k, 100) # discretization?
#         for index, ki in enumerate(kint):
#             print('computing the cylinder Green function for ki = ', ki, index, ' out of ', len(kint))

#             kk = np.sqrt(k**2 - ki**2 + 0j)
            
#             obs_polar = getSphericalCoordinates(x)
#             src_polar = getSphericalCoordinates(y)

#             cos0 = np.cos(ki * (x[2, :, np.newaxis] - y[2, np.newaxis, :]))  # shape (Nx, Ny)
            
#             for m in range(30): # discretization?
#                 # print(m)
#                 cos1 = np.cos(m * (obs_polar[2, :, np.newaxis] - src_polar[2, np.newaxis, :]))  # shape (Nx, Ny)
#                 epsilonm = 1 if m == 0 else 2

#                 Hkm0 = hankel1(m, kk * obs_polar[0, :])  # shape (Nx,)
#                 Hkm1 = hankel1(m, kk * src_polar[0, :])  # shape (Ny,)

#                 Jkm0 = jv(m-1, kk * a)  # shape ()
#                 Jkm1 = jv(m+1, kk * a)  # shape ()
#                 Hkm0_a = hankel1(m-1, kk * a)  # shape ()
#                 Hkm1_a = hankel1(m+1, kk * a)  # shape ()

#                 betam = (Jkm0 - Jkm1)/(Hkm0_a - Hkm1_a) # shape ???

#                 G += epsilonm * (Hkm0[:, np.newaxis] * Hkm1[np.newaxis, :]) * cos0 * cos1 * betam[np.newaxis, np.newaxis]

#         G *= (-1j / 4 / np.pi) 


#         return G

#     else:
#         raise NotImplementedError("Only 3D Green's function is implemented as of now.")

# def CylinderGreenFunction(
#     x, y, k, a,
#     dim=3,
#     Nq_prop=100,
#     mmax=32
# ):
    
#     # TODO: fix!

#     if dim != 3:
#         raise NotImplementedError

#     k = np.atleast_1d(k)
#     Nk = k.size
#     Nx = x.shape[1]
#     Ny = y.shape[1]

#     obs_r, _, obs_phi = getSphericalCoordinatesWRTx(x)
#     src_r, _, src_phi = getSphericalCoordinatesWRTx(y)

#     dz   = x[0, :, None] - y[0, None, :]          # (Nx, Ny), Note: dz is along the cylinder axis!
#     dphi = obs_phi[:, None] - src_phi[None, :]    # (Nx, Ny)

#     G = np.zeros((Nk, Nx, Ny), dtype=np.complex128)

#     # ---- m-dependent quantities ----
#     m = np.arange(mmax)                           # (m,)
#     eps = np.ones(mmax)
#     eps[1:] = 2

#     # angular dependence (m, Nx, Ny)
#     cos1 = np.cos(m[:, None, None] * dphi[None, :, :])

#     for ik, k0 in enumerate(k):

#         # ---- quadrature ----
#         maxk = 10 * k0
#         ki = np.linspace(0, maxk, Nq_prop)         # (q,)
#         w = np.full(Nq_prop, maxk / Nq_prop)       # (q,)

#         # ki = np.linspace(-maxk, maxk, Nq_prop)         # (q,)
#         # w = np.full(Nq_prop, 2 * maxk / Nq_prop)       # (q,)


#         kk = np.sqrt(k0**2 - ki**2 + 0j)            # (q,)
#         cos0 = np.cos(ki[:, None, None] * dz)      # (q, Nx, Ny)

#         # ---- beta(q, m) ----
#         kk_qm = kk[:, None]
#         m_qm = m[None, :]

#         Jm0 = jv(m_qm - 1, kk_qm * a)
#         Jm1 = jv(m_qm + 1, kk_qm * a)
#         Hm0 = hankel1(m_qm - 1, kk_qm * a)
#         Hm1 = hankel1(m_qm + 1, kk_qm * a)

#         beta = (Jm0 - Jm1) / (Hm0 - Hm1)             # (q, m)

#         # ---- Hankel terms ----
#         H_obs = hankel1(
#             m_qm[:, :, None],
#             kk_qm[:, :, None] * obs_r[None, None, :]
#         )                                           # (q, m, Nx)

#         H_src = hankel1(
#             m_qm[:, :, None],
#             kk_qm[:, :, None] * src_r[None, None, :]
#         )                                           # (q, m, Ny)

#         # ---- accumulation ----
#         G[ik] += np.sum(
#             w[:, None, None, None]
#             * eps[None, :, None, None]
#             * beta[:, :, None, None]
#             * H_obs[:, :, :, None]
#             * H_src[:, :, None, :]
#             * cos0[:, None, :, :]
#             * cos1[None, :, :, :],
#             axis=(0, 1)
#         )

#     G *= (-1j / (4 * np.pi))

#     return G

# fully complex?
def CylinderGreenFunction(
    x, y, k, a,
    dim=3,
    Nq_prop=100,
    mmax=32
):
    
    # TODO: fix!

    if dim != 3:
        raise NotImplementedError

    k = np.atleast_1d(k)
    Nk = k.size
    Nx = x.shape[1]
    Ny = y.shape[1]

    # obs_r, obs_theta, obs_phi = getSphericalCoordinatesWRTx(x)
    # src_r, src_theta, src_phi = getSphericalCoordinatesWRTx(y)

    obs_r, obs_phi, obs_x = getCylindricalCoordinatesWRTx(x)
    src_r, src_phi, src_x = getCylindricalCoordinatesWRTx(y)

    dz   = obs_x[:, None] - src_x[None, :]          # (Nx, Ny), Note: dz is along the cylinder axis!
    dphi = obs_phi[:, None] - src_phi[None, :] % (2 * np.pi)    # (Nx, Ny)

    G = np.zeros((Nk, Nx, Ny), dtype=np.complex128)

    # ---- m-dependent quantities ----
    m = np.arange(0, mmax, 1)    # (m,)  
    epsm =  np.ones(mmax)
    epsm[1:] = 2                 

    # angular dependence (m, Nx, Ny)
    cos1 = np.exp(1j* m[:, None, None] * dphi[None, :, :])

    for ik, k0 in enumerate(k):

        # ---- quadrature ----
        maxk = 10 * k0

        ki = np.linspace(0, maxk, Nq_prop)         # (q,)
        w = np.full(Nq_prop, maxk / Nq_prop)       # (q,)


        kk = np.sqrt(k0**2 - ki**2 + 0j)            # (q,)
        cos0 = np.exp(1j * ki[:, None, None] * dz)      # (q, Nx, Ny)

        # ---- beta(q, m) ----
        kk_qm = kk[:, None]
        m_qm = m[None, :]

        Jm0 = jv(m_qm , kk_qm * a)
        Hm0 = hankel1(m_qm, kk_qm * a)
        beta = Jm0 / Hm0

        # Jm0 = jv(m_qm - 1, kk_qm * a)
        # Jm1 = jv(m_qm + 1, kk_qm * a)
        # Hm0 = hankel1(m_qm - 1, kk_qm * a)
        # Hm1 = hankel1(m_qm + 1, kk_qm * a)

        # beta = (Jm0 - Jm1) / (Hm0 - Hm1)             # (q, m)

        # ---- Hankel terms ----
        H_obs = hankel1(
            m_qm[:, :, None],
            kk_qm[:, :, None] * obs_r[None, None, :]
        )                                           # (q, m, Nx)

        H_src = hankel1(
            m_qm[:, :, None],
            kk_qm[:, :, None] * src_r[None, None, :]
        )       
        
        # Create a boolean mask for all kk entries that are NOT zero
        mask = np.abs(kk) > 1e-12 # skip the zero point (pole)

        newterm = np.sum(
            w[mask, None, None, None]
            * beta[mask, :, None, None]
            * H_obs[mask, :, :, None]
            * H_src[mask, :, None, :]
            * cos0[mask, None, :, :]
            * cos1[None, :, :, :]
            * epsm[None, :, None, None],
            axis=(0, 1)  # sum over kk (axis 0) and m (axis 1)
        )

        # if np.max(np.abs(newterm)) >1e12:
        #     newterm = np.zeros_like(newterm)  # skip unstable terms
        # Apply mask along the first axis of all arrays
        G[ik] += newterm

    G *= (-1j / (4 * np.pi))

    return G


class TayloredGreen():
    def __init__(self, G_scattering, c=340., dim=3, kwargs_green={}):
        self.dim = dim
        self.G_scattering = G_scattering
        self.G_scattering_kwargs = kwargs_green
        self.G_base = FreeSpaceGreenFunction
        self.c = c  # speed of sound

    def getGreenFunction(self, x, y, k):
        G0 = self.getFreeSpaceGreen(x, y, k)
        G1 = self.getScatteringGreen(x, y, k)
        return G0 + G1
    
    def getFreeSpaceGreen(self, x, y, k):
        return self.G_base(x, y, k, dim=self.dim)

    def getScatteringGreen(self, x, y, k):
        return self.G_scattering(x, y, k, dim=self.dim, **self.G_scattering_kwargs)
    
    # plot total on 2D plane

    # get 3D directionality
    def getGradientGreen(self, x, y, k, wrt='y', eps=1e-6):

        if wrt == 'y':
            # construct 6 points around y
            # use finite differences to compute the gradient vector at y
            y = np.atleast_2d(y)
            gradG = np.zeros((3, x.shape[1], y.shape[1]), dtype=complex)

            for i in range(3):
                y_comp_plus = y.copy()
                y_comp_minus = y.copy()
                y_comp_plus[i, :] += eps
                y_comp_minus[i, :] -= eps

                G_plus = self.getGreenFunction(x, y_comp_plus, k)
                G_minus = self.getGreenFunction(x, y_comp_minus, k)

                gradG[i, :, :] = (G_plus - G_minus) / (2 * eps)

        return gradG # shape (3, Nx, Ny)
    
    def getFarFieldGradientGreen(self, k, y, eps=1e-6, R=None, Nphi=18, Ntheta=36):

        """
        getFarFieldGradientGreen
        compute the gradient of G_total in the farfield of y, assuming sources are close to (0, 0) compared to R=100/k
        :param k: wave number
        :param y:source positions, size (3, Ny)
        :param eps: spacing for computing the finite-difference gradient
        :param R: (optional) radius of the far field sphere
        :param Nphi: discretization in polar angle
        :param Ntheta: discretization in azimuth
        :return: gradG - gradient of G_total at far field points, size (3, N, Ny)
        """

        # build x as a sphere of points at a distance R >> 1/k away from (midpoint of) y
        # compute gradients wrt y at (x, y)

        if R is None:
            R = 1e6 / k  # large distance

        phi = np.linspace(0, np.pi, Nphi)
        theta = np.linspace(0 - np.pi/2, 2 * np.pi- np.pi/2, Ntheta, endpoint=False)

        Phi, Theta = np.meshgrid(phi, theta, indexing='ij')
        X = R * np.sin(Phi) * np.cos(Theta)
        Y = R * np.sin(Phi) * np.sin(Theta)
        Z = R * np.cos(Phi)

        x = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])  # shape (3, N)

        gradG = self.getGradientGreen(x, y, k, wrt='y', eps=eps)
        
        return gradG, x, Phi, Theta
    

    def plotFarFieldGradient(self, k, y, eps=1e-6, R=None, Nphi=18, Ntheta=36):
        gradG, x, Phi, Theta = self.getFarFieldGradientGreen(k, y, eps=eps, R=R, Nphi=Nphi, Ntheta=Ntheta)
        gx, gy, gz = np.real(gradG) # discard imaginary part
        RR=1.0
        a = self.G_scattering_kwargs.get('a', 0.5)
        X, Y, Z = x[0, :], x[1, :], x[2, :]
        gx, gy, gz = gx.ravel(), gy.ravel(), gz.ravel()

            # ---- vector magnitude for coloring ----
        mag = np.sqrt(gx**2 + gy**2 + gz**2)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # ---- quiver plot with magnitude coloring ----
        q = ax.quiver(
            X, Y, Z,
            gx, gy, gz,
            # np.zeros_like(gx), gy, gz,
            length=0.15,
            normalize=True,
            colors=None,
            cmap="RdBu",
            array=mag
        )

        # colorbar
        cbar = fig.colorbar(q, ax=ax, shrink=0.7, pad=0.1)
        cbar.set_label("|∇G|")

        # # ---- cylinder along x-axis ----
        # x_cyl = np.linspace(-1.2, 1.2, 100)
        # phi_cyl = np.linspace(0, 2*np.pi, 60)

        # Xc, Phic = np.meshgrid(x_cyl, phi_cyl)
        # Yc = a * np.cos(Phic)
        # Zc = a * np.sin(Phic)

        # ax.plot_surface(
        #     Xc, Yc, Zc,
        #     color="gray",
        #     alpha=0.3,
        #     linewidth=0,
        #     zorder=0
        # )

        # ---- labels ----
        ax.set_title("Far-field gradient of Green's function")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        ax.set_box_aspect([1, 1, 1])

        plt.show()
        plt.close(fig)

    def plotScatteringYZ(self, k, y, eps=1e-6, rmin=None, rmax=None, Nr=50, Ntheta=90):

        theta = np.linspace(0, 2 * np.pi, Ntheta, endpoint=False)
        R = np.linspace(rmin, rmax, Nr)
        Theta, R_grid = np.meshgrid(theta, R)
        Y = R_grid * np.cos(Theta)
        Z = R_grid * np.sin(Theta)
        X = np.zeros_like(Y)
        x = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])  # shape (3, N)
        fig, axs = plt.subplots(1, 2, figsize=(12, 5),  sharey=True,constrained_layout=True)

        G0 = self.getFreeSpaceGreen(x, y, k)
        G1 = self.getScatteringGreen(x, y, k)
        G_total = G0 + G1

        G0_reshaped = G0.reshape(Y.shape)
        G_total = G_total.reshape(Y.shape)

        eps0 = np.abs(G0_reshaped)[np.abs(G0_reshaped) > 0].min()
        eps1 = np.abs(G_total)[np.abs(G_total) > 0].min()

        # Plot G0
        im0 = axs[0].pcolormesh(
            Y, Z,
            np.real(G0_reshaped),
            shading='auto',
            cmap='viridis',
            # norm=colors.LogNorm(vmin=eps0, vmax=np.abs(G0_reshaped).max())
        )
        axs[0].set_title('G0 on the yz plane')
        axs[0].set_xlabel('y')
        axs[0].set_ylabel('z')
        A = self.G_scattering_kwargs.get('a', 0.5)
        cyl_y = A * np.cos(theta)
        cyl_z = A * np.sin(theta)
        axs[0].plot(cyl_y, cyl_z, 'k--', linewidth=2, label='Cylinder')
        axs[0].plot(y[1], y[2], 'ro', label='Source')
        axs[0].axis('equal')
        fig.colorbar(im0, ax=axs[0])

        # Plot G0 + G1
        im1 = axs[1].pcolormesh(
            Y, Z,
            np.real(G_total),
            shading='auto',
            cmap='viridis',
            # norm=colors.LogNorm(vmin=eps1, vmax=np.abs(G_total).max())
        )
        axs[1].set_title('G0+G1 on the yz plane')
        axs[1].set_xlabel('y')
        axs[1].set_ylabel('z')
        axs[1].plot(cyl_y, cyl_z, 'k--', linewidth=2, label='Cylinder')
        axs[1].plot(y[1], y[2], 'ro', label='Source')
        axs[1].axis('equal')
        fig.colorbar(im1, ax=axs[1])

        plt.show()

greenCylinder = TayloredGreen(CylinderGreenFunction, dim=3, kwargs_green={'a': 0.5})

if __name__ == "__main__":
    ysrc = np.array([[0.0, 0.0, 1.0]]).T
    RR = 0.5
    greenCylinder.plotFarFieldGradient(k=10.0, y=ysrc, R=RR, Ntheta = 2)
    gradG, x, Phi, Theta = greenCylinder.getFarFieldGradientGreen(k=10.0, y=ysrc, R=RR, Ntheta=2)
    print(np.abs(np.sum(gradG[:, :, 0] * x, axis=0))) # should be (close to) zero, if it is not, something's wrong with the scattering part
    print(np.max(np.abs(np.sum(gradG[:, :, 0] * x, axis=0))))
    # greenCylinder.plotScatteringYZ(k=10.0, y=ysrc, rmin=0.5, rmax=10.5, Nr=50, Ntheta=90)
