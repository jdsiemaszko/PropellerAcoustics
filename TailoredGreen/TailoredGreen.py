import numpy as np
from scipy.special import hankel1, jv, kv
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from Constants.const import PREF
from Constants.helpers import p_to_SPL, compute_distance_matrix, plot_3D_directivity, plot_directivity_contour

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
    
    if isinstance(k, float):
        k = np.array([k])


    if dim == 3:
        G = np.exp(1j * k[:, None, None] * r[None, :, :]) / (4 * np.pi * r[None, :, :])
        return G  # shape (Nk, Nx, Ny)
    else:
        raise NotImplementedError("Only 3D Green's function is implemented as of now.")
class TailoredGreen():
    """
    Generic class for computing and plotting the Tailored Green's function
    """
    def __init__(self, dim=3):
        self.dim = dim
        self.G_base = FreeSpaceGreenFunction

    def getGreenFunction(self, x, y, k):
        """
        Arbitrary total green's function. Should be overwritten by subclasses.
        """

        G0 = self.getFreeSpaceGreen(x, y, k)
        G1 = self.getScatteringGreen(x, y, k)
        return G0 + G1
    
    def getFreeSpaceGreen(self, x, y, k):
        if isinstance(k, float):
            k = np.array([k])
        return self.G_base(x, y, k, dim=self.dim)

    def getScatteringGreen(self, x, y, k):
        if isinstance(k, float):
            k = np.array([k])
        # template for implementation
        return np.zeros((k.shape[0], x.shape[1], y.shape[1]))
    
    def getFreeSpaceGreenGradient(self, x, y, k):
        if isinstance(k, float):
            k = np.array([k])

        r = compute_distance_matrix(x, y)  # shape (Nx, Ny)
        units = (x[:, :, None] - y[:, None, :]) / r[None, :, :]  # shape (3, Nx, Ny)
        if self.dim == 3:
            G0 = self.getFreeSpaceGreen(x, y, k)  # shape (Nk, Nx, Ny)
            #Note: gradient taken w.r.t the SOURCE, i.e. y. w.r.t to x, the sign is opposite
            gradG0 = (-1j * k[None, :, None, None] + 1 / r[None, None, :, :]) * G0[None, :, :, :] * units[:, None, :, :]  # shape (3, Nk, Nx, Ny)
            return gradG0  # shape (3, Nk, Nx, Ny)

    def getScatteringGreenGradient(self, x, y, k):
        if isinstance(k, float):
            k = np.array([k])
        return np.zeros((self.dim, k.shape[0], x.shape[1], y.shape[1]))
    
    def getGradientGreenAnalytical(self, x, y, k):

        if isinstance(k, float):
            k = np.array([k])

        gG0 = self.getFreeSpaceGreenGradient(x, y, k)
        gG1 = self.getScatteringGreenGradient(x, y, k)

        return gG0 + gG1

    # get 3D directionality
    def getGradientGreenFiniteDifference(self, x, y, k, wrt='y', eps=1e-6):

        """
        Arbitrary Gradient using finite differences. Should be overwritten if an analytic form is available.
        """

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
    
    def getFarFieldx(self, k, R=None, Nphi=36, Ntheta=18, eps=np.pi/96):

        """
        getFarFieldx
        compute the farfield points at a distance R=100/k
        :param k: wave number
        :param R: (optional) radius of the far field sphere
        :param Nphi: discretization in polar angle
        :param Ntheta: discretization in azimuth
        :return: x - far field points, size (3, N=Ntheta * Nphi)
        """

        if R is None:
            R = 1e6 / k  # large distance

        theta = np.linspace(0.0+eps, np.pi-eps, Ntheta, endpoint=True)
        phi   = np.linspace(0.0, 2.0 * np.pi, Nphi, endpoint=True)

        Theta, Phi = np.meshgrid(theta, phi,
                                      indexing='ij'
                                      )
        X = R * np.sin(Theta) * np.cos(Phi)
        Y = R * np.sin(Theta) * np.sin(Phi)
        Z = R * np.cos(Theta)

        x = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])  # shape (3, N)

        return x, Theta, Phi
    
    def getFarFieldGradientGreen(self, k, y, R=None,  Nphi=36, Ntheta=18):

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

        x, Phi, Theta = self.getFarFieldx(k, R=R, Nphi=Nphi, Ntheta=Ntheta)

        gradG = self.getGradientGreenAnalytical(x, y, k)
        
        return gradG, x, Phi, Theta
    
    def getFarFieldGreen(self, k, y, R=None, Nphi=18, Ntheta=36):
        """
        getFarFieldGreen
        compute the G_total in the farfield of y, assuming sources are close to (0, 0) compared to R=100/k
        :param k: wave number
        :param y:source positions, size (3, Ny)
        :param eps: spacing for computing the finite-difference gradient
        :param R: (optional) radius of the far field sphere
        :param Nphi: discretization in polar angle
        :param Ntheta: discretization in azimuth
        :return: G - G_total at far field points, size (N, Ny)
        """

        # build x as a sphere of points at a distance R >> 1/k away from (midpoint of) y
        # compute gradients wrt y at (x, y)

        x, Theta, Phi = self.getFarFieldx(k, R=R, Nphi=Nphi, Ntheta=Ntheta)

        G = self.getGreenFunction(x, y, k)
        
        return G, x, Theta, Phi
    
    def plotFarFieldGradient(self, k, y, R=None, Nphi=36, Ntheta=18, extra_script=lambda fig, ax: None, fig=None, ax=None):
        gradG, x, Phi, Theta = self.getFarFieldGradientGreen(k, y, R=R, Nphi=Nphi, Ntheta=Ntheta)

        R = R if R is not None else (10 / k)
        for label, (gx, gy, gz) in zip([
            # "real", "imag", 
            "abs"], [
                # np.real(gradG), np.imag(gradG),
                np.abs(gradG)]):
            gx, gy, gz = np.real(gradG)
            X, Y, Z = x[0, :], x[1, :], x[2, :]

            gx, gy, gz = gx.ravel(), gy.ravel(), gz.ravel()

                # ---- vector magnitude for coloring ----
            mag = np.sqrt(gx**2 + gy**2 + gz**2)

            if fig is None or ax is None:   
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")

            # ---- quiver plot with magnitude coloring ----
            q = ax.quiver(
                X, Y, Z,
                gx, gy, gz,
                # np.zeros_like(gx), gy, gz,
                length=0.15 * R,
                normalize=True,
                colors=None,
                cmap="RdBu",
                array=mag,
                norm=colors.LogNorm(vmin=mag[mag > 0].min(), vmax=mag.max())
            )

            # colorbar
            cbar = fig.colorbar(q, ax=ax, shrink=0.7, pad=0.1)
            cbar.set_label(f"|∇G_{label}|")

            # ---- labels ----
            ax.set_title(f"Far-field gradient of Green's function ({label} part)")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_xlim([-R, R])
            ax.set_ylim([-R, R])
            ax.set_zlim([-R, R])

            ax.plot(y[0, :], y[1, :], y[2, :], 'mo', markersize=10)

            ax.set_box_aspect([1, 1, 1])

            extra_script(fig, ax)

            plt.show()
            plt.close(fig)

        return fig, ax

    def plotScatteringYZ(self, k, y, eps=1e-6, rmin=None, rmax=None, Nr=50, Ntheta=90, fig=None, axs=None):

        theta = np.linspace(0, 2 * np.pi, Ntheta, endpoint=False)
        R = np.linspace(rmin, rmax, Nr)
        Theta, R_grid = np.meshgrid(theta, R)
        Y = R_grid * np.cos(Theta)
        Z = R_grid * np.sin(Theta)
        X = np.zeros_like(Y)
        x = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])  # shape (3, N)
        if fig is None or axs is None:   

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
            # norm=colors.LinNorm(vmin=eps0, vmax=np.abs(G0_reshaped).max())
            norm=colors.CenteredNorm(halfrange = np.max(G0_reshaped) * 0.01),
            edgecolor='k',
        )
        axs[0].set_title('G0 on the yz plane')
        axs[0].set_xlabel('y')
        axs[0].set_ylabel('z')
        fig.colorbar(im0, ax=axs[0])
        axs[0].axis('equal')

        # Plot G0 + G1
        im1 = axs[1].pcolormesh(
            Y, Z,
            np.real(G_total),
            shading='auto',
            cmap='viridis',
            # norm=colors.LogNorm(vmin=eps1, vmax=np.abs(G_total).max())
            norm=colors.CenteredNorm(halfrange = np.max(G_total) * 0.01),
            edgecolor='k',
        )
        axs[1].set_title('G0+G1 on the yz plane')
        axs[1].set_xlabel('y')
        axs[1].set_ylabel('z')
        axs[1].plot(y[1], y[2], 'ro', label='Source')
        axs[1].axis('equal')
        fig.colorbar(im1, ax=axs[1])

        plt.show()
        return fig, axs

    def plot3Ddirectivity(
    self, k, y, R=None, Ntheta=18, Nphi=36, 
    extra_script=lambda fig, ax: None,
    blending=0.1,
    valmin = None, valmax=None, fig=None, ax=None
):
        G, x, Theta, Phi = self.getFarFieldGreen(
            k, y, R=R,  Ntheta=Ntheta, Nphi=Nphi
        )

        R = R if R is not None else (1e3 / k)
        fig, ax = plot_3D_directivity(
            G, Theta, Phi, 
            extra_script=extra_script,
            blending=blending,
            title=f"Far-field directivity of G",
            valmin=valmin,
            valmax=valmax,
            fig=fig,
            ax=ax
        )
        
        return fig, ax

    def plotDirectivityContour():
        pass

    def plotSelf(self, fig, ax):
        """
        empty method for plotting geometry, overwritten in subclasses
        """
        return fig, ax

if __name__=="__main__":

    tg = TailoredGreen(dim=3)
    source = np.array([[0.0], [1e-2], [0.0]])
    R = 100.0
    # tg.plotScatteringYZ(y=np.array([[0.1], [0.0], [0]]), k=np.array([10.0]), rmin=0.6, rmax=10.0)
    tg.plotFarFieldGradient(k=np.array([10.0]), y=source, R=R)
    tg.plot3Ddirectivity(k=np.array([10.0]), y=source, R=R, Nphi=36, Ntheta=18)