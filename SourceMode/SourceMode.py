import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from TailoredGreen.TailoredGreen import TailoredGreen
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from Constants.helpers import p_to_SPL, plot_3D_directivity

def getCylindricalBasis(azimuth:np.ndarray, axis:np.ndarray, radial:np.ndarray, normal:np.ndarray):
    # get the radial,radial, and axial unit vectors given an azimuth, the axis vector, origin point, and zero-azimuth direction

    radial_loc = np.cos(azimuth[:, None]) * radial[None, :] + np.sin(azimuth[:, None]) * normal[None, :] # shape (Nazimuth, 3)

    tangential_loc = np.cross(axis, radial_loc) # shape (Nazimuth, 3) # assuming rotation in ccw direction
    axis_rep = np.tile(axis[None, :], (azimuth.shape[0], 1)) # shape (Nazimuth, 3)

    return radial_loc, tangential_loc, axis_rep

class SourceMode():

    def __init__(self, BLH:np.ndarray, B:int, gamma:float, axis:np.ndarray, origin:np.ndarray, radius:float, green:TailoredGreen, radial:np.ndarray=None,
                  numerics={
                      'Ndipoles':36
                  }

                  ):
        self.green = green
        self.BLH = BLH # blade loading harmonics, shape (Nharmonics, Nr), unit of NEWTONS!
        self.s = np.arange(0, len(self.BLH)) # helper array
        self.B = B
        self.gamma = gamma
        self.axis = axis / np.linalg.norm(axis)
        self.origin = origin
        self.radius = radius
        self.numerics=numerics
        if radial is None:
            # choose an arbitrary radial direction perpendicular to the axis
            if np.allclose(self.axis, np.array([1,0,0])):
                temp_vec = np.array([0,1,0])
            else:
                temp_vec = np.array([1,0,0])
            self.radial = temp_vec - np.dot(temp_vec, self.axis) * self.axis
            self.radial /= np.linalg.norm(self.radial)
        else:
            self.radial = radial / np.linalg.norm(radial) # radial vector of the cylinder, taken as the zero azimuth direction


        self.normal = np.cross(self.axis, self.radial) # assuming counterclockwise rotation?
        self.dipole_positions, self.dipole_angles, self.dalpha = self.getDipoleGeometry() # shape (Ndipoles, 3)
        self.NBLH = len(self.BLH)

        rad, norm, ax = getCylindricalBasis(self.dipole_angles, self.axis, self.radial, self.normal) # shape (Ndipoles, 3)

        self.force_unit = +ax * np.cos(self.gamma) - norm * np.sin(self.gamma) # shape (Ndipoles, 3) # TODO:sign?
        self.force_unit = self.force_unit.T # (3, Ndipoles)

    def getDipoleGeometry(self):
        Ndipoles = self.numerics['Ndipoles']
        dipole_angles = np.linspace(0, 2*np.pi, Ndipoles, endpoint=False)
        dipole_positions = self.origin[:, None] + self.radius * (np.cos(dipole_angles[None, :]) *
                         self.radial[:, None] + np.sin(dipole_angles[None, :]) *
                           self.normal[:, None]) # shape (3, Ndipoles)
        dalpha = 2 * np.pi / Ndipoles
        return dipole_positions, dipole_angles, dalpha

    def computeLoadingVectors(self, m:np.ndarray):

        # loadings = np.zeros((self.NBLH, m.shape[0], self.numerics['Ndipoles'], 3)) # (Ns, Nm, Ndipoles, 3 (ndim)))

        # expterm = np.exp(1j * self.dipole_angles[None, None, :] * (m[None, :, None] * self.B  - self.s[:, None, None]) / (m[None, :, None] * self.B * Omega)) # shape (Ns, Nm, Ndipoles)
        
        expterm = np.exp(-1j *  (m[None, :, None] * self.B  - self.s[:, None, None]) * self.dipole_angles[None, None, :])   # shape (Ns, Nm, Ndipoles)
        expterm_negative = np.exp(-1j * (m[None, :, None] * self.B + self.s[:, None, None]) * self.dipole_angles[None, None, :])  # (Ns, Nm, Ndipoles)
        
        # expterm = np.ones((self.BLH.shape[0],m.shape[0],self.dipole_angles.shape[0]))
        # expterm_negative = expterm

        force_unit = self.force_unit.T # Ndipoles, 3

        loadings_positive_mag = self.BLH[:, None, None] * expterm[:, :, :] # shape (Ns, Nm, Ndipoles)
        loadings_positive = loadings_positive_mag[:,:,:, None] * force_unit[None, None, :, :]

        loadings_negative = np.conjugate(self.BLH[:, None, None, None]) * expterm_negative[:, :, :, None] * force_unit[None, None, :, :] # shape (Ns, Nm, Ndipoles, 3)
        # print(loadings.shape, expterm.shape, force_unit.shape, self.dipole_angles.shape)


        # flip the s-axis, remove the zeroth term
        loadings_negative = loadings_negative[::-1, :, :, :]
        loadings_negative = loadings_negative[1:, :, :, :]

        # 4) append negative harmonics along the first dimension
        loadings = np.concatenate([loadings_negative, loadings_positive], axis=0)  # shape (2*Ns-1, Nm, Ndipoles, 3)

        # TODO: CHECK
        # loadings = loadings_positive

        loadings *= self.dalpha / 2. / np.pi # normalize
        return loadings
    
    def getPressure(self, x:np.ndarray, Omega, m:np.ndarray, c:float = 340.):
        green = self.green

        loadings = self.computeLoadingVectors(m) # shape (2 * Ns - 1, Nm, Ny, 3) units of NEWTON
        gradG = green.getGradientGreenAnalytical(x, self.dipole_positions, m * Omega * self.B / c) # shape (3, Nm, Nx, Ny)

        # print(loadings.shape, gradG.shape)
        pmB = -1.0 * np.einsum('s m y k, k m x y -> x m', loadings, gradG)

        return pmB  * self.B
    
    # def getPressureExplicitFreeField(self, x:np.ndarray, Omega, m:np.ndarray, c:float = 340.):
    #     loadings = self.computeLoadingVectors(m)  #Ns, Nm, Ny, 3
    #     loading_axial = np.einsum('snyc, c -> sny', loadings, self.axis, optimize=True)
    #     loading_mag = loading_axial / np.cos(self.gamma) # Ns, Nm Ny

    #     # spherical coords
    #     r, theta, phi = getPolarFromCylindrical(
    #         x, self.origin, self.axis, self.radial, self.normal
    #     )   # (Nx,)

    #     # pairwise distances
    #     r_alpha = np.linalg.norm(
    #         x[:, :, None] - self.dipole_positions[:, None, :],
    #         axis=0
    #     )   # (Nx, Ny)

    #     kmB = m * Omega * self.B / c        # (Nm,)

    #     # --- broadcast helpers
    #     r_     = r[:, None]
    #     theta_ = theta[:, None]
    #     phi_   = phi[:, None]
    #     alpha_ = self.dipole_angles[None, :]

    #     # scalar angular term (Nx, Ny)
    #     ang = (
    #         np.sin(self.gamma) * np.sin(theta_) * np.sin(phi_ - alpha_)
    #         - np.cos(self.gamma) * np.cos(theta_)
    #     )

    #     # geometric scalar kernel (Nx, Ny)
    #     geom = r_ / (4 * np.pi) / r_alpha**3 * ang #Nx, Ny

    #     # phase term (Nx, Ny, Nm)
    #     phase = np.exp(1j * r_alpha[:, :, None] * kmB[None, None, :]) * (1 - 1j * kmB[None, None, :] * r_alpha[:, :, None])

    #     # ---------------------------------------------------
    #     pmB = np.einsum(
    #         'xy, xym, kmy -> xm',
    #         geom,
    #         phase,
    #         loading_mag,
    #         optimize=True
    #     )

    #     return pmB * self.B

    def plotGeometry(self):
        green = self.green
        fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
        if green is not None:
            green.plotSelf(fig, ax)
        ax.scatter(self.dipole_positions[0], self.dipole_positions[1], self.dipole_positions[2], c='r', s=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1,1,1])
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_zlim([-1,1])
        plt.show()
        plt.close(fig)

    def plotFarFieldPressure(self, m, Omega, R=None, Nphi=36, Ntheta=18,
    c=340,     extra_script=lambda fig, ax: None, blending=0.1,
    valmin = None, valmax=None, fig=None, ax=None):
        # if extra_script is None:
        #     extra_script = self.plotSelf

        R = R if R is not None else (1e3 / k)
        x, Theta, Phi = self.green.getFarFieldx(np.min(m) * self.B * Omega / c, Nphi=Nphi, Ntheta=Ntheta, R=R)

        pmB = self.getPressure(x, Omega, m)
        k = Omega / c * m
        fig, ax = plot_3D_directivity(
            pmB, Theta, Phi, 
            extra_script=extra_script,
            blending=blending,
            title=f"Far-field directivity of $p_{{mB}}$",
            valmin=valmin,
            valmax=valmax,
            fig=fig,
            ax=ax
        )
        
        return fig, ax

    def plotSelf(self, fig, ax):
        self.green.plotSelf(fig, ax)
        axis = self.axis / np.linalg.norm(self.axis)
        e0 = self.radial / np.linalg.norm(self.radial)
        e1 = np.cross(axis, e0)   # completes right-handed basis

        R = self.radius

        zmin, zmax = ax.get_xlim()

        self.plotRing(fig, ax)
        self.plotNormals(fig, ax)
        self.plotAxis(fig, ax)
        
        ax.set_box_aspect([1, 1, 1])

    def plotAxis(self, fig, ax):
        axis_line = np.array([-3 * self.radius * self.axis, 3* self.radius * self.axis])
        ax.plot(
                axis_line[:, 0] + self.origin[0],
                axis_line[:, 1] + self.origin[1],
                axis_line[:, 2] + self.origin[2],
                color="r",
                linestyle="--",
                linewidth=3.0,
                alpha=1.0,
                zorder=9   
            )

    def plotRing(self, fig, ax):
        axis = self.axis / np.linalg.norm(self.axis)
        e0 = self.radial / np.linalg.norm(self.radial)
        e1 = np.cross(axis, e0)   # completes right-handed basis

        R = self.radius

        zmin, zmax = ax.get_xlim()

        # angular parameter
        # phi = np.linspace(0, 2*np.pi, 36)

        # ring = (
            # R * np.cos(phi)[:, None] * e0[None, :]
        # + R * np.sin(phi)[:, None] * e1[None, :]
        # ) + self.origin[None, :]
        x, y, z = self.dipole_positions[0], self.dipole_positions[1], self.dipole_positions[2]
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        z = np.append(z, z[0])

        ax.plot(
            x, y, z,
            color="r",
            linewidth=3.0,
            alpha=0.5,
            marker='x',
            zorder=10

        )

    def plotNormals(self, fig, ax):

        x, y, z = self.dipole_positions[0], self.dipole_positions[1], self.dipole_positions[2]
        u, v, w = self.force_unit[0], self.force_unit[1], self.force_unit[2]
        if hasattr(self, 'force_unit') and hasattr(self, 'dipole_positions'):
            ax.quiver(
                x, y, z, u, v, w,
                length=self.radius / 2.,      # adjust arrow length to scale properly
                normalize=True,  # ensures arrows are unit vectors
                color='b',
                arrow_length_ratio=0.2,
                linewidth=1.5
            )
        

class SourceModeArray():
    def __init__(self, BLH:np.ndarray, B:int,   Omega:float, gamma:np.ndarray, axis:np.ndarray,
                  origin:np.ndarray, radius:np.ndarray, green:TailoredGreen, radial:np.ndarray=None,
                  c = 340.0,
                numerics={
                    'Ndipoles':36
                }
                ):
        """
        BLH - shape (Nk, Nr) - array of blade loading harmonics (complex magnitudes!) in units newton per meter!
        r - array of edges of radial stations (incl start and end point!) (Nr+1)
        gamma - array of blade twists, same as above (Nr+1)

        """ 


        self.green = green
        self.BLH = BLH # blade loading harmonics, shape (Nharmonics)
        self.s = np.arange(1, len(self.BLH) + 1) # helper array
        self.B = B
        self.Omega = Omega
        self.SoS = c # speed of sound 


        self.twist = gamma # Nr + 1
        self.radius = radius # Nr + 1
        self.r0 = radius[0]
        self.r1 = radius[-1]

        self.seg_twist = (gamma[1:] + gamma[:-1]) / 2
        self.seg_radius = (radius[1:] + radius[:-1]) / 2
        self.dr = np.diff(radius) # (Nr)
        self.Nr = len(self.seg_radius) # number of radial segments
        self.Nk = self.BLH.shape[0] 
        # self.Nr = self.BLH.shape[1] #should be the same as size of seg_radius!

        if self.BLH.shape[1] !=  self.Nr:
            raise ValueError('BLH array does not match the number of radial stations, see docstring')


        self.axis = axis
        self.origin = origin
        self.numerics=numerics
        if radial is None:
            # choose an arbitrary radial direction perpendicular to the axis
            if np.allclose(self.axis, np.array([1,0,0])):
                temp_vec = np.array([0,1,0])
            else:
                temp_vec = np.array([1,0,0])
            self.radial = temp_vec - np.dot(temp_vec, self.axis) * self.axis
            self.radial /= np.linalg.norm(self.radial)
        else:
            self.radial = radial # radial vector of the cylinder, taken as the zero azimuth direction
        self.normal = np.cross(self.axis, self.radial)

        self.children = [None] * self.Nr # individual source modes!
        for index, (rad, twst, deltar, BLH_seg) in enumerate(zip(self.seg_radius, self.seg_twist, self.dr, self.BLH.T)):
            self.children[index] = SourceMode(
                BLH = BLH_seg * deltar, # shape (Nk) - RESCALING TO NEWTONS!
                        B=self.B, gamma=twst, axis=self.axis, origin=self.origin, radius=rad, green=self.green, radial=self.radial,
                numerics=self.numerics
            ) # construct source modes at each radial station

    def getPressure(self, x:np.ndarray, m:np.ndarray):
        print('computing pressure')
        pmB = np.zeros((x.shape[1], m.shape[0]), dtype=np.complex128) # Nx, Nm
        for index, child in enumerate(self.children):
            print(f'computing contribution of source mode {index+1} of {self.Nr}')
            pmB += child.getPressure(x, self.Omega, m, c=self.SoS)
            # pmB += child.getPressureExplicitFreeField(x, self.Omega, m, self.SoS)
        return pmB
    
    def plotSelf(self, fig, ax):
        for child in self.children:
            child.plotRing(fig, ax)

        self.green.plotSelf(fig, ax)
        axis = self.axis / np.linalg.norm(self.axis)
        e0 = self.radial / np.linalg.norm(self.radial)
        e1 = np.cross(axis, e0)   # completes right-handed basis

        R = self.radius

        axis_line = np.array([-3 * self.r1 * axis, 3* self.r1 * axis])
        ax.plot(
                axis_line[:, 0] + self.origin[0],
                axis_line[:, 1] + self.origin[1],
                axis_line[:, 2] + self.origin[2],
                color="r",
                linestyle="--",
                linewidth=3.0,
                alpha=0.5
            )
        self.children[-1].plotNormals(fig, ax)
        ax.set_box_aspect([1, 1, 1])

    def plotRing(self, fig, ax):
        self.children[-1].plotRing(fig, ax)

    def plotAxis(self, fig, ax):
        self.children[-1].plotAxis(fig, ax)
        
    def plotFarFieldPressure(self, m, R=None, Nphi=36, Ntheta=18, c=340,extra_script=lambda fig, ax: None, blending=0.1,
                             valmin = None, valmax=None, fig=None, ax=None):
        # if extra_script is None:
        #     extra_script = self.plotSelf

        Omega = self.Omega
        x, Theta, Phi = self.green.getFarFieldx(np.min(m) * self.B * Omega / c, Nphi=Nphi, Ntheta=Ntheta, R=R)

        pmB = self.getPressure(x, m)
        k = Omega/c * m
        R = R if R is not None else (1e3 / k)
        fig, ax = plot_3D_directivity(
            pmB, Theta, Phi, 
            extra_script=extra_script,
            blending=blending,
            title=f"Far-field directivity of $p_{{mB}}$",
            valmin=valmin,
            valmax=valmax,
            fig=fig,
            ax=ax
        )
        return fig, ax

        
if __name__ == "__main__":
    from TailoredGreen.CylinderGreen import CylinderGreen
    
    # green = CylinderGreen(axis=np.array([1.,0,0]), radius=0.5, origin=np.array([0,0,0]))

    green = TailoredGreen() # free-field
    source = SourceMode(BLH=np.array([1.0, 0.5, 0.1]), B=2, gamma=0.05, axis=np.array([0,1.,1.]), origin=np.array([0,1.,0]), radius=0.1, green=green)

    # source.plotGeometry()
    source.plotFarFieldPressure(m=np.array([1]), Omega=100., Nphi=18, Ntheta=36)