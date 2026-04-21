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
                  },
                  dr = None,
                  dt = None,
                  chord = None,

                  ):
        self.green = green
        self.BLH = BLH # blade loading harmonics, shape (3, Nharmonics), unit of NEWTONS!, convention: radial, axial, tangential loadings!
        Nk = BLH.shape[1]
        self.s = np.arange(0, Nk, 1) # helper array, running from 0 to len(self.BLH) - 1
        self.B = B
        self.gamma = gamma
        self.axis = axis / np.linalg.norm(axis)
        self.origin = origin
        self.radius = radius
        self.dr = dr # radial segment length of the element, used to compute thickness noise
        self.dt = dt # thickness of the element, used to compute thickness noise
        self.chord = chord # chord used in thickness noise (Glegg)
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


        self.tangential= np.cross(self.axis, self.radial) # assuming counterclockwise rotation?
        self.dipole_positions, self.dipole_angles, self.dalpha = self.getDipoleGeometry() # shape (Ndipoles, 3)
        self.NBLH = len(self.BLH)

        rad, norm, ax = getCylindricalBasis(self.dipole_angles, self.axis, self.radial, self.tangential) # shape (Ndipoles, 3)

        self.force_unit = +ax * np.cos(self.gamma) - norm * np.sin(self.gamma) # shape (Ndipoles, 3) # TODO:sign?
        self.force_unit = self.force_unit.T # (3, Ndipoles)

    def getDipoleGeometry(self):
        Ndipoles = self.numerics['Ndipoles']
        dipole_angles = np.linspace(0, 2*np.pi, Ndipoles, endpoint=False)
        dipole_positions = self.origin[:, None] + self.radius * (np.cos(dipole_angles[None, :]) *
                         self.radial[:, None] + np.sin(dipole_angles[None, :]) *
                           self.tangential[:, None]) # shape (3, Ndipoles)
        dalpha = 2 * np.pi / Ndipoles
        return dipole_positions, dipole_angles, dalpha
    
    def _rotate_loadings(self, BLH=None):
        """
        rotate loading harmonics along the rotation axis
        BLH - array of size (3, Nk), in blade-centered coordinates, optionally overwriting self.BLH
        output: array of size (3, Nk, Ndipoles) in global coordinates
        """
        angles = self.dipole_angles              # (Ndipoles,)
        axis = self.axis / np.linalg.norm(self.axis)  # ensure unit vector

        BLH = self.BLH if BLH is None else BLH  # (3, Nk)

        Nk = BLH.shape[1]
        Nd = len(angles)
        B = np.column_stack([
            self.radial,   # radial
            self.axis,     # axial
            -self.tangential   # tangential - negative because BLH is positive opposite to the rotation of prop
        ])  # shape (3,3)


        # reshape to GLOBAL cartesian frame: 
        BLH_global = B @ self.BLH   # (3, Nk)

        # Skew-symmetric matrix of axis
        K = np.array([
            [0,        -axis[2],  axis[1]],
            [axis[2],   0,       -axis[0]],
            [-axis[1],  axis[0],  0      ]
        ])

        I = np.eye(3)

        # Allocate rotation matrices (Ndipoles, 3, 3)
        R = np.zeros((Nd, 3, 3))

        for i, theta in enumerate(angles):
            R[i] = (
                I * np.cos(theta)
                + (1 - np.cos(theta)) * np.outer(axis, axis)
                + np.sin(theta) * K
            )

        # Apply rotations: result (Ndipoles, 3, Nk)
        BLH_rotated = np.einsum('nij,jk->nik', R, BLH_global)

        # Reorder to (Nk, Ndipoles, 3)
        BLH_rotated = np.transpose(BLH_rotated, (2, 0, 1))

        return BLH_rotated

    def computeLoadingVectors(self, m:np.ndarray, BLH=None):
        """
        compute the LOCAL loading vectors around the source mode
        BLH - optional overwrite of self.BLH
        """

        BLH = self.BLH if BLH is None else BLH

        # loadings = np.zeros((self.NBLH, m.shape[0], self.numerics['Ndipoles'], 3)) # (Ns, Nm, Ndipoles, 3 (ndim)))

        # expterm = np.exp(1j * self.dipole_angles[None, None, :] * (m[None, :, None] * self.B  - self.s[:, None, None]) / (m[None, :, None] * self.B * Omega)) # shape (Ns, Nm, Ndipoles)
        
        expterm = np.exp(+1j *  (m[None, :, None] * self.B  - self.s[:, None, None]) * self.dipole_angles[None, None, :])   # shape (Ns, Nm, Ndipoles)
        expterm_negative = np.exp(+1j * (m[None, :, None] * self.B + self.s[:, None, None]) * self.dipole_angles[None, None, :])  # (Ns-1, Nm, Ndipoles)
        
        # expterm = np.ones((self.BLH.shape[0],m.shape[0],self.dipole_angles.shape[0]))
        # expterm_negative = expterm

        # LEGACY (1D loading)
        # force_unit = self.force_unit.T # Ndipoles, 3
        # loadings_positive_mag = self.BLH[:, None, None] * expterm[:, :, :] # shape (Ns, Nm, Ndipoles)
        # loadings_positive = loadings_positive_mag[:,:,:, None] * force_unit[None, None, :, :] (Ns, Nm, Ndipoles, 3)
        # loadings_negative = np.conjugate(self.BLH[:, None, None, None]) * expterm_negative[:, :, :, None] * force_unit[None, None, :, :] # shape (Ns, Nm, Ndipoles, 3)
        # print(loadings.shape, expterm.shape, force_unit.shape, self.dipole_angles.shape)

        # UPDATE (3D loading)
        BLH_rotated = self._rotate_loadings(BLH=BLH) # shape (Ns, Ndipoles, 3)
        loadings_positive = expterm[:, :, :, None] * BLH_rotated[:, None, :, :]
        loadings_negative = np.conjugate(BLH_rotated[:, None, :, :]) * expterm_negative[:, :, :, None]

        # remove the zeroth term, flip along the s axis
        loadings_negative = loadings_negative[1:, :, :, :]
        loadings_negative = loadings_negative[::-1, :, :, :]

        # 4) append negative harmonics along the first dimension
        # to achieve s ranging from -Ns+1, ..., 0, ..., Ns-1
        loadings = np.concatenate([loadings_negative, loadings_positive], axis=0)  # shape (2*Ns-1, Nm, Ndipoles, 3)

        loadings *= self.dalpha / 2. / np.pi # normalize


        return loadings
    
    def getPressure(self, x:np.ndarray, Omega, m:np.ndarray, c:float = 340., gradG=None, BLH=None):

        green = self.green
        # EXPENSIVE STEP - avoid by passing gradient directly if pre-computed
        if gradG is None:
            gradG = green.getGradientGreenAnalytical(x, self.dipole_positions, m * Omega * self.B / c) # shape (3, Nm, Nx, Ny)

        return self._getPressureFromGrad(x, m, gradG, BLH=BLH)
    
    def getScatteredPressure(self, x:np.ndarray, Omega, m:np.ndarray, c:float = 340., gradG=None, surface_gradG=None, BLH=None):

        green = self.green

        # EXPENSIVE STEP - avoid by passing gradient directly if pre-computed
        if gradG is None:
            gradG = green.getScatteringGreenGradient(x, self.dipole_positions, m * Omega * self.B / c, green_grad_at_surface = surface_gradG) # shape (3, Nm, Nx, Ny)

        return self._getPressureFromGrad(x, m, gradG, BLH=BLH)

    def getDirectPressure(self, x:np.ndarray, Omega, m:np.ndarray, c:float = 340., BLH=None):
        green = self.green
        gradG = green.getFreeSpaceGreenGradient(x, self.dipole_positions, m * Omega * self.B / c) # shape (3, Nm, Nx, Ny)
        return self._getPressureFromGrad(x, m, gradG, BLH=BLH)
    
    def _getPressureFromGrad(self,x:np.ndarray, m:np.ndarray, GradG:np.ndarray, BLH=None):
        # GradG is of shape (3, Nm, Nx, Ny)
        loadings = self.computeLoadingVectors(m, BLH=BLH) # shape (2 * Ns - 1, Nm, Ny, 3) units of NEWTON
        pmB = -1.0 * np.einsum('s m y k, k m x y -> x m', loadings, GradG)
        return pmB * self.B
    
    def getScatteringGreenGradient(self, x:np.ndarray, k:np.ndarray, gradG_surface=None):
        return self.green.getScatteringGreenGradient(x, self.dipole_positions, k, green_grad_at_surface=gradG_surface)

    def getScatteringGreen(self, x:np.ndarray, k:np.ndarray, G_surface=None):
        return self.green.getScatteringGreen(x, self.dipole_positions, k, green_at_surface=G_surface)

    def getThicknessSources(self, m, Omega,  rho0=1.2):
        # compute equivalent mass terms, shape (Nm, Ny), units of Pa * m = N / m

        # sources = -1j * rho0 * m[:, None] * self.B * Omega * Omega * self.radius * self.dr * self.dt * np.exp(1j
                # * self.dipole_angles[None, :] * (m * self.B)[:, None]  ) * self.dalpha / 2 / np.pi

        sources = -rho0 * m[:, None]**2 * self.B**2 * Omega**2 * self.dr * self.dt * self.chord * np.exp(1j
                * self.dipole_angles[None, :] * (m * self.B)[:, None]  ) * self.dalpha / 2 / np.pi
        return sources


    def _getMonopolePressure(self,x:np.ndarray, m:np.ndarray, G:np.ndarray, Omega:float, rho0:float):
        # G is of shape (Nm, Nx, Ny)
        monopoles = self.getThicknessSources(m, Omega, rho0=rho0) # shape (Nm, Ny) units of Pa * m = N / m
        pmB = np.einsum('m y, m x y -> x m', monopoles, G) # shape Nx, Nm # units of Pa
        return pmB * self.B


    def getThicknessPressureDirect(self, x:np.ndarray, Omega, m:np.ndarray, c:float = 340., rho0=1.2):
        """
        compute the thickness noise, direct radiation only
        """
        green = self.green
        G = green.getFreeSpaceGreen(x, self.dipole_positions, m * Omega * self.B / c) # shape (Nm, Nx, Ny)
        
        return self._getMonopolePressure(x, m, G, Omega, rho0)
    
    def getThicknessPressureScattered(self, x:np.ndarray, Omega, m:np.ndarray, c:float = 340., rho0=1.2, G=None):
        """
        compute the thickness noise, direct radiation only
        """
        green = self.green
        if G is None: # compute the gradient (expensive)
            G = green.getScatteringGreen(x, self.dipole_positions, m * Omega * self.B / c) # shape (Nm, Nx, Ny)
        
        return self._getMonopolePressure(x, m, G, Omega, rho0)

    def getThicknessPressure(self, x:np.ndarray, Omega, m:np.ndarray, c:float = 340., rho0=1.2):
        """
        compute the thickness noise, direct radiation only
        """
        green = self.green
        G = green.getGreenFunction(x, self.dipole_positions, m * Omega * self.B / c) # shape (3, Nm, Nx, Ny)
        
        return self._getMonopolePressure(x, m, G, Omega, rho0)

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
    valmin = None, valmax=None, fig=None, ax=None,
    mode='total'):
        # if extra_script is None:
        #     extra_script = self.plotSelf

        R = R if R is not None else (1e3 / k)
        x, Theta, Phi = self.green.getFarFieldx(np.min(m) * self.B * Omega / c, Nphi=Nphi, Ntheta=Ntheta, R=R)
        if mode == 'total':
            pmB = self.getPressure(x, Omega, m, c)
        elif mode == 'direct':
            pmB = self.getDirectPressure(x, Omega, m, c)
        elif mode == 'scattered':
            pmB = self.getScatteredPressure(x, Omega, m, c)
        else:
            raise ValueError(f'mode {mode} not recognized')
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


    def plotSurfacePressure(self, m:float, Omega, c=340, valmin=None, valmax=None, fig=None, ax=None):
        if not hasattr(self.green, 'getBoundaryEvaluationPoints'):
            raise NotImplementedError('The green function must have the method getBoundaryEvaluationPoints to plot surface pressure, current instance does not match this requirement')
        pmB = self.getPressure(self.green.getBoundaryEvaluationPoints(), Omega, m, c) # shape (Npoints, Nm)
        SPL_mb = p_to_SPL(pmB)
        self.green._plotSurfaceSolution(SPL_mb, fig=fig, ax=ax, levels=20, cmap='viridis', title=None)
        return fig, ax
        

class SourceModeArray():
    def __init__(self, BLH:np.ndarray, B:int,   Omega:float, gamma:np.ndarray, axis:np.ndarray,
                  origin:np.ndarray, radius:np.ndarray, green:TailoredGreen, radial:np.ndarray=None,
                  c = 340.0,
                  rho0 = 1.2,
                numerics={
                    'Ndipoles':36
                },
                dt = None,
                chord = None
                ):
        """
        BLH - shape (3, Nk, Nr) - array of blade loading harmonics (complex magnitudes!) in units newton per meter!
        r - array of edges of radial stations (incl start and end point!) (Nr+1)
        gamma - array of blade twists, same as above (Nr+1)
        dt - segment thickness IN METERS, np.ndarray of size (Nr)
        chord - segment chord in METERS, size (Nr)
        """ 


        self.green = green
        self.BLH = BLH # blade loading harmonics, shape (Nharmonics, Nr)
        self.s = np.arange(1, len(self.BLH) + 1) # helper array
        self.B = B
        self.Omega = Omega
        self.SoS = c # speed of sound 
        self.rho0=rho0


        self.twist = gamma # Nr + 1
        self.radius = radius # Nr + 1
        self.r0 = radius[0]
        self.r1 = radius[-1]

        self.seg_twist = (gamma[1:] + gamma[:-1]) / 2
        self.seg_radius = (radius[1:] + radius[:-1]) / 2
        self.dr = np.diff(radius) # (Nr)
        self.dt = dt
        self.chord = chord # Nr
        self.Nr = len(self.seg_radius) # number of radial segments
        self.Nk = self.BLH.shape[1] 
        # self.Nr = self.BLH.shape[1] #should be the same as size of seg_radius!

        if self.BLH.shape[2] !=  self.Nr:
            raise ValueError('BLH array size does not match the number of radial stations, see docstring')


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
        self.tangential= np.cross(self.axis, self.radial)

        self.children = [None] * self.Nr # individual source modes!
        for index, (rad, twst, deltar, BLH_seg) in enumerate(zip(self.seg_radius, self.seg_twist, self.dr, np.transpose(self.BLH, axes=(2, 0, 1)))):
            self.children[index] = SourceMode(
                BLH = BLH_seg * deltar, # shape (3, Nk) - RESCALING TO NEWTONS!
                        B=self.B, gamma=twst, axis=self.axis, origin=self.origin, radius=rad, green=self.green, radial=self.radial,
                numerics=self.numerics, dr = self.dr[index], dt = self.dt[index] if dt is not None else None,
                chord = self.chord[index] if self.chord is not None else None
            ) # construct source modes at each radial station

    def updateBLH(self, BLH):
        """
        update the BLH value and propagate it to self.children (couldn't be bothered with setters)
        """
        self.BLH = BLH # change in parent, of size 3, Nk, Nr
        for index, (child, BLH_seg) in enumerate(zip(self.children, np.transpose(self.BLH, axes=(2, 0, 1)))):
            child.BLH = BLH_seg # change in child, of size 3, Nk

    def getPressure(self, x:np.ndarray, m:np.ndarray, gradG=None, BLH=None):
        if not isinstance(m, np.ndarray):
            m = np.array([m])
        if gradG is None:
            gradG = [None] * self.Nr # 
        if BLH is None:
            BLH = [None] * self.Nr # 

        print('computing pressure')
        pmB = np.zeros((x.shape[1], m.shape[0]), dtype=np.complex128) # Nx, Nm
        for index, child in enumerate(self.children):
            print(f'computing contribution of source mode {index+1} of {self.Nr}')
            pmB += child.getPressure(x, self.Omega, m, c=self.SoS, gradG=gradG[index], BLH=BLH[index])
            # pmB += child.getPressureExplicitFreeField(x, self.Omega, m, self.SoS)
        return pmB
    
    def getScatteredPressure(self, x:np.ndarray, m:np.ndarray, gradG=None, BLH=None):
        """
        gradG - optional argument to pass pre-computed gradient of the scattering green's function (recommended if dealing with same source-observer pairs)
        of shape (Nr, 3, Nm, Nx, Ny) - one gradG object per source mode, each of shape (3, Nm, Nx, Ny)
        """
        if not isinstance(m, np.ndarray):
            m = np.array([m])
        if gradG is None:
            gradG = [None] * self.Nr # 
        if BLH is None:
            BLH = [None] * self.Nr # 

        print('computing pressure')
        pmB = np.zeros((x.shape[1], m.shape[0]), dtype=np.complex128) # Nx, Nm
        for index, child in enumerate(self.children):
            print(f'computing contribution of source mode {index+1} of {self.Nr}')
            pmB += child.getScatteredPressure(x, self.Omega, m, c=self.SoS, gradG=gradG[index], BLH=BLH[index])
            # pmB += child.getPressureExplicitFreeField(x, self.Omega, m, self.SoS)
        return pmB
    

    
    def getDirectPressure(self, x:np.ndarray, m:np.ndarray, BLH=None):
        if not isinstance(m, np.ndarray):
            m = np.array([m])
        if BLH is None:
            BLH = [None] * self.Nr # 
        print('computing pressure')
        pmB = np.zeros((x.shape[1], m.shape[0]), dtype=np.complex128) # Nx, Nm
        for index, child in enumerate(self.children):
            print(f'computing contribution of source mode {index+1} of {self.Nr}')
            pmB += child.getDirectPressure(x, self.Omega, m, c=self.SoS, BLH=BLH[index])
            # pmB += child.getPressureExplicitFreeField(x, self.Omega, m, self.SoS)
        return pmB
    
    def getThicknessPressureDirect(self, x:np.ndarray, m:np.ndarray):
        if not isinstance(m, np.ndarray):
            m = np.array([m])

        print('computing direct thickness acoustic  pressure')
        pmB = np.zeros((x.shape[1], m.shape[0]), dtype=np.complex128) # Nx, Nm
        for index, child in enumerate(self.children):
            print(f'computing contribution of source mode {index+1} of {self.Nr}')
            pmB += child.getThicknessPressureDirect(x, self.Omega, m, c=self.SoS, rho0=self.rho0)
        return pmB
    

    def getThicknessPressureScattered(self, x:np.ndarray, m:np.ndarray, G=None):
        if not isinstance(m, np.ndarray):
            m = np.array([m])

        if G is None:
            G = [None] * self.Nr # 

        print('computing scattered thickness acoustic  pressure')
        pmB = np.zeros((x.shape[1], m.shape[0]), dtype=np.complex128) # Nx, Nm
        for index, child in enumerate(self.children):
            print(f'computing contribution of source mode {index+1} of {self.Nr}')
            pmB += child.getThicknessPressureScattered(x, self.Omega, m, c=self.SoS, rho0=self.rho0, G=G[index])
        return pmB
    
    def getThicknessPressure(self, x:np.ndarray, m:np.ndarray):
        if not isinstance(m, np.ndarray):
            m = np.array([m])

        print('computing thickness acoustic pressure')
        pmB = np.zeros((x.shape[1], m.shape[0]), dtype=np.complex128) # Nx, Nm
        for index, child in enumerate(self.children):
            print(f'computing contribution of source mode {index+1} of {self.Nr}')
            pmB += child.getThicknessPressure(x, self.Omega, m, c=self.SoS, rho0=self.rho0)
        return pmB

    
    def _getSurfacePressureEstFullCylinder(self, x:np.ndarray, m:np.ndarray):
        if not isinstance(m, np.ndarray):
            m = np.array([m])

        # pre-compute green for efficiency
        all_dipole_positions = np.concatenate([child.dipole_positions for child in self.children], axis=1) # shape (3, Nr * Ndipoles)
        gradG = self.green.full_cylinder_green.getGradientGreenAnalytical(x, all_dipole_positions, m * self.Omega * self.B / self.SoS) # shape (3, Nm, Nx, Ny)


        print('computing pressure')
        pmB = np.zeros((x.shape[1], m.shape[0]), dtype=np.complex128) # Nx, Nm
        for index, child in enumerate(self.children):
            print(f'computing contribution of source mode {index+1} of {self.Nr}')
            # compute the guess: use cylinder green's function
            # compute the pressure from each ring
            pmB += child._getPressureFromGrad(x, m, gradG[:, :, :, index*child.numerics['Ndipoles']:(index+1)*child.numerics['Ndipoles']])
            # pmB += child.getPressureExplicitFreeField(x, self.Omega, m, self.SoS)
        return pmB 
    
    def plotSelf(self, fig=None, ax=None, plot_normals='last'):
        
        if fig is None or ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
        
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
        if plot_normals == 'last':
            self.children[-1].plotNormals(fig, ax)
        elif plot_normals == 'all':
            for child in self.children:
                child.plotNormals(fig, ax)
        else:
            print(f'plot_normals value of {plot_normals} not recognized, not plotting any normals')
        ax.set_box_aspect([1, 1, 1])

        return fig, ax
    
    
    def plotRing(self, fig, ax):
        self.children[-1].plotRing(fig, ax)

    def plotAxis(self, fig, ax):
        self.children[-1].plotAxis(fig, ax)
        
    def plotFarFieldPressure(self, m, R=None, Nphi=36, Ntheta=18, c=340,extra_script=lambda fig, ax: None, blending=0.1,
                             valmin = None, valmax=None, fig=None, ax=None, mode='tl'):
        # if extra_script is None:
        #     extra_script = self.plotSelf



        """
        mode: one of strings {'t', 'tl', 'dl', 'sl', 'd', 'tt', 'dt', 'st', 's'} representing, in order:
        total, total loading, direct loading, scattered loading, direct, total thickness,
        direct thickness, scattered thickness, scattered noise
        """

        Omega = self.Omega
        x, Theta, Phi = self.green.getFarFieldx(np.min(m) * self.B * Omega / c, Nphi=Nphi, Ntheta=Ntheta, R=R)


        if mode == 't':
            pmB1 = self.getPressure(x, m)
            pmB2 = self.getThicknessPressure(x, m)
            pmB = pmB1 + pmB2
            modestring = 'total'
        elif mode == 'tl':
            pmB = self.getPressure(x, m)
            modestring = 'total loading'

        elif mode == 'dl':
            pmB = self.getDirectPressure(x, m)
            modestring = 'direct loading'

        elif mode == 'sl':
            pmB = self.getScatteredPressure(x, m)
            modestring = 'scattered loading'

        elif mode == 'd':
            pmB1 = self.getDirectPressure(x, m)
            pmB2 = self.getThicknessPressureDirect(x, m)
            pmB = pmB1 + pmB2

            modestring = 'direct'

        elif mode == 'tt':
            pmB = self.getThicknessPressure(x, m)
            modestring = 'total thickness'

        elif mode == 'dt':
            pmB = self.getThicknessPressureDirect(x, m)
            modestring = 'direct thickness'

        elif mode == 'st':
            pmB = self.getThicknessPressureScattered(x, m)

            modestring = 'scattered thickness'

        elif mode == 's':
            pmB1 = self.getScatteredPressure(x, m)
            pmB2 = self.getThicknessPressureScattered(x, m)
            pmB = pmB1 + pmB2
            modestring = 'scattered thickness'

        else:
            raise ValueError(f'mode {mode} not recognized')

        k = Omega/c * m
        R = R if R is not None else (1e3 / k)
        fig, ax = plot_3D_directivity(
            pmB, Theta, Phi, 
            extra_script=extra_script,
            blending=blending,
            title=f"Far-field directivity of $p_{{mB}}$ ({modestring})",
            valmin=valmin,
            valmax=valmax,
            fig=fig,
            ax=ax
        )
        return fig, ax
    
    def plotSurfacePressureFullCylinder(self, m:float, valmin=None, valmax=None, fig=None, ax=None, extend_z=None):
        if not hasattr(self.green, 'getBoundaryEvaluationPoints'):
            raise NotImplementedError('The green function must have the method getBoundaryEvaluationPoints to plot surface pressure, current instance does not match this requirement')

        eval_points = self.green.getBoundaryEvaluationPoints()

        z_edges = self.green.panel_z_edges
        z_centers = (z_edges[:-1] + z_edges[1:]) / 2
        th_edges = self.green.panel_th_edges
        th_centers = (th_edges[:-1] + th_edges[1:]) / 2
        if extend_z is not None:
            # Select axial indices inside desired range
            indices_z = np.where(
                (z_centers > extend_z[0]) & (z_centers < extend_z[1])
            )[0]



            Nazim = len(th_centers)
            Nax = len(z_centers)

            # Build global indices: each z-slice contains Nazim consecutive points
            indices = np.concatenate([
                np.arange(iz * Nazim, (iz + 1) * Nazim)
                for iz in indices_z
            ])

            # Extract filtered evaluation points
            eval_points = eval_points[:, indices]

            # Keep only selected z centers
            z_centers = z_centers[indices_z]
        pmB = self._getSurfacePressureEstFullCylinder(eval_points, m) # shape (Npoints, Nm)

        SPL_mb = p_to_SPL(pmB)
        self.green._plotSurfaceSolution(SPL_mb, z_centers, th_centers, fig=fig, ax=ax, levels=20, cmap='viridis', title=None, extent_z=extend_z, )
        return fig, ax, pmB, z_centers, th_centers


# sample class instance
from PotentialInteraction.beam_to_blade import NACA0012_T10_PIN
BLH = NACA0012_T10_PIN.getBladeLoadingHarmonics()
NRADIALSEGMENTS = 20
NHARMONICS = 40
gf = TailoredGreen(dim=3) # free-field version!
axis_prop = np.array([0.0, 0.0, 1.0]) # z-direction propeller...
origin_prop = np.array([0.0, 0.0, 0.0]) # ... at z=0
NACA0012_T10_SOURCEMODE_FF = SourceModeArray(
                        BLH=BLH, # isolate the steady component 
                        B = 2,
                        Omega=8000 / 60 * 2 * np.pi, gamma = np.deg2rad(10) * np.ones(NRADIALSEGMENTS + 1),
                        axis=axis_prop, origin=origin_prop,
                        radius=np.linspace(0.016, 0.1, NRADIALSEGMENTS + 1),
                        green=gf,
                        numerics={'Ndipoles' : 36*2},
                        c = 340.0,
                        rho0=1.2,
                        dt = 0.0809 * 0.025 * np.ones(NRADIALSEGMENTS),
                        chord = 0.025 * np.ones(NRADIALSEGMENTS),
                        )
from TailoredGreen.HalfCylinderGreen import CG_NACA0012_T10
NACA0012_T10_SOURCEMODE_HALFCYLINDER = SourceModeArray(
                        BLH=BLH, # isolate the steady component 
                        B = 2,
                        Omega=8000 / 60 * 2 * np.pi, gamma = np.deg2rad(10) * np.ones(NRADIALSEGMENTS + 1),
                        axis=axis_prop, origin=origin_prop,
                        radius=np.linspace(0.016, 0.1, NRADIALSEGMENTS + 1),
                        green=CG_NACA0012_T10,
                        numerics={'Ndipoles' : 36*2},
                        c = 340.0,
                        rho0=1.2,
                        dt = 0.0809 * 0.025 * np.ones(NRADIALSEGMENTS),
                        chord = 0.025 * np.ones(NRADIALSEGMENTS),
                        )



        
if __name__ == "__main__":
    from TailoredGreen.CylinderGreen import CylinderGreen
    
    # green = CylinderGreen(axis=np.array([1.,0,0]), radius=0.5, origin=np.array([0,0,0]))

    green = TailoredGreen() # free-field
    source = SourceMode(BLH=np.array([1.0, 0.5, 0.1]), B=2, gamma=0.05, axis=np.array([0,1.,1.]), origin=np.array([0,1.,0]), radius=0.1, green=green)

    # source.plotGeometry()
    source.plotFarFieldPressure(m=np.array([1]), Omega=100., Nphi=18, Ntheta=36)