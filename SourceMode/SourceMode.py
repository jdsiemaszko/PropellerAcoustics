import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from TailoredGreen.TailoredGreen import TailoredGreen
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from Constants.helpers import p_to_SPL, plot_3D_directivity, plot_directivity_contour, find_alpha
from PotentialInteraction.PIN import PotentialInteraction
from Hanson.far_field import HansonModel
import aerosandbox as asb
import neuralfoil as nf

def getCylindricalBasis(azimuth:np.ndarray, axis:np.ndarray, radial:np.ndarray, normal:np.ndarray):
    # get the radial,radial, and axial unit vectors given an azimuth, the axis vector, origin point, and zero-azimuth direction

    radial_loc = np.cos(azimuth[:, None]) * radial[None, :] + np.sin(azimuth[:, None]) * normal[None, :] # shape (Nazimuth, 3)

    tangential_loc = np.cross(axis, radial_loc) # shape (Nazimuth, 3) # assuming rotation in ccw direction
    axis_rep = np.tile(axis[None, :], (azimuth.shape[0], 1)) # shape (Nazimuth, 3)

    return radial_loc, tangential_loc, axis_rep

def getSearsFunction(x_c):
    return np.sqrt((1-x_c) / (1+x_c))

def _sears_antiderivative(x):
    ans = np.sqrt(1.0 - x**2) - 2.0 * np.arctan(np.sqrt((1.0 - x) / (1.0 + x))) + np.pi
    ans[x<=-1] = 0.0 # truncate for x<=-1
    ans[x>=1] = np.pi # truncate for x>=1
    return ans


def getSearsFunctionHistograms(x_c, x_c_outer):
    """
    Return the integral of the Sears function over bins whose
    edges are given by x_c_outer.
    """
    F = _sears_antiderivative(x_c_outer)
    return F[1:] - F[:-1]


class SourceMode():

    def __init__(self, BLH:np.ndarray, B:int, gamma:float, axis:np.ndarray, origin:np.ndarray, radius:float, green:TailoredGreen,
                 airfoil,
                 Omega, rho0, c0, nu,
                dr,
                  dt,
                  chord ,
                  chord_extent,
                  azimuth_offset,
                  CL,
                    parent,
                  index_r, index_layer,
                  radial:np.ndarray=None,
                  numerics={
                      'Nsources':36
                  },


                  ):
        self.numerics = numerics
        
        self.green = green
        self.BLH = BLH # blade loading harmonics, shape (3, Nharmonics), unit of NEWTONS!, convention: radial, axial, tangential loadings!
        Nk = BLH.shape[1]
        self.CL = CL # lift coefficient of the RADIAL section, which may comprise of multiple segments/source-modes

        self.s = np.arange(0, Nk, 1) # helper array, running from 0 to len(self.BLH) - 1
        self.B = B
        self.gamma = gamma
        self.axis = axis / np.linalg.norm(axis)
        self.origin = origin
        self.radius = radius
        self.dr = dr # radial segment length of the element, used to compute thickness noise
        # self.dt = dt # thickness of the element, used to compute thickness noise

        self.index_radius = index_r
        self.index_layer = index_layer
        self.parent = parent

        self.chord = chord # chord used in thickness noise (Glegg)
        self.airfoil = airfoil
        self.Omega = Omega
        self.nu = nu # kinematic viscosity
        self.rho0 = rho0
        self.SoS = c0

        self.chord_0, self.chord_1 = chord_extent
        self.chord_extent = self.chord_1 - self.chord_0
        self.azimuth_offset = azimuth_offset # loading w.r.t. this azimuth, we will need to shift it later to match the dipole positions


        # TODO: shift chord stations by c/4?
        if isinstance(dt, float): # float: assume constant dt across the chord :(
            self.dt = dt 

            # self.chord_stations = np.linspace(-3 * self.chord / 4, self.chord / 4, self.numerics.get('Nchordstations', 1000))
            self.chord_stations = np.linspace(-self.chord/2, self.chord/2, self.numerics.get('Nchordstations', 1000))

            # mind the chord in the parent frame is in opposite convention: here we go from LEADING EDGE to TRAILING EDGE
            # transformation is x -> -x to the local frame, so (c0, c1) corresponds to (-c1, -c0)
            self.where_extent = np.where(np.logical_and(self.chord_stations >= -self.chord_1, self.chord_stations < -self.chord_0))[0] # only pick the extent of this segment

            self.t_c_distribution = np.ones_like(self.chord_stations) * self.dt / self.chord
        elif isinstance(dt, np.ndarray): # array: assume a known dt distribution
            # self.chord_stations = np.linspace(-3 * self.chord / 4, self.chord / 4, dt.shape[0])
            self.chord_stations = np.linspace(-self.chord/2, self.chord/2, dt.shape[0]) # assume equidistant thickness stations over the chord length
            self.where_extent = np.where(np.logical_and(self.chord_stations >= -self.chord_1, self.chord_stations < -self.chord_0))[0] # only pick the extent of this segment

            # mean thickness of the section -  need to account for the size of the element!
            # self.dt = 1 / self.chord * np.trapezoid(dt, self.chord_stations) # mean thickness
            self.dt = 1/self.chord_extent * np.trapezoid(dt[self.where_extent], self.chord_stations[self.where_extent])
            self.t_c_distribution = dt / self.chord


        self.Nsources = numerics.get('Nsources', 36)

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
        self.dipole_positions, self.dipole_angles, self.dalpha = self.getDipoleGeometry() # shape (Nsources, 3)
        self.NBLH = len(self.BLH)

        rad, norm, ax = getCylindricalBasis(self.dipole_angles, self.axis, self.radial, self.tangential) # shape (Nsources, 3)

        self.force_unit = +ax * np.cos(self.gamma) - norm * np.sin(self.gamma) # shape (Nsources, 3) # TODO:sign?
        self.force_unit = self.force_unit.T # (3, Nsources)

    def getDipoleGeometry(self):
        Nsources = self.numerics.get('Nsources', 180)
        dipole_angles = np.linspace(0, 2*np.pi, Nsources, endpoint=False)
        dipole_positions = self.origin[:, None] + self.radius * (np.cos(dipole_angles[None, :]) *
                         self.radial[:, None] + np.sin(dipole_angles[None, :]) *
                           self.tangential[:, None]) # shape (3, Nsources)
        dalpha = 2 * np.pi / Nsources
        return dipole_positions, dipole_angles, dalpha
    
    def _rotate_loadings(self, BLH=None):
        """
        rotate loading harmonics along the rotation axis
        BLH - array of size (3, Nk), in blade-centered coordinates, optionally overwriting self.BLH
        output: array of size (3, Nk, Nsources) in global coordinates
        """
        angles = self.dipole_angles              # (Nsources,)
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
        BLH_global = B @ BLH   # (3, Nk)

        # Skew-symmetric matrix of axis
        K = np.array([
            [0,        -axis[2],  axis[1]],
            [axis[2],   0,       -axis[0]],
            [-axis[1],  axis[0],  0      ]
        ])

        I = np.eye(3)

        # Allocate rotation matrices (Nsources, 3, 3)
        R = np.zeros((Nd, 3, 3))

        for i, theta in enumerate(angles):
            R[i] = (
                I * np.cos(theta)
                + (1 - np.cos(theta)) * np.outer(axis, axis)
                + np.sin(theta) * K
            )

        # Apply rotations: result (Nsources, 3, Nk)
        BLH_rotated = np.einsum('nij,jk->nik', R, BLH_global)

        # Reorder to (Nk, Nsources, 3)
        BLH_rotated = np.transpose(BLH_rotated, (2, 0, 1))

        return BLH_rotated

    def computeLoadingVectors(self, m:np.ndarray, BLH=None):
        """
        compute the LOCAL loading vectors around the source mode
        BLH - optional overwrite of self.BLH
        """

        BLH = self.BLH if BLH is None else BLH

        expterm = np.exp(+1j *  (m[None, :, None] * self.B  - self.s[:, None, None]) * self.dipole_angles[None, None, :])   # shape (Ns, Nm, Nsources)
        expterm_negative = np.exp(+1j * (m[None, :, None] * self.B + self.s[:, None, None]) * self.dipole_angles[None, None, :])  # (Ns-1, Nm, Nsources)
        
        # UPDATE (3D loading)

        # TODO: fix!
        BLH_rotated = self._rotate_loadings(BLH=BLH) # shape (Ns, Nsources, 3)

        loadings_positive = expterm[:, :, :, None] * BLH_rotated[:, None, :, :]
        loadings_negative = np.conjugate(BLH_rotated[:, None, :, :]) * expterm_negative[:, :, :, None]

        # remove the zeroth term, flip along the s axis
        loadings_negative = loadings_negative[1:, :, :, :]
        loadings_negative = loadings_negative[::-1, :, :, :]

        # 4) append negative harmonics along the first dimension
        # to achieve s ranging from -Ns+1, ..., 0, ..., Ns-1
        loadings = np.concatenate([loadings_negative, loadings_positive], axis=0)  # shape (2*Ns-1, Nm, Nsources, 3)

        loadings *= self.dalpha / 2. / np.pi # normalize


        if self.numerics.get('CompactnessCorrection', False):
            loadings = self._getCompactnessCorrectionLoading(loadings, m)

        return loadings
    
    def _getMeanLoadingChordDistribution(self, Omega=None, nu=None, rho0=None, BLH = None):
        """
        get mean loading distribution by interfacing with neuralfoil,
        assume net sectional loading Lprime is known
        nu - kinematic viscosity, used to compute Re
        """

        BLH = self.BLH if BLH is None else BLH
        Omega = Omega if Omega is not None else self.Omega
        nu = nu if nu is not None else self.nu
        rho0 = rho0 if rho0 is not None else self.rho0

        # Lnet = np.sqrt(BLH[0, 0]**2 + BLH[1, 0]**2 + BLH[2, 0]**2)

        CL = self.CL
        Re = self.chord * Omega * self.radius / nu if nu is not None else 5e6 # pick high Re if not provided

        # find parameters corresponding to this CL
        alpha, aero = find_alpha(CL, Re, self.airfoil)

        # compute the pressure distribution (assumption: incompressible flow)
        ue_upper = np.array([aero[f'upper_bl_ue/vinf_{ind}'] for ind in range(0, 32, 1)]) # hard-coded range?
        ue_lower = np.array([aero[f'lower_bl_ue/vinf_{ind}'] for ind in range(0, 32, 1)])

        # chord stations (hard-coded in nf)
        x_c_outer = np.linspace(0, 1, 32 + 1)
        x_c = (x_c_outer[1:] + x_c_outer[:-1]) * 0.5

        cp_upper = 1 - ue_upper**2
        cp_lower = 1 - ue_lower**2

        f = cp_lower - cp_upper # pressure distribution (arbitrary scale)
        # we should have np.sum(f) * 1/32 = CL

        f = f.reshape(32)

        # interpolate at the physical chord stations, whatever the x=0 position is
        # f_interp = np.interp(self.chord_stations, x_c / 2, f)

        f_interp = np.interp((self.chord_stations-self.chord_stations[0])/self.chord, x_c, f)

        return self.chord_stations, f_interp

    def _getCompactnessCorrectionLoading(self, loading, m):
        splus = self.s
        Ns = splus.shape[0]
        sminus = -self.s[::-1]
        s = np.concatenate((sminus[:-1], splus)) # twosided, as in the function above, shape 2Ns-1

        # N = self.numerics.get('Nchordstations', 1000)
        # chord_stations = np.linspace(-self.chord/2+1e-12, self.chord/2, N)

        theta = np.linspace(0, np.pi, self.chord_stations.shape[0])  # no singularity
        u = -np.cos(theta)
        # weight = 2 * np.cos(theta / 2)**2  # comes from transformation, integrand * du/dtheta


        weight = (1+np.cos(theta)) # loading concentrated around the trailing edge (+) or leading edge (-)?
        # weight = (1+np.cos(theta)) # loading concentrated around the trailing edge (+) or leading edge (-)? 

        # Roger et al. (2006):
        """
        The [reversed Sears'] model ensures a concentration of the induced loads at the trailing edge. 
        In fact, the classical Sears’ problem, assuming concentrated unsteady loads at the leading edge,
        is valid at zero or moderate ﬂow rate. In contrast, the assumption of concentated loads
        at the trailing edge is more reliable at high ﬂow rate because the trailing edges come closer to the
        transmission shaft.
        """

        # TODO: implement PARRY with LEADING EDGE BACKSCATTERING

        # weight = np.sqrt((1-u) / (1+u)) * np.sin(theta) #integrand * du/dtheta
        # weight = np.sin(theta) * 1 / np.tan(theta/2) #integrand * du/dtheta
        # weight[0] = 0 # ignore the nan at theta=0, limit converges to 0?

        # such that np.trapezoid(weight, theta) = pi !

        phase = np.exp(
            1j * (m[None, None, :] * self.B - s[None, :, None])
            # * (self.chord / 2 * u[:, None, None]) / self.radius
            * np.arctan((self.chord / 2 * u[:, None, None] * np.cos(self.gamma)) / self.radius) # near the root, the approximation phi ~= x/r may fail!
        ) # shape Nchord stations, 2Ns-1, Nm, symmetric in s?

        where_extent = self.where_extent
        weight = weight[where_extent]
        phase = phase[where_extent, :, :]
        theta = theta[where_extent]

        factor = np.trapezoid(
            weight[:, None, None] * phase,
            theta,
            axis=0
        ) / np.trapezoid(
            weight[:, None, None],
            theta,
            axis=0
        )
        
        # shape 2*Ns-1, Nm


        # OVERWRITE THE MEAN LOADING FACTOR! - mean lift behaves different from unsteady gust responses!
        chord_stations, f0 = self._getMeanLoadingChordDistribution() # f0 of shape Nchord stations

        phase0 = np.exp(
            1j * (m[None, :] * self.B)
            * (chord_stations[:, None]) / self.radius
        ) # shape Nchord stations, Nm

        dx = np.diff(chord_stations)[0] # assumed uniform

        phase0 = phase0[where_extent, :]
        f0 = f0[where_extent]

        factor[Ns-1, :] = np.sum(
            f0[:, None] * phase0 * dx,
            axis=0
        ) / np.sum(f0 * dx) # shape Nm, mind the normalization

        # phase = np.exp(
        #     1j * (m[None, None, :] * self.B - s[None, :, None])
        #     * (chord_stations[:, None, None]) / self.radius
        # ) # shape Nc, 2*Ns-1, Nm

        # dx = np.diff(chord_stations)[0] # assumed uniform
        # factor = np.sum(
        #     f0[:, None, None] * phase * dx,
        #     axis=0
        # ) / np.sum(f0 * dx) # shape 2*Ns-1, Nm

        loading *= factor[:, :, None, None]

        return loading
    
    def getPressure(self, x:np.ndarray, Omega, m:np.ndarray, c:float = 340., gradG=None, BLH=None):

        green = self.green
        # EXPENSIVE STEP - avoid by passing gradient directly if pre-computed
        if gradG is None:
            gradG = green.getGradientGreenAnalytical(x, self.dipole_positions, m * Omega * self.B / c) # shape (3, Nm, Nx, Ny)

        return self._getPressureFromGrad(x, m, gradG, BLH=BLH)
    
    def getScatteredPressure(self, x:np.ndarray, Omega, m:np.ndarray, c:float = 340., gradG=None, gradG_surface=None, BLH=None):

        green = self.green

        # EXPENSIVE STEP - avoid by passing gradient directly if pre-computed
        if gradG is None:
            gradG = green.getScatteringGreenGradient(x, self.dipole_positions, m * Omega * self.B / c, green_grad_at_surface = gradG_surface) # shape (3, Nm, Nx, Ny)

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

        # total clumped at c/2
        # sources = -rho0 * m[:, None]**2 * self.B**2 * Omega**2 * self.dr * self.dt * self.chord * np.exp(1j
        #         * self.dipole_angles[None, :] * (m * self.B)[:, None]  ) * self.dalpha / 2 / np.pi
        
        # accounting for chordwise extent!
        sources = -rho0 * m[:, None]**2 * self.B**2 * Omega**2 * self.dr * self.dt * self.chord_extent * np.exp(1j
            * self.dipole_angles[None, :] * (m * self.B)[:, None]  ) * self.dalpha / 2 / np.pi

        if self.numerics.get('CompactnessCorrection', False):
            sources = self._getCompactnessCorrectionThickness(sources, m)

        return sources

    def _getCompactnessCorrectionThickness(self, sources, m):

        chord_stations = self.chord_stations # Nchord

        phase = np.exp(1j * m[None, :] * self.B 
                    #  * chord_stations[:, None] / self.radius
            * np.arctan(chord_stations[:, None] / self.radius * np.cos(self.gamma)) # near the root, the approximation phi ~= x/r may fail!
        )
        # apply the integral

        where_extent = self.where_extent
        chord_stations = chord_stations[where_extent]
        phase = phase[where_extent, :]
        t_c_dist = self.t_c_distribution[where_extent]

        # TODO: triple check
        factor = np.trapezoid(t_c_dist[:, None] * phase,
                        chord_stations, axis=0) / np.trapezoid(t_c_dist[:, None],
                        chord_stations, axis=0)  
                                             # shape Nm, should add up to t_c_mean if summed over the layers
            
        return sources * factor[:, None] # Nm, Ny

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
            alpha=0.1,
            # marker='x',
            zorder=10
        )

        ax.scatter(
            x, y, z,
            color="r",
            linewidth=3.0,
            # alpha=0.1,
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
                  origin:np.ndarray, radius:np.ndarray, green:TailoredGreen, 
                  airfoil,
                  radial:np.ndarray=None,
                  c0 = 340.0,
                  rho0 = 1.2,
                  nu = 14.61e-6, # m^2/s, 
                numerics={
                    'Nsources':180,
                    'Nlayers':1
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
        airfoil - an instance of aerosandbox.Airfoil or a string used to construct such, or a list of Nr such instances, one per radial station
        """ 


        self.green = green
        self.BLH = BLH # blade loading harmonics, shape (Nharmonics, Nr)
        self.s = np.arange(1, len(self.BLH) + 1) # helper array
        self.B = B
        self.Omega = Omega
        self.SoS = c0 # speed of sound 
        self.rho0=rho0
        self.nu = nu



        self.twist = gamma # Nr + 1
        self.radius = radius # Nr + 1
        self.r0 = radius[0]
        self.r1 = radius[-1]

        self.seg_twist = (gamma[1:] + gamma[:-1]) / 2
        self.seg_radius = (radius[1:] + radius[:-1]) / 2
        self.seg_chord = (chord[1:] + chord[:-1]) / 2 if chord is not None else None
        self.dr = np.diff(radius) # (Nr)
        self.dt = dt # array of size Nr or Nr, Nc, passed to children for each ind_r
        self.chord = chord # Nr
        self.Nr = len(self.seg_radius) # number of radial segments
        self.Nk = self.BLH.shape[1] 
        # self.Nr = self.BLH.shape[1] #should be the same as size of seg_radius!

        if self.BLH.shape[2] !=  self.Nr:
            raise ValueError('BLH array size does not match the number of radial stations, see docstring')
        
        if isinstance(airfoil, str):
            self.airfoil = [asb.Airfoil(airfoil)] * self.Nr
        elif isinstance(airfoil, list) or isinstance(airfoil, np.ndarray):
            self.airfoil = []
            for element in airfoil: # for each element, check if its a str or Airfoil instance
                if isinstance(airfoil, str):
                    self.airfoil.append(asb.Airfoil(airfoil))
                else:
                    self.airfoil.append(element)

        else: # if not any of the above, assume airfoil is an instance of asb.Airfoil and fill the array with it
            self.airfoil = [airfoil] * self.Nr



        self.Nsources = numerics.get('Nsources', 180) # number of dipoles in EACH source mode
        self.Nlayers = numerics.get('Nlayers', 1) # number of layers in the axial direction
        # total number of sources: Nr * Nsources * Nlayers

        # shape (Nlayers, Nr), axial offsets of each layer: ranging from -0.5 to 0.5 of c * sin(gamma): DATUM IS THE MIDCHORD
        # should default to zero with a single layer

        theta_outer = np.linspace(0, np.pi, self.Nlayers+1) # dummy variable for discretizing the chord
        theta_inner = (theta_outer[1:] + theta_outer[:-1]) / 2
        self.chord_discretization = -np.cos(theta_inner)[:, None] * self.seg_chord[None, :] / 2 # Nlayers, Nr, ranging from TRAILING EDGE to LEADING EDGE
        self.chord_discretization_edges = -np.cos(theta_outer)[:, None] * self.seg_chord[None, :] / 2 # Nlayers+1, Nr
        self.axial_offsets = self.chord_discretization * np.sin(self.seg_twist)[None, :]  # Nlayers, Nr
        self.normal_offsets = self.chord_discretization * np.cos(self.seg_twist)[None, :] 
        self.azimuthal_offsets = np.arctan(self.normal_offsets/self.seg_radius[None, :]) # shape Nlayers, Nr - change in azimuth between layers, should default to zero for 1 layer

        # np.arctan((self.chord / 2 * u[:, None, None] * np.cos(self.gamma)) / self.radius)

        # distribute BLH along the chord: assume neuralfoil + sears?
        Lnet = np.sqrt(BLH[0, 0]**2 + BLH[1, 0]**2 + BLH[2, 0]**2) # Nr
        self.CL = Lnet / 0.5 / rho0 / Omega**2 / self.seg_radius**2 / self.seg_chord # Nr, sectional CL, used to interface with neuralfoil.
        self.BLH_distributed = self.distributeBLH() # shape (3, Nk, Nr, Nlayers)

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

        self.children = [None] * self.Nr * self.Nlayers # individual source modes!

        index = 0
        for index_r, (rad, twst, deltar, BLH_seg) in enumerate(zip(self.seg_radius, self.seg_twist, self.dr, np.transpose(self.BLH, axes=(2, 0, 1)))):

            # create child source-modes, each at a given radial station
            for layer, offset in enumerate(self.axial_offsets[:, index_r]):

                origin_offset = self.origin + offset * self.axis # new attachment point: offset along the prop axis, shape 3

                self.children[index] = SourceMode(
                    
                    # BLH = BLH_seg * deltar, # shape (3, Nk) - RESCALING TO NEWTONS!

                    BLH = self.BLH_distributed[:, :, index_r, layer], # shape (3, Nk), in units of NEWTON

                    B=self.B, gamma=twst,
                    axis=self.axis,
                    origin=origin_offset,
                    radius=rad, green=self.green, radial=self.radial,
                    numerics=self.numerics, dr = self.dr[index_r], dt = self.dt[index_r] if dt is not None else None,
                    chord = self.seg_chord[index_r] if self.seg_chord is not None else None,
                    airfoil = self.airfoil[index_r],
                    c0 = c0, Omega=Omega, nu=nu, rho0=rho0,
                    azimuth_offset = self.azimuthal_offsets[layer, index_r],
                    chord_extent = (self.chord_discretization_edges[layer, index_r],
                                    self.chord_discretization_edges[layer+1, index_r]), # fraction of the chord covered by this source-mode, used for compactness corrections
                    CL = self.CL[index_r], # only for interface with neuralfoil
                    index_r = index_r, index_layer = layer, # for ordering purposes
                    parent = self
                ) # construct source modes at each radial station

                index += 1

        self.Nchildren = len(self.children)

        self.Hanson = self.getHanson()
        self.PIN = self.getPIN(self.BLH[1, 0, :], self.BLH[2, 0, :])

    def distributeBLH(self):
        """
        distribute BLH along the layers:
        - assume a Sears distribution for a flat plate for the harmonics,
        - assume mean loading from neuralfoil
        - account for

        return: BLH_distributed of shape (3, Nk, Nr, Nlayers) in units of NEWTON,
        the distributed fields should be such that summing along the layers, the total loading is recovered
        """

        # TODO: distribute with histograms rather than interpolation

        x_c = 2 * self.chord_discretization / self.seg_chord # -1 to 1, shape Nlayers, Nr
        x_c_outer = 2 * self.chord_discretization_edges / self.seg_chord # shape Nlayers+1, Nr
        # fsears = getSearsFunction(x_c) # Nlayers, Nr
        fsears = getSearsFunctionHistograms(x_c, x_c_outer)# Nlayers, Nr

        _, f0 = self._getMeanLoadingChordDistribution() # f0 of shape Nlayers, Nr

        # define loading distribution, could be improved by a better model for harmonics
        ftotal = np.zeros((self.Nlayers, self.Nr, self.Nk)) 
        ftotal[:, :, 0] = f0
        ftotal[:, :, 1:] = fsears[:, :, None]

        # BLH_distributed = np.zeros((3, self.Nk, self.Nr, self.Nlayers))
        BLH_distributed = np.einsum(
            'dkr,lrk, r ->dkrl',
            self.BLH, ftotal, self.dr
        ) # in units NEWTON

        mask = np.abs(self.BLH) > 1e-12

        num = self.BLH[..., None] * self.dr[None, None, :, None]
        den = np.sum(BLH_distributed, axis=3, keepdims=True)

        BLH_distributed[mask, :] *= (num / den)[mask, :]

        return BLH_distributed
    
    def _getMeanLoadingChordDistribution(self, ):
        """
        get mean loading distribution by interfacing with neuralfoil,
        assume net sectional loading Lprime is known
        nu - kinematic viscosity, used to compute Re

        Note: use np.sum(f_interp) / c to get CL, not np.trapezoid() due to singularity at the LE
        """

        BLH = self.BLH # shape 3, Nk, Nr
        Omega = self.Omega
        nu = self.nu
        rho0 = self.rho0

        # Lnet = np.sqrt(BLH[0, 0]**2 + BLH[1, 0]**2 + BLH[2, 0]**2) # Nr?
        # CL = Lnet / 0.5 / rho0 / Omega**2 / self.seg_radius**2 / self.dr / self.seg_chord # Nr
        CL = self.CL
        Re = self.seg_chord * Omega * self.seg_radius / nu if nu is not None else np.ones_like(CL) * 5e6 # Nr

        # find parameters corresponding to this CL

        fs = np.zeros((self.Nlayers, self.Nr)) # Nlayers, Nr
        for index, CL_val in enumerate(CL):
            alpha, aero = find_alpha(CL[index], Re[index], self.airfoil[index]) # Nr?

            # compute the pressure distribution (assumption: incompressible flow)
            ue_upper = np.array([aero[f'upper_bl_ue/vinf_{ind}'] for ind in range(0, 32, 1)]) # hard-coded discretization... (im not fixing it)
            ue_lower = np.array([aero[f'lower_bl_ue/vinf_{ind}'] for ind in range(0, 32, 1)])

            # chord stations (hard-coded in nf)
            x_c_outer = np.linspace(0, 1, 32 + 1)
            x_c = (x_c_outer[1:] + x_c_outer[:-1]) * 0.5

            cp_upper = 1 - ue_upper**2
            cp_lower = 1 - ue_lower**2

            f = cp_lower - cp_upper # pressure distribution (arbitrary scale)
            # we should have np.sum(f) * 1/32 = CL

            f = f.reshape(32)

            # interpolate at the physical chord stations, whatever the x=0 position is
            # f_interp = np.interp(self.chord_stations, x_c / 2, f)

            f_interp = np.interp((self.chord_discretization-self.chord_discretization[0, :])/self.seg_chord, x_c, f) # shape Nlayers, Nr - should work out the shapes?

            f_interp *= np.real(CL[None, :] / (np.sum(f_interp, axis=0) / self.Nlayers) ) # rescale to preserve CL!

            fs[:, index] = f_interp[:, 0] # ignore the rest

        return self.chord_discretization, f_interp

    def updateBLH(self, BLH):
        """
        update the BLH value and propagate it to self.children (couldn't be bothered with setters)
        """
        self.BLH = BLH # change in parent, of size 3, Nk, Nr

        # update CL
        Lnet = np.abs(np.sqrt(BLH[0, 0]**2 + BLH[1, 0]**2 + BLH[2, 0]**2)) # Nr
        self.CL = Lnet / 0.5 / self.rho0 / self.Omega**2 / self.seg_radius**2 / self.seg_chord # Nr, sectional CL, used to interface with neuralfoil.
        # update the distributed loads
        self.BLH_distributed = self.distributeBLH() # shape (3, Nk, Nr, Nlayers)

        for index, child  in enumerate(self.children):
            # IN NEWTONS!
            # child.BLH = BLH_seg * self.dr[index] # propagate change to children, result of shape 3, Nk
            index_r, layer = child.index_radius, child.index_layer
            child.BLH = self.BLH_distributed[:, :, index_r, layer]
            child.CL = self.CL[index_r] # update CL as well, used for neuralfoil interface

        PIN = self.getPIN(BLH[1, 0, :], BLH[2, 0, :], self.PIN.Dcylinder, self.PIN.Lcylinder)
        self.PIN = PIN # overwrite PIN as well!

        return
        
    def getPressure(self, x:np.ndarray, m:np.ndarray, gradG=None, BLH=None):
        if not isinstance(m, np.ndarray):
            m = np.array([m])
        if gradG is None:
            gradG = [None] * self.Nchildren # 
        if BLH is None:
            BLH = [None] * self.Nchildren # 

        print('computing pressure')
        pmB = np.zeros((x.shape[1], m.shape[0]), dtype=np.complex128) # Nx, Nm
        for index, child in enumerate(self.children):
            index_r = child.index_radius
            print(f'computing contribution of source mode {index+1} of {self.Nchildren}')
            pmB += child.getPressure(x, self.Omega, m, c=self.SoS, gradG=gradG[index], BLH=BLH[index] * self.dr[index] if BLH[index_r] is not None else None)
            # pmB += child.getPressureExplicitFreeField(x, self.Omega, m, self.SoS)
        return pmB
    
    def getScatteredPressure(self, x:np.ndarray, m:np.ndarray, gradG=None, BLH=None, gradG_surface=None):
        """
        gradG - optional argument to pass pre-computed gradient of the scattering green's function (recommended if dealing with same source-observer pairs)
        of shape (Nr, 3, Nm, Nx, Ny) - one gradG object per source mode, each of shape (3, Nm, Nx, Ny)
        """
        if not isinstance(m, np.ndarray):
            m = np.array([m])
        if gradG is None:
            gradG = [None] * self.Nchildren # 
        if BLH is None:
            BLH = [None] * self.Nchildren # 
        if gradG_surface is None:
            gradG_surface = [None] * self.Nchildren # 

        print('computing pressure')
        pmB = np.zeros((x.shape[1], m.shape[0]), dtype=np.complex128) # Nx, Nm
        for index, child in enumerate(self.children):
            index_r = child.index_radius
            print(f'computing contribution of source mode {index+1} of {self.Nchildren}')
            pmB += child.getScatteredPressure(x, self.Omega, m, c=self.SoS, gradG=gradG[index], BLH=BLH[index] * self.dr[index_r] if BLH[index] is not None else None, gradG_surface=gradG_surface[index])
            # pmB += child.getPressureExplicitFreeField(x, self.Omega, m, self.SoS)
        return pmB
    
    def getDirectPressure(self, x:np.ndarray, m:np.ndarray, BLH=None):
        if not isinstance(m, np.ndarray):
            m = np.array([m])
        if BLH is None:
            BLH = [None] * self.Nchildren # 
        print('computing pressure')
        pmB = np.zeros((x.shape[1], m.shape[0]), dtype=np.complex128) # Nx, Nm

        for index, child in enumerate(self.children):
            index_r = child.index_radius
            pmB += child.getDirectPressure(x, self.Omega, m, c=self.SoS, BLH= BLH[index]  * self.dr[index_r] if BLH[index] is not None else None)
            print(f'computing contribution of source mode {index+1} of {self.Nchildren}')
            # pmB += child.getPressureExplicitFreeField(x, self.Omega, m, self.SoS)
        return pmB
    
    def getThicknessPressureDirect(self, x:np.ndarray, m:np.ndarray):
        if not isinstance(m, np.ndarray):
            m = np.array([m])

        print('computing direct thickness acoustic  pressure')
        pmB = np.zeros((x.shape[1], m.shape[0]), dtype=np.complex128) # Nx, Nm
        for index, child in enumerate(self.children):
            print(f'computing contribution of source mode {index+1} of {self.Nchildren}')
            pmB += child.getThicknessPressureDirect(x, self.Omega, m, c=self.SoS, rho0=self.rho0)
        return pmB
    
    def getThicknessPressureScattered(self, x:np.ndarray, m:np.ndarray, G=None):
        if not isinstance(m, np.ndarray):
            m = np.array([m])

        if G is None:
            G = [None] * self.Nchildren # 

        print('computing scattered thickness acoustic  pressure')
        pmB = np.zeros((x.shape[1], m.shape[0]), dtype=np.complex128) # Nx, Nm
        for index, child in enumerate(self.children):
            print(f'computing contribution of source mode {index+1} of {self.Nchildren}')
            pmB += child.getThicknessPressureScattered(x, self.Omega, m, c=self.SoS, rho0=self.rho0, G=G[index])
        return pmB
    
    def getThicknessPressure(self, x:np.ndarray, m:np.ndarray):
        if not isinstance(m, np.ndarray):
            m = np.array([m])

        print('computing thickness acoustic pressure')
        pmB = np.zeros((x.shape[1], m.shape[0]), dtype=np.complex128) # Nx, Nm
        for index, child in enumerate(self.children):
            print(f'computing contribution of source mode {index+1} of {self.Nchildren}')
            pmB += child.getThicknessPressure(x, self.Omega, m, c=self.SoS, rho0=self.rho0)
        return pmB

    # def _getSurfacePressure(self, x:np.ndarray, m:np.ndarray, gradG_surface=None, BLH=None):
    #     if not isinstance(m, np.ndarray):
    #         m = np.array([m])

    #     if gradG_surface is None:
    #         gradG_surface = [None] * self.Nchildren 
    #     if BLH is None:
    #         BLH = [None] * self.Nchildren 

    #     print('computing pressure')
    #     pmB = np.zeros((x.shape[1], m.shape[0]), dtype=np.complex128) # Nx, Nm
    #     for index, child in enumerate(self.children):
    #         print(f'computing contribution of source mode {index+1} of {self.Nchildren}')
    #         pmB += child._getPressureFromGrad(x, m, gradG = gradG_surface[index], BLH=BLH[index]  * self.dr[index] if BLH[index] is not None else None)
    #     return pmB 
    
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
    
    def plotSurfacePressure(self, m:float, valmin=None, valmax=None, fig=None, ax=None, extent_z=None, gradG_surface=None, BLH=None):
        """
        gradG_surface should be of shape Nr, 3, 1, NevalPoints, Nsources

        plotting harmonic m * B of Omega
        """
        
        if not hasattr(self.green, 'getBoundaryEvaluationPoints'):
            raise NotImplementedError('The green function must have the method getBoundaryEvaluationPoints to plot surface pressure, current instance does not match this requirement')

        # eval_points = self.green.getBoundaryEvaluationPoints()
        eval_points = self.green.getBoundaryCollocPoints()

        z_edges = self.green.panel_z_edges
        z_centers = (z_edges[:-1] + z_edges[1:]) / 2
        th_edges = self.green.panel_th_edges
        th_centers = (th_edges[:-1] + th_edges[1:]) / 2




        if extent_z is not None:
            # Select axial indices inside desired range
            indices_z = np.where(
                (z_centers > extent_z[0]) & (z_centers < extent_z[1])
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

            # extract filtred gradG
            # gradG_surface = gradG_surface[:, :, :, indices, :] # along the x axis

            # Keep only selected z centers
            z_centers = z_centers[indices_z]

        # pmB = self._getSurfacePressure(eval_points, m, gradG_surface=gradG_surface, BLH=BLH) # shape (Npoints, Nm)

        # pmB = self.getPressure(eval_points, m, gradG=gradG_surface, BLH=None) # use the main function, works the same!, output of shape Neval points, 1

        pmB = self.getScatteredPressure(eval_points, m, gradG_surface=gradG_surface)
        pmB += self.getDirectPressure(eval_points, m, BLH=BLH)


        TH, PHI = np.meshgrid(th_centers, z_centers, indexing='ij')

        fig, ax = plot_directivity_contour(Theta=np.rad2deg(TH), Phi=PHI, magnitudes=pmB, fig=fig, ax=ax, ylabel=r'$\theta$ [deg]', xlabel='$z$ [m]', title=f'Surface Pressure $p_{{{m*self.B}}}$ (dB)')
        
        ax.scatter(PHI, np.rad2deg(TH), color='k', marker='x',alpha=0.25)
        
        print(f'maximum surface SPL: {np.max(p_to_SPL(pmB))} dB')
        return fig, ax
      
    def getLoading(self, Fzprime, Fphiprime, D, L, numerics=None, steady_only=False):
        """
        interface with PIN module, computing the loading and setting self.BLH to that loading
        """

        PIN = self.getPIN(Fzprime, Fphiprime, D, L, numerics)

        BLH = PIN.getBladeLoadingHarmonics()
        BLH_US = np.zeros_like(BLH)
        BLH_US[:, 1:, :] = BLH[:, 1:, :]
        BLH_S = np.zeros_like(BLH)
        BLH_S[:, 0, :] = BLH[:, 0, :]
        
        if steady_only: # use only the steady loading
            self.updateBLH(BLH_S)
        else:
            self.updateBLH(BLH) # use the full loading distribution instead.
        
        return BLH, BLH_S, BLH_US, PIN
    
    def getPIN(self, Fzprime, Fphiprime, D=0.02, L=0.02, numerics=None):
        """
        interface with PIN module, computing the loading and setting self.BLH to that loading
        """

        PIN = PotentialInteraction(
        twist_rad=self.twist,
        chord_m=self.chord,
        radius_m=self.radius,
        t_c = self.dt / self.chord if len(self.dt.shape) == 1 else (self.dt).mean(axis=1) / self.chord, # TODO: fix????
        Fzprime_Npm=Fzprime,
        Fphiprime_Npm=Fphiprime,
        B=self.B,
        # Dcylinder_m=self.green.radius * 2,
        # Lcylinder_m=-1 * self.green.origin[2],
        Dcylinder_m= D, Lcylinder_m=L,
        Omega_rads=self.Omega,
        rho_kgm3=self.rho0,
        c_mps=self.SoS,
        kmax=self.Nk-1,
        nb=1,
        numerics=numerics if numerics is not None else {'Nphi': 180, 'Nthetab': 36}
        )

        return PIN

    def getHanson(self):

        han = HansonModel(
        radius_m=self.radius, # blade radius stations [m] of size Nr + 1
        axis=self.axis, origin=self.origin,
        Omega_rads=self.Omega, # rotation speed [rad/s]
        rho_kgm3=self.rho0, # fluid density [kg/m^3]
        c_mps= self.SoS, # speed of sound [m/s]
        nb=1 # number of beams 
        )

        return han
        
    def getGreen(self):
        return self.green

if __name__ == "__main__":
    from TailoredGreen.CylinderGreen import CylinderGreen
    
    # green = CylinderGreen(axis=np.array([1.,0,0]), radius=0.5, origin=np.array([0,0,0]))

    green = TailoredGreen() # free-field
    source = SourceMode(BLH=np.array([1.0, 0.5, 0.1]), B=2, gamma=0.05, axis=np.array([0,1.,1.]), origin=np.array([0,1.,0]), radius=0.1, green=green)

    # source.plotGeometry()
    source.plotFarFieldPressure(m=np.array([1]), Omega=100., Nphi=18, Ntheta=36)