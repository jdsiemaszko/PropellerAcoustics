from scipy.special import jv, jve, jvp
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from Constants.helpers import getSphericalCoordinates, p_to_SPL, plot_3D_directivity, plot_2D_directivity, plot_directivity_contour

class HansonModel():

    def __init__(self, radius_m:np.ndarray,
                    axis:np.ndarray=np.array([0, 0, 1]), origin:np.ndarray=np.array([0, 0, 0]), radial:np.ndarray=None,
                  B:int=2, Omega_rads:float=1.0, rho_kgm3:float=1.2, c_mps:float = 340., nb:int = 1):

        """
        Hanson model for propeller noise propagation
        Input arrays of size (Nr+1), defined as SEGMENT EDGES, so that the segment centers are at (twist_rad[1:]+twist_rad[:-1])/2 and segment sizes are dr = np.diff(radius_m) etc.
        modes k range from 0 to Nk-1 and correspond to frequencies k*Omega
        """

        self.B = B
        self.Omega=Omega_rads
        self.rho = rho_kgm3
        self.c = c_mps # speed of sound
        self.nbeam = nb


        self.radius_e = radius_m # Nr+1
        self.r0 = radius_m[0]
        self.r1 = radius_m[-1]

        self.radius_c = (radius_m[1:] + radius_m[:-1]) / 2

        self.dr = np.diff(radius_m) # (Nr)
        self.Nr = len(self.dr)

        if len(self.radius_c) != self.Nr:
            raise ValueError("Inconsistent input sizes: seg_twist, seg_chord, seg_radius should all have size Nr")

        # orientation of the propeller in space
        self.axis = axis / np.linalg.norm(axis) # axis vector of the prop
        self.origin = origin # intersection point between axis and plane of the propeller
        if radial is None:
            # choose an arbitrary radial direction perpendicular to the axis
            if np.allclose(self.axis, np.array([1,0,0])):
                temp_vec = np.array([0,1,0])
            else:
                temp_vec = np.array([1,0,0])
            self.radial = temp_vec - np.dot(temp_vec, self.axis) * self.axis
            self.radial /= np.linalg.norm(self.radial)
        else:
            self.radial = radial / np.linalg.norm(radial) # radial vector in the prop plane, taken as the zero azimuth direction

        self.normal = np.cross(self.axis, self.radial) # normal vector, completing the right-handed coordinate system


    def getPressureRotor(self, x:np.ndarray, m:np.ndarray, Fblade:np.ndarray, multiplier:float=None):
        """
        Computing the Rotor loading noise based on loading harmonics, see Hanson & Patrzych 1993
        x:np.ndarray of shape (Nx,) -  observer position expressed in the GLOBAL CARTESIAN coordinate system
        Fblade:np.ndarray - blade loading harmonics array of size (3, Nk, Nr),
        defining the distribution of LOADING PER UNIT SPAN along the SINGLE blade, for a total of Nk modes from 0 to Nk-1!
        multiplier:float - an overall multiplier for total the pressure mode. For B blades it should be B (default behavior).

        SIGN CONVENTION:
        Fblade are the loading acting ON the blade BY the fluid
        Fblade are constructed as 1/T int_0^T F(t)e^{i*k*Omega*t}dt
        Fblade are ordered as radial, axial, tangential force along axis 0
        Fblade[0] is positive outwards, Fblade[1] is positive upstream, Fblade[2] is positive opposite to the direction of rotation.

        returns: p_mB: np.ndarray of size (Nx, Nm) - array of pressure modes at frequencies m*B*self.Omega
        at observation points x, x is also returned for convenience
        """

        if not np.all(m != 0):
            raise ValueError("m=0 is not supported")

        if multiplier is None:
            multiplier = self.B

        c0 = self.c # SoS
        Omega = self.Omega
        B = self.B
        nb = self.nbeam

        # convert observation point to cylidrical relative to the prop
        R, theta, phi = getSphericalCoordinates(
            x, self.axis, self.origin, self.radial, self.normal
        ) # each of size Nx

        radius = self.radius_c # Nr
        dr = self.dr # Nr, size of segment

        mB = m * B # Nm

        wavenumber = mB * Omega / c0 # (Nm )

        Nk = Fblade.shape[1] # shape 3, Nk, Nr
        k = np.arange(0, Nk, 1)  # array of modal orders, shape Nk, note that we assume order of Fbeam

        k = np.concatenate((-k[-1:0:-1], k)) # add the minus part!, shape (2Nk-1 -> Nk)
        Fblade = np.concatenate((np.conjugate(Fblade[:, -1:0:-1, :]), Fblade), axis=1) # minus loadings are conjugates of positive!
        # Fblade and k are now of shape (2Nk-1, Nr) with negative modes first

        # --- explicit broadcasting ---
        R_x      = R[:, None, None, None]          # (Nx, 1, 1, 1)
        theta_x = theta[:, None, None, None]       # (Nx, 1, 1, 1)
        phi_x   = phi[:, None, None, None]         # (Nx, 1, 1, 1)

        mB_m    = mB[None, :, None, None]           # (1, Nm, 1, 1)
        k_k     = k[None, None, :, None]            # (1, 1, Nk, 1)

        radius_r = radius[None, None, None, :]      # (1, 1, 1, Nr)
        dr_r     = dr[None, None, None, :]          # (1, 1, 1, Nr)

        wavenumber_m = wavenumber[None, :, None, None]  # (1, Nm, 1, 1)

        Fphi = Fblade[2, :, :][None, None, :, :] # (1, 1, Nk, Nr) NOTE: this is drag, oriented opposite to direction of travel
        Fz = Fblade[1, :, :][None, None, :, :] # (1, 1, Nk, Nr)
        Fr = Fblade[0, :, :][None, None, :, :] # (1, 1, Nk, Nr)
        
        # --- matrix construction ---
        # matrix shape: (Nx, Nm, Nk, Nr)
        matrix = np.zeros((x.shape[0], m.shape[0], Nk, radius.shape[0]), dtype=np.complex128)
        matrix = (
            - Fphi * (mB_m - k_k) / radius_r / (wavenumber_m) # positive since Fphi is positive backwards!
            + np.cos(theta_x) * Fz
        )

        matrix *= jv(mB_m - k_k, mB_m * Omega * radius_r / c0 * np.sin(theta_x))

        matrix += jvp(mB_m - k_k, mB_m * Omega * radius_r / c0 * np.sin(theta_x)) * Fr # TODO: done hastily, check if correctly implemented

        matrix *= np.exp(
           +1j * (mB_m - k_k) * (phi_x  - np.pi / 2)
        )
        
        # reduce by summing along Nk and Nr axes
        pmb = np.sum (
            matrix
              * dr_r # integrate over r! NOTE: the factor dr should be omitted if loading is given in NEWTONS
              ,
            axis=-1
        ) # integrate along the r axis, shape (Nx, Nm, Nk)
        pmb = np.sum(pmb, axis=-1) # sum along the k axis, shape (Nx, Nm)
     
        # pre-factor
        pmb *= +1j * wavenumber[None, :] * multiplier / (4 * np.pi * R[:, None]) * np.exp(+1j * wavenumber[None, :] *  R[:, None])

        return pmb, x
    
    def getThicknessNoiseRotor(self, x:np.ndarray, m:np.ndarray, chord:np.ndarray, thickness_to_chord:np.ndarray, multiplier:float=None, c0=340, rho0=1.2):
        
        if not np.all(m != 0):
            raise ValueError("m=0 is not supported")

        if multiplier is None:
            multiplier = self.B

        # convert observation point to cylidrical relative to the prop
        R, theta, phi = getSphericalCoordinates(
            x, self.axis, self.origin, self.radial, self.normal
        ) # each of size Nx

        Mtip = self.r1 * self.Omega / c0
        Mr = self.radius_c * self.Omega / c0
        kx = 2 * m[:, None] * self.B * chord[None, :] / self.r1 / 2 * Mtip / Mr[None, :] # shape Nm, Nr

        matrix = Mr[None, None, :]**2 * jv(m[None, :, None]*self.B, m[None, :, None]*self.B*Mr[None, None, :] *
                                           np.sin(theta[:, None, None])) * kx[None, :, :]**2 * thickness_to_chord[None, None, :] # Nx, Nm, Nr

        pt_mb = rho0 * c0**2 * multiplier * np.exp(1j * (m[None, :]  * self.B) * (phi[:, None]  - np.pi/2) +
                                                    1j * m[None, :] * self.B * self.Omega * R[:, None]  / c0) / 4 / np.pi / R[:, None] * np.sum(
                                                        matrix * self.dr[None, None, :], axis=-1) # integrate over r
        
        return pt_mb, x

    def getPressureStator(self, x:np.ndarray, m:np.ndarray, Fstator:np.ndarray, multiplier:float=None):
        """
        Computing the Stator loading noise based on loading harmonics, derivations based on Hanson & Patrzych 1993, though simplified for convenience
        x:np.ndarray of shape (Nx,) -  observer position expressed in the GLOBAL CARTESIAN coordinate system
        Fbeam:np.ndarray - beam loading harmonics array of size (3, Nk, Nr),
        defining the distribution of LOADING PER UNIT SPAN along the SINGLE blade, for a total of Nk modes from 0 to Nk-1!
        multiplier:float - an overall multiplier for total the pressure mode. For nb stators it should be nb (default behavior).

        SIGN CONVENTION:
        Fstator are the loading acting ON the blade BY the fluid
        Fstator are constructed as 1/T int_0^T F(t)e^{i*k*Omega*t}dt
        Fstator are ordered as radial, axial, tangential force along axis 0
        Fstator[0] is positive outwards, Fstator[1] is positive upstream, Fstator[2] is positive opposite to the direction of rotation.

        returns: p_m: np.ndarray of size (Nx, Nm) - array of pressure modes at frequencies m*self.Omega - MIND THE DIFFERENCE between this and getPressureRotor()
        x is also returned for convenience
        """
  
        if not np.all(m != 0):
            raise ValueError("m=0 is not supported")

        if multiplier is None:
            multiplier = self.nbeam

        c0 = self.c # SoS
        Omega = self.Omega

        # convert observation point to cylidrical relative to the prop
        R, theta, phi = getSphericalCoordinates(
            x, self.axis, self.origin, self.radial, self.normal
        ) # each of size Nx

        radius = self.radius_c
        dr = self.dr # Nr, size of segment

        # mB = m * B # Nm

        wavenumber = m * Omega / c0 # Nm, issue if mb = 0?

        Nk = Fstator.shape[1] # shape 3, Nk, Nr
        k = np.arange(0, Nk, 1)  # array of modal orders, shape Nk, note that we assume order of Fstator

        k = np.concatenate((-k[-1:0:-1], k)) # add the minus part!, shape (2Nk-1 -> Nk)
        Fstator = np.concatenate((np.conjugate(Fstator[:, -1:0:-1, :]), Fstator), axis=1) 

        m_int = np.asarray(m, dtype=np.int64)

        lookup = {val: i for i, val in enumerate(k)}

        Nm = len(m_int)
        Nr = Fstator.shape[2]

        # Preallocate with zeros (this automatically handles missing modes)
        Fm = np.zeros((3, Nm, Nr), dtype=np.complex_)

        # Find which requested modes exist
        valid_mask = np.array([mb in lookup for mb in m_int])

        if np.any(valid_mask):
            valid_modes = m_int[valid_mask]
            idx = np.array([lookup[mb] for mb in valid_modes])
            Fm[:, valid_mask, :] = Fstator[:, idx, :]

        Fphi = Fm[2, :, :] # Nk, Nr each
        Fz = Fm[1, :, :]


        # --- matrix construction ---
        matrix = (
            -Fphi[None, :, :] * np.sin(theta[:, None, None]) * np.sin(phi[:, None, None]) # NOTE: minus sign! see docs to see where it comes from
            + np.cos(theta[:, None, None]) * Fz[None, :, :]
        ) # shape Nx, Nm, Nr
        matrix *= np.exp(-1j * wavenumber[None, :, None] * radius[None, None, :] * np.sin(theta)[:, None, None] * np.cos(phi)[:, None, None]) # shape Nx, Nm, Nr

        # reduce by summing along Nr axis
        pm = np.sum (
            matrix
            * dr[None, None, :] # integration over r, note: we assume that Fstator is per unit span, in units N/m.
              ,
            axis=-1
        ) # integrate along the r axis

        # pre-factor
        pm *= 1j * wavenumber[None, :] / (4 * np.pi * R[:, None]) * np.exp(1j * wavenumber[None, :] * R[:, None])

        #TODO: verify interference function!
        if self.nbeam > 0:
            pm *= np.sum(
            np.exp(
                -1j * m[:, None] * 2 * np.pi / self.nbeam * 
                np.arange(0, self.nbeam, 1)[None, :]
            )
            , axis=-1
            )[None, :]

        return pm, x

    def getPolarMesh(self, R=1.0, Nphi=36, Ntheta=18, eps=0):
        # angular coordinates
        theta = np.linspace(0.0+eps, np.pi-eps, Ntheta, endpoint=True)
        phi   = np.linspace(0.0, 2.0 * np.pi, Nphi, endpoint=True)

        # 2D mesh
        theta_m, phi_m = np.meshgrid(theta, phi, indexing='ij')
        # shapes: (Ntheta, Nphi)

        # flatten
        R_arr     = np.full(theta_m.size, R)
        theta_arr = theta_m.ravel()
        phi_arr   = phi_m.ravel()

        X = R_arr * np.sin(theta_arr) * np.cos(phi_arr)
        Y = R_arr * np.sin(theta_arr) * np.sin(phi_arr)
        Z = R_arr * np.cos(theta_arr)

        # return np.array([R_arr, theta_arr, phi_arr])
        return np.array([X, Y, Z]), np.array([R_arr, theta_arr, phi_arr]), theta_m, phi_m # shape (3, Ntheta * Nphi) each
    
    def plot3Ddirectivity(self, m:float, loadings:np.ndarray, valmax=None, valmin=None, R=1.0,
                        Nphi=18, Ntheta=36, blending=0.1, title=None, fig=None, ax=None, mode='rotor', loadings_2=None,
                        chord=None, t_c=None):
        
        """
        plot a 3D directivity pattern for a given mode m

        fig, ax are matplotlib instances on which to execute the plot, they are returned after plotting

        """

        # --- observation mesh ---
        x_cart, x_spherical, Theta, Phi = self.getPolarMesh(R=R, Nphi=Nphi, Ntheta=Ntheta)

        # --- pressure / magnitude ---
        if mode=='rotor':
            pmB, _ = self.getPressureRotor(x_cart, np.array([m]).reshape(1,), Fblade=loadings, multiplier=self.B) # of shape (Nx=Ntheta*Nphi, 1)
        elif mode=='stator':
            pmB, _ = self.getPressureStator(x_cart, np.array([m * self.B]).reshape(1,), Fstator=loadings, multiplier=self.nbeam) # of shape (Nx=Ntheta*Nphi, 1)
        elif mode=='total':
            if loadings_2 is None:
                raise ValueError("For mode='total', both loading and loadings_2 (rotor and stator loadings respectively) must be provided")
            pmB_rotor, _ = self.getPressureRotor(x_cart, np.array([m]).reshape(1,), Fblade=loadings, multiplier=self.B) # of shape (Nx=Ntheta*Nphi, 1)
            if (chord is not None and t_c is not None):
                pmB_rotor_thickness, _ = self.getThicknessNoiseRotor(x_cart, np.array([m]), chord, t_c)
            pmB_stator, _ = self.getPressureStator(x_cart, np.array([m * self.B]).reshape(1,), Fstator=loadings_2, multiplier=self.nbeam) # of shape (Nx=Ntheta*Nphi, 1)
            pmB = pmB_rotor + pmB_stator + pmB_rotor_thickness
        else:
            raise ValueError("Invalid mode, should be 'rotor', 'stator', or 'total'")
        pmB = pmB[:, 0] # shape (Nx,)

        fig, ax = plot_3D_directivity(
            pmB, Theta, Phi, 
            blending=blending,
            title=f"Far-field directivity of $p_{{{int(m * self.B)}}}$",
            valmin=valmin,
            valmax=valmax,
            fig=fig,
            ax=ax
        )
        
        return fig, ax

    def plot3DdirectivityRotor(self, m:float, loadings:np.ndarray, valmax=None, valmin=None, R=1.0,
                        Nphi=18, Ntheta=36, blending=0.1, title=None, fig=None, ax=None):
        # wrapper for ploting rotor only
        return self.plot3Ddirectivity(m, loadings, valmax, valmin, R, Nphi, Ntheta, blending, title, fig, ax, mode='rotor')
    
    def plot3DdirectivityStator(self, m:float, loadings:np.ndarray, valmax=None, valmin=None, R=1.0,
                        Nphi=18, Ntheta=36, blending=0.1, title=None, fig=None, ax=None):
        # wrapper for ploting stator only
        return self.plot3Ddirectivity(m, loadings, valmax, valmin, R, Nphi, Ntheta, blending, title, fig, ax, mode='stator')
    
    def plot3DdirectivityTotal(self, m:float, loadings:np.ndarray, loadings_2:np.ndarray, valmax=None, valmin=None, R=1.0,
                        Nphi=18, Ntheta=36, blending=0.1, title=None, fig=None, ax=None, chord=None, t_c=None):
        # wrapper for ploting total directivity
        return self.plot3Ddirectivity(m, loadings=loadings, valmax=valmax, valmin=valmin, R=R,
                                       Nphi=Nphi, Ntheta=Ntheta, blending=blending, title=title,
                                         fig=fig, ax=ax, mode='total', loadings_2=loadings_2, chord=chord, t_c=t_c)

    def plot2Ddirectivity(
        self,
        m: float,
        loadings: np.ndarray,
        normalize=False,
        R=1.0,
        Ntheta=360,
        plane='yz',
        title=None,
        fig=None,
        ax=None,
        mode='rotor',
        loadings_2=None
    ):
        """
        Plot a 2D polar directivity pattern for mode m in a given plane.
        """

        theta = np.linspace(0, 2*np.pi, Ntheta, endpoint=False)

        # --- build observation ring ---
        if plane in ['yz', 'zy']:
            X = np.zeros_like(theta)
            Y = np.cos(theta)
            Z = np.sin(theta)

        elif plane in ['xy', 'yx']:
            X = np.cos(theta)
            Y = np.sin(theta)
            Z = np.zeros_like(theta)

        elif plane in ['xz', 'zx']:
            X = np.cos(theta)
            Y = np.zeros_like(theta)
            Z = np.sin(theta)

        else:
            raise ValueError(f"plane {plane} not recognized")

        dirs = np.vstack([X, Y, Z])
        x_cart = R * dirs

        # --- pressure ---
        if mode == 'rotor':
            pmB, _ = self.getPressureRotor(
                x_cart,
                np.array([m]).reshape(1,),
                Fblade=loadings,
                multiplier=self.B
            )

        elif mode == 'stator':
            pmB, _ = self.getPressureStator(
                x_cart,
                np.array([m * self.B]).reshape(1,),
                Fstator=loadings,
                multiplier=self.nbeam
            )

        elif mode == 'total':

            if loadings_2 is None:
                raise ValueError(
                    "For mode='total', both loadings and loadings_2 must be provided"
                )

            pmB_rotor, _ = self.getPressureRotor(
                x_cart,
                np.array([m]).reshape(1,),
                Fblade=loadings,
                multiplier=self.B
            )

            pmB_stator, _ = self.getPressureStator(
                x_cart,
                np.array([m * self.B]).reshape(1,),
                Fstator=loadings_2,
                multiplier=self.nbeam
            )

            pmB = pmB_rotor + pmB_stator

        else:
            raise ValueError("Invalid mode, should be 'rotor', 'stator', or 'total'")

        pmB = pmB[:, 0]

        fig, ax = plot_2D_directivity(
            p_to_SPL(pmB),
            theta,
            fig=fig,
            ax=ax,
             normalize=normalize,
            title=title if title is not None else f"Directivity of $p_{{{int(m*self.B)}}}$",
        )

        return fig, ax
    
    def plot2DdirectivityRotor(
        self,
        m: float,
        loadings: np.ndarray,
        R=1.0,
        normalize=False,
        Ntheta=360,
        plane='yz',
        title=None,
        fig=None,
        ax=None
    ):
        return self.plot2Ddirectivity(
            m=m,
            loadings=loadings,
            R=R,
            Ntheta=Ntheta,
            plane=plane,
            title=title,
            fig=fig,
            ax=ax,
            normalize=normalize,
            mode='rotor'
    )

    def plot2DdirectivityStator(
        self,
        m: float,
        loadings: np.ndarray,
        R=1.0,
        normalize=False,
        Ntheta=360,
        plane='yz',
        title=None,
        fig=None,
        ax=None
    ):
        return self.plot2Ddirectivity(
            m=m,
            loadings=loadings,
            R=R,
            Ntheta=Ntheta,
            plane=plane,
            title=title,
            fig=fig,
            ax=ax,
            normalize=normalize,
            mode='stator'
        )

    def plot2DdirectivityTotal(
        self,
        m: float,
        loadings: np.ndarray,
        loadings_2: np.ndarray,
        R=1.0,
        Ntheta=360,
        normalize=False,
        plane='yz',
        title=None,
        fig=None,
        ax=None
    ):
        return self.plot2Ddirectivity(
            m=m,
            loadings=loadings,
            R=R,
            Ntheta=Ntheta,
            plane=plane,
            title=title,
            fig=fig,
            ax=ax,
            normalize=normalize,
            mode='rotor',
            loadings_2=loadings_2
        )
    
    def plotDirectivityContour(self, m:float, loadings:np.ndarray, valmax=None, valmin=None, R=1.0,
                        Nphi=36, Ntheta=18, blending=0.1, levels=20, title=None, fig=None, ax=None, mode='rotor', loadings_2=None):
        
        """
        plot a 2D contour of the directivity pattern for a given mode m

        fig, ax are matplotlib instances on which to execute the plot, they are returned after plotting

        """

        # --- observation mesh ---
        x_cart, x_spherical, Theta, Phi = self.getPolarMesh(R=R, Nphi=Nphi, Ntheta=Ntheta)

        # --- pressure / magnitude ---
        if mode=='rotor':
            pmB, _ = self.getPressureRotor(x_cart, np.array([m]).reshape(1,), Fblade=loadings, multiplier=self.B) # of shape (Nx=Ntheta*Nphi, 1)
        elif mode=='stator':
            pmB, _ = self.getPressureStator(x_cart, np.array([m * self.B]).reshape(1,), Fstator=loadings, multiplier=self.nbeam) # of shape (Nx=Ntheta*Nphi, 1)
        elif mode=='total':
            if loadings_2 is None:
                raise ValueError("For mode='total', both loading and loadings_2 (rotor and stator loadings respectively) must be provided")
            pmB_rotor, _ = self.getPressureRotor(x_cart, np.array([m]).reshape(1,), Fblade=loadings, multiplier=self.B) # of shape (Nx=Ntheta*Nphi, 1)
            pmB_stator, _ = self.getPressureStator(x_cart, np.array([m * self.B]).reshape(1,), Fstator=loadings_2, multiplier=self.nbeam) # of shape (Nx=Ntheta*Nphi, 1)
            pmB = pmB_rotor + pmB_stator
        else:
            raise ValueError("Invalid mode, should be 'rotor', 'stator', or 'total'")
        pmB = pmB[:, 0] # shape (Nx,)

        fig, ax = plot_directivity_contour(
            np.rad2deg(Theta),np.rad2deg(Phi), pmB.reshape(Nphi,Ntheta), 
            levels=levels,
            title=title if title is not None else f"Far-field directivity of $p_{{{int(m * self.B)}}}$",
            ylabel=f'Theta [deg]',
            xlabel=f'Phi [deg]',
            fig=fig,
            ax=ax
        )
        
        return fig, ax


    def plotPressureSpectrum(self, fig, ax, x:tuple, m:np.ndarray, loadings:np.ndarray, plot_kwargs={'color' : 'k', 'marker' : 's'}):


        # --- pressure / magnitude ---
        pmB = self.getPressureRotor(x, m, Fblade=loadings, multiplier=self.B)[0] # of shape (Nx=Ntheta*Nphi, Nm)

        ax.plot(m, p_to_SPL(pmB[0]), **plot_kwargs)


        ax.minorticks_on()
        ax.grid(which='minor', alpha=0.5)
        ax.grid(which='major')
        # ax.legend()

        ax.set_xscale('log')
        ax.set_xlabel('$f^+ = \omega/B\Omega$')
        ax.set_ylabel('SPL w.r.t 20e-6 Pa')
        plt.tight_layout()

        return fig, ax
