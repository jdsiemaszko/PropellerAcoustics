from scipy.special import jv
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from Constants.helpers import getSphericalCoordinates, p_to_SPL, plot_3D_directivity

class HansonModel():

    def __init__(self, twist_rad:np.ndarray, chord_m:np.ndarray, radius_m:np.ndarray, loadings_Npm:np.ndarray,
                    axis:np.ndarray, origin:np.ndarray, radial:np.ndarray=None,
                  B:int=2, Omega_rads:float=1.0, rho_kgm3:float=1.0, c_mps:float = 340., nb:int = 1):

        """
        Hanson model for propeller noise propagation
        Input arrays of size (Nr+1), defined as SEGMENT EDGES, so that the segment centers are at (twist_rad[1:]+twist_rad[:-1])/2 and segment sizes are dr = np.diff(radius_m) etc.
        loadings: input array of size (Nk, Nr), defining the distribution of LOADING PER UNIT SPAN along the blade, for a total of Nk modes!
        modes k range from 0 to Nk-1 and correspond to frequencies k*Omega
        """

        self.B = B
        self.Omega=Omega_rads
        self.rho = rho_kgm3
        self.c = c_mps # speed of sound
        if nb!=1:
            raise ValueError("WARNING: case nb>1 not implemented yet!")
        self.nbeam = nb


        self.twist_e = twist_rad # Nr
        self.chord_e = chord_m # Nr
        self.radius_e = radius_m # Nr
        self.r0 = radius_m[0]
        self.r1 = radius_m[-1]

        self.twist_c = (twist_rad[1:] + twist_rad[:-1]) / 2
        self.chord_c = (chord_m[1:] + chord_m[:-1]) / 2
        self.radius_c = (radius_m[1:] + radius_m[:-1]) / 2

        self.loadings = loadings_Npm # size (Nk, Nr), 

        self.dr = np.diff(radius_m) # (Nr -1)

        self.Nk, self.Nr = loadings_Npm.shape

        if len(self.twist_c) != self.Nr or len(self.chord_c) != self.Nr or len(self.radius_c) != self.Nr:
            raise ValueError("Inconsistent input sizes: seg_twist, seg_chord, seg_radius should all have size Nr")

        self.k = np.arange(0, self.Nk, 1) # array of modal orders

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


    def getPressureRotor(self, x:np.ndarray, m:np.ndarray, multiplier:float=None):
        """
        Generic function for computing the hanson formulation of noise for rotors
        x is expressed in the GLOBAL coordinate system

        multiplier is an overall multiplier for total the pressure mode. For B blades it should be B, for one stator/beam it should be 1.

        returns: p_mB of size (Nx, Nm) - array of pressure modes m*B at observation points x, x is returned for convenience
        """
        if not np.all(m != 0):
            raise ValueError("m=0 is not supported")

        if multiplier is None:
            multiplier = self.B



        c0 = self.c # SoS
        Omega = self.Omega
        Fblade = self.loadings # Nk, Nr
        B = self.B
        nb =self.nbeam

        # convert observation point to cylidrical relative to the prop
        R, theta, phi = getSphericalCoordinates(
            x, self.axis, self.origin, self.radial, self.normal
        ) # each of size Nx

        radius, twist, chord = self.radius_c, self.twist_c, self.chord_c # all of size # Nr
        dr = self.dr # Nr, size of segment

        mB = m * B # Nm

        wavenumber = mB * Omega / c0 # (Nm )

        k = self.k * nb # Nk multiplied by the number of beams! (TODO: check?)

        k = np.concatenate((-k[-1:0:-1], k)) # add the minus part!, shape (2Nk-1 -> Nk)
        Fblade = np.concatenate((np.conjugate(Fblade[-1:0:-1]), Fblade), axis=0) # minus loadings are conjugates of positive!
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

        Fphi = np.sin(twist)[None, None, None, :] * Fblade[None, None, :, :] # (1, 1, Nk, Nr) NOTE: this is drag, oriented opposite to direction of travel
        Fz = np.cos(twist)[None, None, None, :] * Fblade[None, None, :, :] # (1, 1, Nk, Nr)
        
        
        # --- matrix construction ---
        # matrix shape: (Nx, Nm, Nk, Nr)
        matrix = (
            + Fphi * (mB_m - k_k) / radius_r / (wavenumber_m) # positive since Fphi is positive backwards!
            + np.cos(theta_x) * Fz
        )

        matrix *= jv(mB_m - k_k, mB_m * Omega * radius_r / c0 * np.sin(theta_x))

        matrix = matrix.astype(np.complex128)

        matrix *= np.exp(
           -1j * (mB_m - k_k) * (phi_x  + np.pi / 2)
            +1j * (mB_m) * Omega * R_x / c0       
        )
        
        # reduce by summing along Nk and Nr axes
        pmb = np.sum (
            matrix
              * dr_r # integrate over r! NOTE: this should be omitted if loading is given in NEWTONS
              ,
            axis=-1
        ) # integrate along the r axis, shape (Nx, Nm, Nk)
        pmb = np.sum(pmb, axis=-1) # sum along the k axis, shape (Nx, Nm)

        # pre-factor
        pmb *= 1j * wavenumber[None, :] * multiplier / (4 * np.pi * R[:, None]) 

        return pmb, x

    def getPressureStator(self, x:np.ndarray, m:np.ndarray, multiplier:float=None):
        """
        Generic function for computing the hanson formulation of noise for stators

        multiplier is an overall multiplier for total the pressure mode. For B blades it should be B, for one stator/beam it should be 1.

        returns: p_m of size (Nx, Nm) - array of pressure modes m (NOTE: NOT m*B !!!!!!!!!) at observation points x, x is returned for convenience

        """
        if not np.all(m != 0):
            raise ValueError("m=0 is not supported")

        if multiplier is None:
            multiplier = self.nbeam

        c0 = self.c # SoS
        Omega = self.Omega
        Fblade = self.loadings # Nk, Nr
        B = self.B
        nb =self.nbeam

        # convert observation point to cylidrical relative to the prop
        R, theta, phi = getSphericalCoordinates(
            x, self.axis, self.origin, self.radial, self.normal
        ) # each of size Nx

        radius, twist, chord = self.radius_c, self.twist_c, self.chord_c # all of size # Nr
        dr = self.dr # Nr, size of segment

        mB = m * B # Nm

        wavenumber = mB * Omega / c0 # Nm, issue if mb = 0?

        k = self.k * nb # Nk multiplied by the number of beams!

        k = np.concatenate((-k[-1:0:-1], k)) # add the minus part!, shape (2Nk-1 -> Nk)
        Fblade = np.concatenate((np.conjugate(Fblade[-1:0:-1]), Fblade), axis=0) 

        m_int = np.asarray(m, dtype=np.int64)

        lookup = {val: i for i, val in enumerate(k)}

        Nm = len(m_int)
        Nr = Fblade.shape[1]

        # Preallocate with zeros (this automatically handles missing modes)
        Fm = np.zeros((Nm, Nr), dtype=Fblade.dtype)

        # Find which requested modes exist
        valid_mask = np.array([mb in lookup for mb in m_int])

        if np.any(valid_mask):
            valid_modes = m_int[valid_mask]
            idx = np.array([lookup[mb] for mb in valid_modes])
            Fm[valid_mask, :] = Fblade[idx, :]

        # Downstream projections
        Fphi = Fm * np.sin(twist)[None, :] #Nm, Nr
        Fz   = Fm * np.cos(twist)[None, :] #Nm, Nr

        # --- matrix construction ---
        matrix = (
            -Fphi[None, :, :] * np.sin(theta[:, None, None]) * np.sin(phi[:, None, None]) # NOTE: minus sign! see docs to see where it comes from
            + np.cos(theta[:, None, None]) * Fz[None, :, :]
        ) # shape Nx, Nm, Nr
        matrix *= np.exp(-1j * wavenumber[None, :, None] * R[:, None, None] * np.sin(theta)[:, None, None] * np.cos(phi)[:, None, None]) # shape Nx, Nm, Nr

        # reduce by summing along Nr axis
        pmb = np.sum (
            matrix
            * dr[None, None, :] # integration over r, note: we assume that Fblade is per unit span, in units N/m.
              ,
            axis=-1
        ) # integrate along the r axis
    
        # pre-factor
        pmb *= 1j * wavenumber[None, :] * multiplier / (4 * np.pi * R[:, None]) * np.exp(1j * wavenumber[None, :] * R[:, None])

        return pmb, x

    def getPolarMesh(self, R=1.0, Nphi=36, Ntheta=18, eps=np.pi/96):
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
    
    def plot3Ddirectivity(self, fig, ax, m:float, valmax=None, valmin=None, R=1.62,
                        Nphi=18, Ntheta=36, blending=0.1, title=None):
        
        """
        plot a 3D directivity pattern for a given mode m

        fig, ax are matplotlib instances on which to execute the plot, they are returned after plotting

        """

        # --- observation mesh ---
        x_cart, x_spherical, Theta, Phi = self.getPolarMesh(R=R, Nphi=Nphi, Ntheta=Ntheta)

        # --- pressure / magnitude ---
        pmB, _ = self.getPressureRotor(x_cart, np.array([m]).reshape(1,), multiplier=self.B) # of shape (Nx=Ntheta*Nphi, 1)
        pmB = pmB[:, 0] # shape (Nx,)

        fig, ax = plot_3D_directivity(
            pmB, Theta, Phi, 
            blending=blending,
            title=f"Far-field directivity of $p_{{mB}}$",
            valmin=valmin,
            valmax=valmax,
            fig=fig,
            ax=ax
        )
        
        return fig, ax

    def plotPressureSpectrum(self, fig, ax, x:tuple, m:np.ndarray, plot_kwargs={'color' : 'k', 'marker' : 's'}):


        # --- pressure / magnitude ---
        pmB = self.getPressureRotor(x, m, multiplier=self.B)[0] # of shape (Nx=Ntheta*Nphi, Nm)


        ax.plot(m, p_to_SPL(pmB[0]), **plot_kwargs)


        ax.minorticks_on()
        ax.grid(which='minor', alpha=0.5)
        ax.grid(which='major')
        ax.legend()

        ax.set_xscale('log')
        ax.set_xlabel('$f^+$')
        ax.set_ylabel('SPL w.r.t 20e-6 Pa')
        plt.tight_layout()

        return fig, ax


if __name__ == "__main__":
    pass