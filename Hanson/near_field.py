from .far_field import HansonModel
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.special import hankel2, jve, jv
from Constants.helpers import getCylindricalCoordinates, getSphericalCoordinates, distance_in_polar, p_to_SPL



class NearFieldHansonModel(HansonModel):

    def __init__(self, N_points=128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Npoints = N_points
        self.alpha = np.linspace(0, 2 * np.pi, self.Npoints, endpoint=False) # azimuthal points for near-field evaluation (equivalent to source-mode radial stations)
        self.dalpha = self.alpha[1] - self.alpha[0] # azimuthal step size


        # Broadcast seg_radius to shape (Nalpha Nr)
        seg_radius_broadcasted = np.broadcast_to(self.radius_c[None, :], (self.Npoints, self.Nr))

        # Broadcast Omega*time to shape (Nt, Nr)
        phi_array = np.broadcast_to(self.alpha[:, None], (self.Npoints, self.Nr))

        theta_array = np.ones_like(phi_array) * np.pi / 2 # evaluate in the plane of the rotor!

        # Stack along new first axis to get shape (3, Nalpha, Nr)
        self.positions_m = np.stack([seg_radius_broadcasted, theta_array, phi_array], axis=0) # shape (3, Nalpha, Nr)
        self.positions_m_stator = np.stack([self.radius_c, np.ones_like(self.radius_c)*np.pi/2, np.zeros_like(self.radius_c)], axis=0) # shape (3, Nr)

    def _getPressureRotorParallel(self, x: np.ndarray, m: np.ndarray, multiplier: float = None):
        """
        Fully vectorized near-field Hanson rotor pressure.
        Time integration rewritten as alpha-integration.
        Tensor reductions used instead of time loop.
        """

        if not np.all(m != 0):
            raise ValueError("m=0 is not supported")

        if multiplier is None:
            multiplier = self.B

        c0 = self.c
        Omega = self.Omega
        B = self.B
        radius, twist, chord = self.radius_c, self.twist_c, self.chord_c # all of size # Nr


        # --------------------------------------------
        # Harmonics
        # --------------------------------------------
        mB = m * B                          # (Nm,)
        wavenumber = mB * Omega / c0        # (Nm,)

        k = self.k
        Fblade = self.loadings              # (3, Nk, Nr)

        k = np.concatenate((-k[-1:0:-1], k)) # add the minus part!, shape (2Nk-1 -> Nk)
        Fblade = np.concatenate((np.conjugate(Fblade[-1:0:-1]), Fblade), axis=0) # minus loadings are conjugates of positive!
        # Fblade and k are now of shape (2Nk-1, Nr) with negative modes first

        Nk = k.size
        Nr = self.Nr
        Nm = mB.size
        Nx = x.shape[0]
        Nalpha = self.Npoints

        # --------------------------------------------
        # Geometry
        # --------------------------------------------
        Rprime = distance_in_polar(x, self.positions_m)  
        # (Nx, Nalpha, Nr)

        Rprime = Rprime[:, None, :, None, :]  
        # (Nx, 1, Nalpha, 1, Nr)

        alpha = self.alpha[None, None, :, None, None]  
        # (1, 1, Nalpha, 1, 1)

        dalpha = self.dalpha
        prefac = dalpha / (2*np.pi)

        # --------------------------------------------
        # Broadcast harmonic structure
        # --------------------------------------------
        mB = mB[None, :, None, None, None]         # (1, Nm, 1, 1, 1)
        k = k[None, None, None, :, None]           # (1, 1, 1, Nk, 1)

        mB_minus_k = mB - k                        # (1, Nm, 1, Nk, 1)

        wavenumber = wavenumber[None, :, None, None, None]

        radius = self.radius_c[None, None, None, None, :]
        dr = self.dr[None, None, None, None, :]

        # --------------------------------------------
        # Green functions
        # --------------------------------------------
        G1 = (
            np.exp(1j * wavenumber * Rprime)
            / (Rprime**2)
            * (1 - 1/(1j*wavenumber*Rprime))
        )

        G2 = G1 * np.sin(alpha)

        # --------------------------------------------
        # Alpha-integration (NO TIME LOOP)
        # --------------------------------------------
        phase = np.exp(-1j * mB_minus_k * alpha)
        # (1, Nm, Nalpha, Nk, 1)

        G1_int = np.sum(G1 * phase, axis=2) * prefac
        G2_int = np.sum(G2 * phase, axis=2) * prefac
        # result: (Nx, Nm, Nk, Nr)

        # --------------------------------------------
        # Loadings
        # --------------------------------------------
        Fphi = np.sin(twist)[None, None, None, :] * Fblade[None, None, :, :] # (1, 1, Nk, Nr) NOTE: this is drag, oriented opposite to direction of travel
        Fz = np.cos(twist)[None, None, None, :] * Fblade[None, None, :, :] # (1, 1, Nk, Nr)
        
        theta = np.pi/2

        matrix = (
            Fz * np.cos(theta) * G1_int +
            Fphi * np.sin(theta) * G2_int
        ) * dr

        matrix *= 1j * wavenumber[..., 0, 0] / (4*np.pi)

        # --------------------------------------------
        # Final reductions
        # --------------------------------------------
        pmb = np.sum(matrix, axis=(-2, -1))   # sum over Nk, Nr

        pmb *= multiplier

        return pmb, x
    
    def getPressureRotor(self, x: np.ndarray, m: np.ndarray, multiplier: float = None):
        """
        Memory-optimized near-field Hanson rotor pressure.

        - Loops over observer positions (Nx)
        - Alpha integration via tensor contraction (einsum)
        """

        if not np.all(m != 0):
            raise ValueError("m=0 is not supported")

        if multiplier is None:
            multiplier = self.B

        c0 = self.c
        Omega = self.Omega
        B = self.B

        radius = self.radius_c
        twist = self.twist_c
        dr = self.dr

        # --------------------------------------------
        # Harmonics
        # --------------------------------------------
        mB = m * B                         # (Nm,)
        wavenumber = mB * Omega / c0       # (Nm,)

        k = self.k
        Fblade = self.loadings             # (Nk, Nr)

        # Add negative modes
        k = np.concatenate((-k[-1:0:-1], k))
        Fblade = np.concatenate(
            (np.conjugate(Fblade[-1:0:-1]), Fblade),
            axis=0
        )

        Nm = mB.size
        Nk = k.size
        Nr = self.Nr
        Nx = x.shape[1]
        Nalpha = self.Npoints

        alpha = self.alpha
        dalpha = self.dalpha
        prefac = dalpha / (2*np.pi)

        # --------------------------------------------
        # Precompute alpha-phase matrices
        # --------------------------------------------
        # (Nm, Nα)
        E_m = np.exp(-1j * np.outer(mB, alpha))

        # (Nk, Nα)
        E_k = np.exp(1j * np.outer(k, alpha))

        # --------------------------------------------
        # Loadings (pre-rotated)
        # --------------------------------------------
        Fphi = np.sin(twist)[None, :] * Fblade
        Fz   = np.cos(twist)[None, :] * Fblade

        theta = np.pi / 2

        # Output
        pmb = np.zeros((Nx, Nm), dtype=complex)

        # switch x to polar:

        x_polar = getSphericalCoordinates(x, self.axis, self.origin, self.radial, self.normal) # r, theta, phi


        # ==========================================================
        # LOOP OVER OBSERVER POSITIONS  (memory limiting done here)
        # ==========================================================
        for ix in range(Nx):
            # print(f'Computing pressure for observer {ix+1}/{Nx}...')
            r, theta, phi = x_polar[:, ix]
            # --------------------------------------------
            # Geometry for ONE observer
            # --------------------------------------------
            Rprime = distance_in_polar(
                x_polar[:, ix], self.positions_m
            )                       # (Nα, Nr)

            # --------------------------------------------
            # Green function (Nm, Nα, Nr)
            # --------------------------------------------
            # expand only in Nm direction
            R_exp = Rprime[None, :, :]  # (1, Nα, Nr)
            k_exp = wavenumber[:, None, None]  # (Nm,1,1)

            G1 = (
                np.exp(1j * k_exp * R_exp)
                / (R_exp**2)
                * (1 - 1/(1j*k_exp*R_exp))
            )  # (Nm, Nα, Nr)

            G2 = G1 * np.sin(alpha - phi)[None, :, None]

            # --------------------------------------------
            # Alpha integration via tensor contraction
            #
            # G_int(m,k,r) =
            #   Σ_α G(m,α,r) * E_m(m,α) * E_k(k,α)
            # --------------------------------------------

            # Combine m and α first
            G1m = G1 * E_m[:, :, None]        # (Nm, Nα, Nr)
            G2m = G2 * E_m[:, :, None]

            # Contract α with k-phase
            # result: (Nm, Nk, Nr)
            G1_int = np.einsum(
                'm a r, k a -> m k r',
                G1m,
                E_k
            ) * prefac

            G2_int = np.einsum(
                'm a r, k a -> m k r',
                G2m,
                E_k
            ) * prefac

            # --------------------------------------------
            # Final radial + modal reduction
            # --------------------------------------------
            contrib = (
                Fz[None, :, :] * np.cos(theta) * G1_int +
                Fphi[None, :, :] * np.sin(theta) * G2_int
            ) * dr[None, None, :]

            contrib *= 1j * wavenumber[:, None, None] / (4*np.pi) * r

            # sum over k and r
            pmb[ix] = np.sum(contrib, axis=(1, 2))

        pmb *= multiplier

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
        x_polar = getSphericalCoordinates(
            x, self.axis, self.origin, self.radial, self.normal
        ) # each of size Nx

        Rprime = distance_in_polar(
                x_polar, self.positions_m_stator
            )    # (Nx, Nr)?
        R, theta, phi = x_polar
        

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

        matrix *= R[:, None, None] / Rprime[:, None, :]
        matrix *= (np.ones(matrix.shape) - 1/(1j*wavenumber[None, :, None]*Rprime[:, None, :]))
        matrix *= np.exp(1j * wavenumber[None, :, None] * Rprime[:, None, :])
        matrix *= 1j * wavenumber[None, :, None] / (4*np.pi) / Rprime[:, None, :]

        # reduce by summing along Nr axis
        pmb = np.sum (
            matrix
            * dr[None, None, :] # integration over r, note: we assume that Fblade is per unit span, in units N/m.
              ,
            axis=-1
        ) # integrate along the r axis, result of shape Nx, Nm
    
        # pre-factor
        pmb *= multiplier
        return pmb, x
