import numpy as np
import numpy as np
from Constants.helpers import p_to_SPL, theodorsen
from scipy.special import jv

class BladeLoadings():

    def __init__(self, twist_rad:np.ndarray, chord_m:np.ndarray, radius_m:np.ndarray,
                 Uz0_mps:np.ndarray,
                 Tprime_Npm:np.ndarray,
                 Qprime_Npm:np.ndarray,
                  B=2, Dcylinder_m=0.0, Lcylinder_m=0.0, Omega_rads=1.0, rho_kgm3=1.0, c_ms = 340, kmax = 20, nb:float = 1):

        """
        arrays: twist, chord, radius of size Nr+1, defined as edges of radial stations
        Uz, Tprime, Qprime of size Nr, defined at centers of radial stations.
        These are the  mean velocity, thrust and axial force along the blade. T and Q per unit span

        """
        self.B = B
        self.Dcylinder = Dcylinder_m
        self.Lcylinder = Lcylinder_m
        self.Omega=Omega_rads
        self.rho = rho_kgm3
        self.c = c_ms # speed of sound
        if nb!=1:
            raise ValueError("WARNING: case nb>1 not implemented yet!")
        self.nbeam = nb


        self.twist = twist_rad # Nr
        self.chord = chord_m # Nr
        self.radius = radius_m # Nr
        self.r0 = radius_m[0]
        self.r1 = radius_m[-1]

        self.seg_twist = (twist_rad[1:] + twist_rad[:-1]) / 2
        self.seg_chord = (chord_m[1:] + chord_m[:-1]) / 2
        self.seg_radius = (radius_m[1:] + radius_m[:-1]) / 2

        self.dr = np.diff(radius_m) # (Nr -1)

        self.Nr = len(twist_rad) - 1
        self.Nk = kmax+1
        self.kmax = kmax
        self.k = np.arange(0, kmax+1, 1) # array of modal orders

        self.Uz = Uz0_mps # Nr
        self.Tprime = Tprime_Npm # Nr
        self.Qprime = Qprime_Npm # Nr

    def getDistortionHarmonics(self):
        # --- sizes ---
        Nk = self.Nk
        Nr = self.Nr

        # --- base arrays ---
        # Uz = self.getAxialInducedVelocity()  # Nr
        # Uz = self.Omega * self.seg_radius * np.tan(np.pi - self.seg_twist)
        Uz = self.Uz # Nr

        Uz_r = Uz[None, :]  # (1, Nr)
        k_k = self.k[:, None]                            # (Nk, 1)

        r_r = self.seg_radius[None, :]                   # (1, Nr)
        # beta_r = self.seg_twist[None, :]            # (1, Nr)

        # beta = np.pi/2 - np.arctan(Uz / self.Omega / self.seg_radius) # stagger angle! Nr

        beta = np.pi/2 - self.seg_twist
        beta_r = beta[None, :] # (1, Nr)

        Dc = self.Dcylinder
        Lc = self.Lcylinder

        # --- distortion harmonics ---
        wk = (
            -1j
            * k_k 
            * Uz_r
            * Dc**2
            / 8.0
            / r_r**2
            * np.exp(
                -k_k / r_r * Lc
                + 1j * beta_r
            )
            # * (-1.0) ** k_k
            * np.exp(1j * np.pi * k_k)
        )                                                 # (Nk, Nr)

        return wk
    
    def getBladeLoadingHarmonics(self):
        Omega = self.Omega
        rho = self.rho

        # --- sizes ---
        Nk = self.Nk
        Nr = self.Nr

        # --- base arrays ---
        wk = self.getDistortionHarmonics()          # (Nk, Nr)
        # Uz = self.getAxialInducedVelocity() # Nr
        Uz = self.Uz # Nr

        k_k = self.k[:, None]                       # (Nk, 1)

        r_r = self.seg_radius[None, :]              # (1, Nr)
        c_r = self.seg_chord[None, :]               # (1, Nr)
        # beta_r = self.seg_twist[None, :]            # (1, Nr)

        beta = np.pi/2 - self.seg_twist


        # beta = np.arctan(self.Omega * self.seg_radius / Uz)
        beta_r = beta[None, :] # (1, Nr)

        # --- kinematics ---
        # Ur = np.sqrt((Omega * r_r)**2 + Uz**2)                            # (1, Nr)
        Ur = Omega * r_r

        # --- Theodorsen arguments ---
        sigma = k_k * Omega * c_r / (2.0 * Ur)      # (Nk, Nr)

        mu = (
            1j * k_k * c_r / (2.0 * r_r)
            * np.exp(-1j * beta_r)
        )                                           # (Nk, Nr)
        
        US_TERM = np.ones((Nk, Nr), dtype=np.complex128)
        US_TERM[0, :] = 1.0
        C = theodorsen(sigma[1:, :])
        T1 = jv(0, mu[1:, :]) - 1j * jv(1, mu[1:, :])
        T2 = 1j * sigma[1:, :] / mu[1:, :] * jv(1, mu[1:, :])

        US_TERM[1:, :] = np.conjugate(C * T1 + T2)
        
        Lkprime = (
            np.pi * rho * c_r * Ur
            * wk
            * US_TERM
        )                                       # (Nk, Nr)

        # --- allocate blade forces ---
        Fblade = np.zeros((3, Nk, Nr), dtype=np.complex128) # radial, axial, tangential

        # lift components (rotate by stagger)
        #TODO: signs?
        # Fblade[1, :, :] = -Lkprime * np.sin(beta_r)   # axial
        # Fblade[2, :, :] = Lkprime * np.cos(beta_r)   # tangential
        Fblade[1, :, :] = -Lkprime * np.cos(self.seg_twist[None, :]) # positive upwards, but Lkprime is oriented downwards for positive wk!
        Fblade[2, :, :] = -Lkprime * np.sin(self.seg_twist[None, :]) # DRAG, oriented BACKWARDS


        # steady loads. Note: phase shift


        Fblade[1, 0, :] = self.Tprime # axial, positive upwards
        Fblade[2, 0, :] = self.Qprime # tangential, positive backwards

        return Fblade
    
    def getBladeLoadingMagnitude(self):
        Fblade = self.getBladeLoadingHarmonics()
        return Fblade[1, :, :] / np.cos(self.seg_twist[None, :]) # shape (Nk, Nr) assuming orientation