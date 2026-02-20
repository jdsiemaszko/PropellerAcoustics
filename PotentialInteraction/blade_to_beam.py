

import numpy as np
import numpy as np
from Constants.helpers import p_to_SPL, theodorsen
from scipy.special import jv
from Constants.helpers import periodic_sum

class BeamLoadings():

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

    def getBeamLoadingHarmonics(self, D__Dref_max=50.0, points_per_period = 20):

        period = 2 * np.pi / self.B / self.Omega

        dref = self.Lcylinder # some lengthscale
        Nr = self.seg_radius.shape[0]

        tmax = D__Dref_max * dref / self.Omega / self.seg_radius # N_r ?
        tmaxmax = np.max(tmax) # should be the FIRST ENTRY

        dt = period / (points_per_period * self.kmax) # timestep chosen small enough to resolve the maximum frequency!

        # tmaxmax = 0.0

        # integration time
        time_1d = np.arange(
            -tmaxmax - period/2,
            tmaxmax + period/2 + dt,
                            dt) # (Nt, Nr) ?, ensure all times in (-period/2, period/2) have plenty of datapoints outside to sum
        Nt = time_1d.size

        # --- expand time to (Nt, Nr)
        time = time_1d[:, None] * np.ones((1, Nr))  # (Nt, Nr)


        Fhat = self.__getBeamVortexLoads(time) # size (3, Nt, Nr)

        T_periodic, F_beam = periodic_sum(Fhat, period, time_1d) # shapes (Np), (3, Np, Nr)

        Np = T_periodic.shape[0] # should be equal to points_per_period!

        k = self.k # Nk
        # shape (3, Nk, Nr)
        
        F_beam_k = self.B * self.Omega / 4 / np.pi * np.sum(F_beam[:, None, :, :] * np.exp(1j *
                 k[None, :, None, None] * self.B * self.Omega * 
                 T_periodic[None, None, :, None]) * dt, axis=2)

        return F_beam_k

    def __getBeamVortexLoads(self, time, Npoints=36):
        """
        time: array, shape (Nt, Nr)
        """
        Nt, Nr = time.shape

        T_per_unit_span = self.Tprime # Nr
        
        Uz = self.Uz # Nr


        stagger = np.arctan(Uz / self.Omega / self.seg_radius) # Nr
        L_per_unit_span = T_per_unit_span / np.cos(stagger)

        Ur = np.sqrt(Uz**2 + (self.Omega * self.seg_radius)**2) # Nr
        gamma = L_per_unit_span / self.rho / Ur # Nr

        # --- explicitly expand radial quantities
        Uz_e     = Uz[None, None, :]        # (1, 1, Nr)
        gamma_e = gamma[None, None, :]      # (1, 1, Nr)
        r_e     = self.seg_radius[None, None, :]  # (1, 1, Nr)

        Zv = self.Omega * self.seg_radius[None, :] * time + 1j * self.Lcylinder # vortex position, complex, size (Nt, Nr), vortex is moving from negative x to positive with speed Omega * r
        Zvbar = np.conjugate(Zv)

        Zv_e = Zv[None, :, :]   # (1, Nt, Nr)
        Zvbar_e = Zvbar[None, :, :] # (1, Nt, Nr)

        thetab = np.linspace(0, 2 * np.pi, Npoints, endpoint=False) # angles on the cylinder surface, measured from the prop. plane, size (Npoints)
        deltathetab = 2 * np.pi / Npoints

        Z = self.Dcylinder/2 * np.exp(1j * thetab) # positions along the cylinder surface, complex, size, (Npoints)

        Z_circ_conjugate = self.Dcylinder**2 / 4 / Z

        Z_e = Z[:, None, None] # (Npoints, 1, 1)
        Z_circ_conjugate_e = Z_circ_conjugate[:, None, None]

        # complex derivative of the complex potential dfdz = u - i * v (which follows from f being holomorphic on Z!=Zv), size (Npoints, Nt, Nr)
        dfdz = -1j * gamma_e / 2 / np.pi / (Z_e-Zv_e) + 1j * gamma_e / 2 / np.pi / (Z_circ_conjugate_e - Zvbar_e) * (-Z_circ_conjugate_e / Z_e) + 1j * Uz_e * (1 + Z_circ_conjugate_e / Z_e)

        u = np.real(dfdz) # (Npoints, Nt, Nr)
        v = -np.imag(dfdz) # (Npoints, Nt, Nr)
        U = np.sqrt(u**2 + v**2) # (Npoints, Nt, Nr)

        pressure = 0.5 * self.rho * (Uz_e**2 - U**2) + self.rho * gamma_e * self.Omega * r_e / 2 / np.pi * np.real(
            1j / Zvbar_e + 1j / (Zv_e - Z_e) - 1j / (Zvbar_e - Z_circ_conjugate_e)
        ) # (Npoints, Nt, Nr)


        # --- force integration on cylinder
        cos_t = np.cos(thetab)[:, None, None]   # (Npoints, 1, 1)
        sin_t = np.sin(thetab)[:, None, None]   # (Npoints, 1, 1)


        Fphi = self.Dcylinder / 2 * np.sum(
            pressure *
            cos_t *
            deltathetab,
            axis=0
        ) # (Nt, Nr), drag, oriented backwards


        Fz = -self.Dcylinder / 2 * np.sum(
            pressure *
            sin_t * 
            deltathetab,
            axis=0
        ) # (Nt, Nr) # lift, oriented upwards

        Fbeam = np.zeros((3, Nt, Nr)) # Note: in the time domain!, size (3, Nt, Nr)
        Fbeam[1, :, :] = Fz
        Fbeam[2, :, :] = Fphi

        return Fbeam
