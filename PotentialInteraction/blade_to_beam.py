

import numpy as np
import numpy as np
from scipy.special import jv
from Constants.helpers import periodic_sum, plot_directivity_contour, p_to_SPL, periodic_sum_interpolated
import matplotlib.pyplot as plt
class BeamLoadings():

    def __init__(self, twist_rad:np.ndarray, chord_m:np.ndarray, radius_m:np.ndarray,
                 Uz0_mps:np.ndarray,
                 Tprime_Npm:np.ndarray,
                 Qprime_Npm:np.ndarray,
                  B=2, Dcylinder_m=0.0, Lcylinder_m=0.0, Omega_rads=1.0, rho_kgm3=1.0, c_mps = 340, kmax = 20, nb:float = 1):

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
        self.c = c_mps # speed of sound
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

    def getBeamLoadingHarmonics(self, D__Dref_max=10.0, points_per_period = 20):

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
        Fhat = self._getBeamVortexLoads(time_1d) # size (3, Nt, Nr)

        # T_periodic, F_beam = periodic_sum(Fhat, period, time_1d) # shapes (Np), (3, Np, Nr)
        k = self.k # Nk

        T_periodic = np.linspace(-period/2, period/2, points_per_period * np.max(k), endpoint=False) # Np
        T_periodic, F_beam = periodic_sum_interpolated(Fhat, period=period, time=time_1d, kind='cubic', t_new=T_periodic)

        Np = T_periodic.shape[0] # should be equal to points_per_period * max(k)!

        # shape (3, Nk, Nr)
        
        F_beam_k = 1/period * np.sum(F_beam[:, None, :, :] * np.exp(-1j *
                 k[None, :, None, None] * 2 * np.pi / period * 
                 T_periodic[None, None, :, None]) * dt, axis=2) # our convention for fourier transform: minus in the exp

        return F_beam_k

    # def getBeamLoadingMagnitude(self):
    #     Fbeam = self.getBeamLoadingHarmonics()
    #     angle = np.atan2(Fbeam[2, :, :] , Fbeam[1, :, :])
    #     return Fbeam[1, :, :] / np.cos(angle) # shape (Nk, Nr) assuming orientation
    
    def _getBeamVortexLoads(self, time, Npoints=360):

        Nt = time.shape[0]
        Nr = self.Nr

        pressure, thetab, deltathetab = self._getBeamVortexPressure(time, Npoints) # (Npoints, Nt, Nr)


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

    def _getBeamVortexPressure(self, time, Npoints=360, overwrite_positions=None):
        """
        time: array, shape (Nt, Nr)
        """

        Nt = time.shape[0]
        Nr = self.Nr

        T_per_unit_span = self.Tprime # Nr
        
        Uz = -self.Uz # Nr # negative


        # stagger = np.arctan(Uz / self.Omega / self.seg_radius) # Nr
        stagger = self.seg_twist
        L_per_unit_span = T_per_unit_span * np.cos(stagger)

        Ur = np.sqrt(Uz**2 + (self.Omega * self.seg_radius)**2) # Nr
        gamma = L_per_unit_span / self.rho / Ur # Nr


        # --- explicitly expand radial quantities
        Uz_e     = Uz[None, None, :]        # (1, 1, Nr)
        gamma_e = gamma[None, None, :]      # (1, 1, Nr)
        r_e     = self.seg_radius[None, None, :]  # (1, 1, Nr)
        time_r = time[:, None]

        Zv = self.Omega * self.seg_radius[None, :] * time_r + 1j * self.Lcylinder # vortex position, complex, size (Nt, Nr), vortex is moving from negative x to positive with speed Omega * r
        Zvbar = np.conjugate(Zv)

        Zv_e = Zv[None, :, :]   # (1, Nt, Nr)
        Zvbar_e = Zvbar[None, :, :] # (1, Nt, Nr)

        thetab = np.linspace(0, 2 * np.pi, Npoints, endpoint=False) # angles on the cylinder surface, measured from the prop. plane, size (Npoints)
        deltathetab = 2 * np.pi / Npoints


        if overwrite_positions is not None:
            Z = overwrite_positions[0, :] + 1j *overwrite_positions[1, :] # pass external position coordinates, mostly for debugging
        else:
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

        if overwrite_positions is not None:
            # pressure[np.abs(Z) < self.Dcylinder/2, :, :] = 0.0 
            return pressure, overwrite_positions
        else:
            return pressure, thetab, deltathetab
    
    def getBeamPressureHarmonics(self, D__Dref_max=10.0, points_per_period = 20):

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

        pressure, thetab, deltathetab = self._getBeamVortexPressure(time_1d) # pressure of shape Npoints, Nt, Nr

        # T_periodic, pressure_periodic = periodic_sum(pressure, period, time_1d) # shapes (Np), (Npoints, Np, Nr)
        k = self.k # Nk

        T_periodic = np.linspace(-period/2, period/2, points_per_period * np.max(k), endpoint=False) # Np
        T_periodic, pressure_periodic = periodic_sum_interpolated(pressure, period=period, time=time_1d, kind='cubic', t_new=T_periodic)

        # shape (Npoints, Nk, Nr)
        p_k = 1/period * np.sum(pressure_periodic[:, None, :, :] * np.exp(-1j *
                 k[None, :, None, None] * 2 * np.pi / period *
                 T_periodic[None, None, :, None]) * dt, axis=2) # our convention for fourier transform: minus in the exp        
        
        return p_k, thetab, self.seg_radius # shape (Npoints, Nk, Nr), Npoints, Nr
    
    def plotSurfacePressureContour(self, m:int, fig=None, ax=None, D__Dref_max=10.0, points_per_period = 20):
        p, thetab, radius = self.getBeamPressureHarmonics(D__Dref_max, points_per_period) # p_k of shape (Npoints, Nk, Nr), thetab of shape (Npoints), radius of shape (Nr)
        index = np.where(self.k == m)[0][0] # index of the desired mode 
        # if does not exist, will throw an error, which is fine
        pk = p[:, index, :]

                # --- shift theta from [0, 2π) → [-π, π) ---
        th_shifted = (thetab + np.pi) % (2*np.pi) - np.pi

        # --- sort theta so it increases from -π to π ---
        sort_idx = np.argsort(th_shifted)

        th_sorted = th_shifted[sort_idx]
        pk = pk.reshape(len(thetab), len(radius))
        pk = pk[sort_idx,]
        pk = pk.flatten()

        fig, ax = plot_directivity_contour(magnitudes=p_to_SPL(pk), theta=np.rad2deg(th_sorted), phi=radius, fig=fig, ax=ax, ylabel=r'$\theta$ [deg]', xlabel='$z$ [m]', title='Surface Pressure Directivity (dB)')

        return fig, ax

