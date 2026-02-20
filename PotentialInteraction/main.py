from scipy.special import hankel2, jve, jv
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .placeholder import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from Constants.const import p_to_SPL

def periodic_sum(array, period, time):
    """
    Perform a discrete periodic sum (modal sum) of a signal onto one period.

    Parameters
    ----------
    array : array-like, shape (..., Nt, Nr)
        Signal sampled at 'time'.
        The **last-but-one axis** must correspond to time.
    period : float
        Period of the signal.
    time : 1D array, shape (Nt,)
        Time samples covering a large window, e.g., (-tmax-period/2, tmax+period/2)

    Returns
    -------
    t_mod : 1D array, shape (Np,)
        Time samples in [0, period), evenly spaced with spacing dt.
    psum : array, shape (..., Np, Nr)
        Periodic sum of the signal onto one period.
    """

    array = np.asarray(array)
    time = np.asarray(time)

    # --- assume uniform time spacing
    dt = np.mean(np.diff(time))
    n_per = int(np.ceil(period / dt))
    Np = n_per

    # --- output time grid
    t_mod = np.arange(Np) * dt
    t_mod = t_mod % period - period/2  # ensure [-period/2, period/2)

    # --- prepare output array
    psum_shape = list(array.shape)
    psum_shape[-2] = Np  # replace time axis
    psum = np.zeros(psum_shape, dtype=array.dtype)

    # --- fold each time sample onto the correct bin
    # last-but-one axis is assumed to be time
    time_idx = np.floor((time % period) / dt).astype(int)
    time_idx = np.clip(time_idx, 0, Np-1)  # safety

    # iterate over time samples (vectorized over all other axes)
    for i, k in enumerate(time_idx):
        # np.take along time axis
        psum[..., k, :] += array[..., i, :]

    return t_mod, psum


def theodorsen(sigma):
    res = hankel2(1, sigma) / (hankel2(1, sigma) + 1j * hankel2(0, sigma))

    res[np.where(np.abs(sigma) < 1e-12)] = 1.0 # low freq limit

    return res
    # return 1.0

def theodorsen_stable(sigma, tol_small=0.1, tol_large=5.0):
    sigma = np.asarray(sigma, dtype=np.complex128)
    res = np.empty_like(sigma)

    small = np.abs(sigma) < tol_small
    large = np.abs(sigma) > tol_large
    mid   = ~(small | large)

    # ---- small σ: series
    if np.any(small):
        s = sigma[small]
        res[small] = 1.0 - 0.5j * np.pi * s

    # ---- large σ: asymptotic expansion
    if np.any(large):
        s = sigma[large]
        res[large] = 0.5 + 1j / (8.0 * s)

    # ---- intermediate σ: direct evaluation
    if np.any(mid):
        s = sigma[mid]
        H1 = hankel2(1, s)
        H0 = hankel2(0, s)
        res[mid] = H1 / (H1 + 1j * H0)

    return res

def bessel_terms(mu, sigma, tol_small=0.1, tol_large=5.0):
    mu = np.asarray(mu, dtype=np.complex128)
    sigma = np.asarray(sigma, dtype=np.complex128)

    T1 = np.empty_like(mu)  # J0 - i J1
    T2 = np.empty_like(mu)  # i σ/μ J1

    small = np.abs(mu) < tol_small
    large = np.abs(mu) > tol_large
    mid   = ~(small | large)

    # ---- small μ: series
    if np.any(small):
        m = mu[small]
        s = sigma[small]

        J0 = 1.0 - m**2 / 4.0
        J1 = m / 2.0 - m**3 / 16.0

        T1[small] = J0 - 1j * J1
        T2[small] = 1j * s * (0.5 - m**2 / 16.0)

    # ---- large μ: asymptotic (scaled)
    if np.any(large):
        m = mu[large]
        s = sigma[large]

        pref = np.sqrt(2.0 / (np.pi * m))

        # dominant scaled phase
        phase = np.exp(-1j * (m - np.pi/4))

        J0m_iJ1m = pref * phase
        J1m = pref * np.cos(m - 3*np.pi/4)

        T1[large] = J0m_iJ1m
        T2[large] = 1j * s / m * J1m

    # ---- intermediate μ: scaled special functions
    if np.any(mid):
        m = mu[mid]
        s = sigma[mid]

        J0 = jve(0, m)
        J1 = jve(1, m)

        T1[mid] = J0 - 1j * J1
        T2[mid] = 1j * s / m * J1

    return T1, T2
class HansonModel():

    def __init__(self, twist_rad, chord_m, radius_m, B=2, Dcylinder_m=0.0, Lcylinder_m=0.0, Omega_rads=1.0, rho_kgm3=1.0, c_ms = 340, kmax = 20, nb:float = 1):
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

    def getDistortionHarmonics(self):
        # --- sizes ---
        Nk = self.Nk
        Nr = self.Nr

        # --- base arrays ---
        Uz = self.getAxialInducedVelocity()  # Nr
        # Uz = self.Omega * self.seg_radius * np.tan(np.pi - self.seg_twist)


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

    def getAxialInducedVelocity(self):
        return interpolate_UZ(self.seg_radius / self.r1, self.Omega)
    
    def getBladeLoadingHarmonics(self):
        Omega = self.Omega
        rho = self.rho

        # --- sizes ---
        Nk = self.Nk
        Nr = self.Nr

        # --- base arrays ---
        wk = self.getDistortionHarmonics()          # (Nk, Nr)
        Uz = self.getAxialInducedVelocity() # Nr

        k_k = self.k[:, None]                       # (Nk, 1)

        r_r = self.seg_radius[None, :]              # (1, Nr)
        c_r = self.seg_chord[None, :]               # (1, Nr)
        # beta_r = self.seg_twist[None, :]            # (1, Nr)

        # beta = np.pi/2 - np.arctan(Uz / self.Omega / self.seg_radius) # stagger angle! Nr
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

        # --- lift harmonic --- (force per unit span)
        # US_TERM = np.ones((Nk, Nr), dtype=np.complex128)
        # US_TERM[0, :] = 1.0
        # US_TERM[1:, :] = np.conjugate(theodorsen(sigma[1:, :])  * (jve(0, mu[1:, :]) - 1j * jve(1, mu[1:, :])) + 1j * sigma[1:, :] / mu[1:, :] * jve(1, mu[1:, :]))
        
        
        
        US_TERM = np.ones((Nk, Nr), dtype=np.complex128)
        US_TERM[0, :] = 1.0

        # C = theodorsen_stable(sigma[1:, :])
        # T1, T2 = bessel_terms(mu[1:, :], sigma[1:, :])

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
        # Fblade[1, 0, :] = np.interp(self.seg_radius/self.r1, r_rt_common, thrust_per_unit_span_i) * self.B
        # Fblade[2, 0, :] = np.interp(self.seg_radius/self.r1, r_rt_common, tan_per_unit_span_i) * self.B
        Fblade[1, 0, :] = np.interp(self.seg_radius/self.r1, R_RT_EXACT, DT_EXACT) * self.B
        Fblade[2, 0, :] = np.interp(self.seg_radius/self.r1, R_RT_EXACT, DQ_EXACT) * self.B


        # Fblade[:, 1:, :]=0 # TODO: remove

        return Fblade
    
    def getBladeLoadingMagnitude(self):
        Fblade = self.getBeamLoadingHarmonics()
        return Fblade[1, :, :] / np.cos(self.seg_twist[None, :]) # shape (Nk, Nr)

    def getHansonPressure(self, x, m, B, Omega, loading, nb, multiplier):
        """
        Generic function for computing the hanson formulation of noise for rotors
        loading should be: radial, axial, tangential
        x should be: radius, polar, azimuth

        multiplier is an overall multiplier for total the pressure mode. For B blades it should be B, for one stator/beam it should be 1.
        """
        assert np.all(m != 0), "m=0 (steady loading) is not supported"

        c0 = self.c # SoS

        # todo: make the transformation explicit
        R, theta, phi = x[0], x[1], x[2] # all of size Nx

        radius, twist, chord = self.seg_radius, self.seg_twist, self.seg_chord # all of size # Nr
        dr = self.dr # Nr, size of segment

        mB = m * B # Nm

        wavenumber = mB * Omega / c0 # Nm, issue if mb = 0?

        k = self.k * nb # Nk multiplied by the number of beams!


        # Fblade = self.getLoadingHarmonics() # (3, Nr, Nk)
        Fblade = loading
        # p_mB (Nx, Nm)

        k = np.concatenate((-k[-1:0:-1], k)) # add the minus part!, shape (2Nk-1 -> Nk)
        Fblade = np.concatenate((np.conjugate(Fblade[:, -1:0:-1]), Fblade), axis=1) # minus loadings are conjugates of positive!

        # --- explicit broadcasting ---
        R_x      = R[:, None, None, None]          # (Nx, 1, 1, 1)
        theta_x = theta[:, None, None, None]       # (Nx, 1, 1, 1)
        phi_x   = phi[:, None, None, None]         # (Nx, 1, 1, 1)

        mB_m    = mB[None, :, None, None]           # (1, Nm, 1, 1)
        k_k     = k[None, None, :, None]            # (1, 1, Nk, 1)

        radius_r = radius[None, None, None, :]      # (1, 1, 1, Nr)
        dr_r     = dr[None, None, None, :]          # (1, 1, 1, Nr)

        wavenumber_m = wavenumber[None, :, None, None]  # (1, Nm, 1, 1)

        Fphi = Fblade[2][None, None, :, :] # (1, 1, Nk, Nr) NOTE: this is drag, oriented opposite to direction of travel
        Fz = Fblade[1][None, None, :, :] # (1, 1, Nk, Nr)
        # Fphi = Fz / np.cos(self.seg_twist[None, :]) * np.sin(self.seg_twist[None, :]) # convert to tangential force, oriented opposite to direction of travel

        # --- matrix construction ---
        matrix = (
            + Fphi * (mB_m - k_k) / radius_r / (wavenumber_m) # sure about: drag is with a positive sign
            + np.cos(theta_x) * Fz
        )

        # --- apply condition: wavenumber=0 and m!=k ---
        # Create a mask that broadcasts correctly to (Nx, Nm, Nk, Nr)
        # mask = (wavenumber_m == 0) & (mB_m != k_k)  # shapes: (1, Nm, 1, 1) & (1, Nm, Nk, 1) -> broadcast to (Nx, Nm, Nk, Nr)
        # matrix = np.where(mask, 0, matrix)

        matrix *= jv(mB_m - k_k, mB_m * Omega * radius_r / c0 * np.sin(theta_x))

        matrix = matrix.astype(np.complex128)

        matrix *= np.exp(
           -1j * (mB_m - k_k) * (phi_x  + np.pi / 2)
            # +1j * (mB_m - k_k) * Omega * R_x / c0 # NOTE: is this correct?
            +1j * (mB_m) * Omega * R_x / c0 # NOTE: is this correct?        

        # 1j * (mB_m - k_k) * (phi_x - np.pi/2)
        )
        # matrix shape: (Nx, Nm, Nk, Nr)

        # reduce by summing along Nk and Nr axes
        pmb = np.sum (
            matrix
              * dr_r
              ,
            axis=-1
        ) # integrate along the r axis
        pmb = np.sum(pmb, axis=-1) # sum along the k axis
        
        
        # np.sum(matrix, axis=(-1, -2))  # -> (Nx, Nm)

        # scaling
        pmb *= 1j * wavenumber[None, :] * multiplier / (4 * np.pi * R[:, None]) 
        # * np.exp(1j * wavenumber[None, :] * R[:, None])

        return pmb, x

    def getHansonPressureStator(self, x, m, B, Omega, loading, nb, multiplier=1):
        """
        Generic function for computing the hanson formulation of noise for rotors
        loading should be: radial, axial, tangential
        x should be: radius, polar, azimuth

        multiplier is an overall multiplier for total the pressure mode. For B blades it should be B, for one stator/beam it should be 1.
        """
        assert np.all(m != 0), "m=0 (steady loading) is not supported"

        c0 = self.c # SoS

        # todo: make the transformation explicit
        R, theta, phi = x[0], x[1], x[2] # all of size Nx

        radius, twist, chord = self.seg_radius, self.seg_twist, self.seg_chord # all of size # Nr
        dr = self.dr # Nr, size of segment

        mB = m * B # Nm

        wavenumber = mB * Omega / c0 # Nm, issue if mb = 0?

        k = self.k * nb # Nk multiplied by the number of beams!

        Fblade = loading # loading harmonics of size (3, Nk, Nr)

        k = np.concatenate((-k[-1:0:-1], k)) # add the minus part!, shape (2Nk-1 -> Nk)
        Fblade = np.concatenate((np.conjugate(Fblade[:, -1:0:-1]), Fblade), axis=1) # minus loadings are conjugates of positive!

        # p_mB (Nx, Nm)

        # --- explicit broadcasting ---
        R_x      = R[:, None, None, None]          # (Nx, 1, 1, 1)
        theta_x = theta[:, None, None, None]       # (Nx, 1, 1, 1)
        phi_x   = phi[:, None, None, None]         # (Nx, 1, 1, 1)

        mB_m    = mB[None, :, None, None]           # (1, Nm, 1, 1)
        k_k     = k[None, None, :, None]            # (1, 1, Nk, 1)

        radius_r = radius[None, None, None, :]      # (1, 1, 1, Nr)
        dr_r     = dr[None, None, None, :]          # (1, 1, 1, Nr)

        wavenumber_m = wavenumber[None, :, None, None]  # (1, Nm, 1, 1)



        mB_int = np.asarray(m, dtype=np.int64)

        lookup = {val: i for i, val in enumerate(k)}
        idx = np.array([lookup[mb] for mb in mB_int])

        Fphi = Fblade[2, idx, :]
        Fz   = Fblade[1, idx, :]

        Fphi = Fphi[None, :, None, :] # mind the shape change!
        Fz = Fz[None, :, None, :] # mind the shape change!

        # --- matrix construction ---
        matrix = (
            +Fphi * (mB_m - k_k) / radius_r / (wavenumber_m) # issue for m=0 or Omega=0
            +np.cos(theta_x) * Fz
        )

        # --- apply condition: wavenumber=0 and m!=k ---
        # Create a mask that broadcasts correctly to (Nx, Nm, Nk, Nr)
        # mask = (wavenumber_m == 0) & (mB_m != k_k)  # shapes: (1, Nm, 1, 1) & (1, Nm, Nk, 1) -> broadcast to (Nx, Nm, Nk, Nr)
        # matrix = np.where(mask, 0, matrix)

        matrix *= jv(mB_m - k_k, mB_m * Omega * radius_r / c0 * np.sin(theta_x))

        # matrix = matrix.astype(np.complex128)

        matrix *= np.exp(
           -1j * (mB_m - k_k) * (phi_x + np.pi / 2)
            # +1j * (mB_m - k_k) * Omega * R_x / c0 # NOTE: is this correct?
            +1j * (mB_m) * Omega * R_x / c0 # NOTE: is this correct?

        )
        # matrix shape: (Nx, Nm, Nk, Nr)

        # reduce by summing along Nk and Nr axes
        pmb = np.sum (
            matrix
              * dr_r
              ,
            axis=-1
        ) # integrate along the r axis
        pmb = np.sum(pmb, axis=-1) # sum along the k axis
        
        
        # np.sum(matrix, axis=(-1, -2))  # -> (Nx, Nm)

        # scaling
        pmb *= 1j * wavenumber[None, :] * multiplier / (4 * np.pi * R[:, None])
        # * np.exp(1j * wavenumber[None, :] * R[:, None])

        return pmb, x

    def getBladeLoadingPressure(self, x, m, loadings=None):
        return self.getHansonPressure(x, m, self.B, self.Omega,
                                    #    np.conjugate(self.getBladeLoadingHarmonics()),
                                       self.getBladeLoadingHarmonics() if loadings is None else loadings,
                                         nb=self.nbeam, multiplier=self.B) 
    
    def getBeamLoadingPressure(self, x, m, loadings=None):   # stator formulation, a bit shit
        # loads = self.getBeamLoadingHarmonics() # (3, Nk, Nr)
        # pm_build = np.zeros((x.shape[1], m.shape[0]), dtype=np.complex128)
        # for indexm, mval in enumerate(m):
        #     for indexk, k in enumerate(self.k):
        #         if np.abs(k-mval) < 1e-12: #if k is sufficiently close to mB
        #             loadings = np.ones_like(loads) * loads[:, indexk, :][:, None, :] # fill all entries with k==mB!
        #             # loadings[2, :, :] *=-1

        #             pm, _ = self.getHansonPressure(x, np.array([mval]), self.B, self.Omega, 
        #                                            loadings,
        #                                         # np.conjugate(loadings),
        #                                              nb=self.nbeam, multiplier=self.nbeam)
        #             pm_build[:, indexm] = pm[:, 0]
        
        # return pm_build, x

        return self.getHansonPressureStator(x, m, self.B, self.Omega,
                                    self.getBeamLoadingHarmonics()  if loadings is None else loadings,
                                         nb=self.nbeam, multiplier=self.nbeam) 

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

        # Note: two-sided FFT multiplied by sqrt(2) to result in one-sided harmonic forcing
        F_beam_k = self.B * self.Omega / 4 / np.pi * np.sum(F_beam[:, None, :, :] * np.exp(1j *
                 k[None, :, None, None] * self.B * self.Omega * 
                 T_periodic[None, None, :, None]) * dt, axis=2)

        return F_beam_k

    def __getBeamVortexLoads(self, time, Npoints=36):
        """
        time: array, shape (Nt, Nr)
        """
        Nt, Nr = time.shape

        # Lst = np.interp(self.seg_radius / self.r1, r_rt_common, lift_per_unit_span_i) # interpolated from data, size Nr

        # T_per_unit_span = np.interp(self.seg_radius / self.r1, r_rt_common, thrust_per_unit_span_i)
        T_per_unit_span = np.interp(self.seg_radius /self.r1, R_RT_EXACT, DT_EXACT)
        
        Uz =  self.getAxialInducedVelocity()  # Nr

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
        ) # (Nt, Nr), drag!


        Fz = -self.Dcylinder / 2 * np.sum(
            pressure *
            sin_t * 
            deltathetab,
            axis=0
        ) # (Nt, Nr) # lift!

        Fbeam = np.zeros((3, Nt, Nr)) # Note: in the time domain!, size (3, Nt, Nr)
        Fbeam[1, :, :] = Fz
        Fbeam[2, :, :] = Fphi

        return Fbeam

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

        return np.array([R_arr, theta_arr, phi_arr])
    
    def plotDirectivity(self, fig, ax, m:float, valmax=None, valmin=None, R=1.62,
                        Nphi=18, Ntheta=36, blending=0.1, mode='total', loadings=None):

        # --- observation mesh ---
        x = self.getPolarMesh(R=R, Nphi=Nphi, Ntheta=Ntheta)

        # --- pressure / magnitude ---
        if mode=='total':
            pmB1, _ = self.getBladeLoadingPressure(x, np.array([m]), loadings=loadings)      # (Nx,)
            pmB2, _ = self.getBeamLoadingPressure(x, np.array([m]), loadings=loadings)

            pmB = pmB1 + pmB2

        elif mode=='blade':
            pmB, _ = self.getBladeLoadingPressure(x, np.array([m]), loadings=loadings)      # (Nx,)
        elif mode=='beam':
            pmB, _ = self.getBeamLoadingPressure(x, np.array([m]), loadings=loadings)
        else:
            raise ValueError(f'mode {mode} not recognized')

        magnitudes = p_to_SPL(pmB)
        R_arr, theta, phi = x[0], x[1], x[2]          # (Nx,)

        print(f'maximum amplitude: {np.max(magnitudes)} [dB]')

        # --- reshape to grid ---
        magnitudes = magnitudes.reshape(Ntheta, Nphi)
        theta = theta.reshape(Ntheta, Nphi)
        phi   = phi.reshape(Ntheta, Nphi)

        # --- color limits ---
        if valmax is None:
            valmax = magnitudes.max()
        if valmin is None:
            valmin = magnitudes.min()

        r = (magnitudes - valmin) / (valmax - valmin) * (1 - blending) + blending

        # --- unit vectors from spherical coordinates ---
        X = r * np.sin(theta) * np.cos(phi)
        Y = r * np.sin(theta) * np.sin(phi)
        Z = r * np.cos(theta)

        # --- color normalization ---
        norm = colors.Normalize(vmin=valmin, vmax=valmax)
        facecolors = plt.cm.viridis(norm(magnitudes))

        # --- build quad faces ---
        faces = []
        face_colors = []

        for i in range(Ntheta - 1):
            for j in range(Nphi - 1):
                verts = [
                    [X[i, j],     Y[i, j],     Z[i, j]],
                    [X[i+1, j],   Y[i+1, j],   Z[i+1, j]],
                    [X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1]],
                    [X[i, j+1],   Y[i, j+1],   Z[i, j+1]],
                ]
                faces.append(verts)
                face_colors.append(facecolors[i, j])

        poly = Poly3DCollection(
            faces,
            facecolors=face_colors,
            edgecolors='k',
            linewidths=0.5,
            alpha=0.75
        )

        ax.add_collection3d(poly)

        # ax.plot_surface(
        # X, Y, Z,
        # facecolors=facecolors,
        # rstride=1,
        # cstride=1,
        # linewidth=1.,        # line width of cell edges
        # edgecolors='k',        # black edges
        # # antialiased=True,
        # shade=False,
        # alpha=0.75

        #  )


        # --- colorbar ---
        mappable = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        mappable.set_array(magnitudes)
        fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.1,
                    label="Directivity [dB]")

        # --- axes limits ---
        rmax = np.max(r) * 0.65 if np.max(r)>0 else 1.0
        ax.set_xlim(-rmax, rmax)
        ax.set_ylim(-rmax, rmax)
        ax.set_zlim(-rmax, rmax)
        z = np.linspace(-R, R, 2)
        x = np.zeros_like(z)
        y = np.zeros_like(z)

        ax.plot(x, y, z, linewidth=2, color='k', linestyle='dashed', zorder=9)
        if mode != 'blade':
            x = np.linspace(0, R, 10)
            y = np.zeros_like(x)
            z = np.zeros_like(x)
            ax.plot(x, y, z, linewidth=5, color='r', zorder=11)

        # circle of radius R (x–y plane)
        theta_c = np.linspace(0, 2*np.pi, 200)
        ax.plot(
            rmax * np.cos(theta_c) * 1.5,
            rmax * np.sin(theta_c) * 1.5,
            np.zeros_like(theta_c) * 1.5,
            color='k',
            linewidth=1.5,
            linestyle='dashed',
            zorder=10      # higher than other geometry
)

        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()
        ax.set_title(f"Far-field directivity ({mode}), mode $m={m:.0f} \cdot B$")
        ax.view_init(elev=30, azim=-45)

    def plotLoadingHarmonics(self, fig, ax, r_stations, mode='blade', LIFT=False):
        """
        Plot blade loading harmonics as histograms at specified radial stations.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
        ax : array_like of Axes, shape (len(r_stations), 2)
            ax[:,0] -> tangential, ax[:,1] -> axial
        r_stations : array_like
            Radial positions (physical radius) where to plot the harmonics
        """
        if mode == 'blade':
            Fk = self.getBladeLoadingHarmonics()  # (3, Nk, Nr)
        elif mode == 'beam':
            Fk = self.getBeamLoadingHarmonics()  # (3, Nk, Nr)
        else:
            raise ValueError(f'mode {mode} not recognized')
        
        # Fk *= self.dr[None, None, :]
        r_seg = self.seg_radius          # (Nr,)

        Nk = Fk.shape[1]

        for i, r0 in enumerate(r_stations):
            # --- interpolate Fk at this radial station ---
            F_tan = np.array([np.interp(r0 * self.r1, r_seg, Fk[2, k, :]) for k in range(Nk)])
            F_ax  = np.array([np.interp(r0 * self.r1, r_seg, Fk[1, k, :]) for k in range(Nk)])
            F_tan[0] = 0 # ignore the means
            F_ax[0] = 0

            # --- histogram over harmonics k=0..Nk-1 ---
            k_vals = self.k

            ax[0, i].bar(k_vals, np.abs(F_tan), color='tab:blue', alpha=0.7)
            ax[0, i].set_title(f"$r/r_{{tip}}$={r0:.2f}")
            # ax[0, i].set_xlabel("Harmonic k")

            ax[1, i].bar(k_vals, np.abs(F_ax), color='tab:orange', alpha=0.7, label='current')
            # ax[1, i].set_title(f"Axial loading at r={r0:.2f}")
            ax[1, i].set_xlabel("Harmonic k")
            # ax[1, i].set_yscale('log')
            # ax[0, i].set_yscale('log')
            # ax[2, i].set_yscale('log')
            # ax[2, i].set_ylim(1e-5, 1e0)
            # ax[1, i].set_ylim(1e-5, 1e0)
            # ax[0, i].set_ylim(1e-5, 1e0)
            if LIFT:
                ax[2, i].bar(k_vals, np.abs(np.sqrt(F_ax**2 + F_tan**2)), color='tab:green', alpha=0.7, label='current')

            


        ax[0, 0].set_ylabel(r"$|F^{tan}_k|$ [N/m]")
        ax[1, 0].set_ylabel(r"$|F^{ax}_k|$ [N/m]")
        if LIFT:
            ax[2, 0].set_ylabel(r"$|L_k|$ [N]")



        SCALE = 1/DR_EXACT

        if mode == 'blade':
            for i, (datasetax, datasettan) in enumerate(zip([BLH_AX_05, BLH_AX_08, BLH_AX_09], [BLH_TAN_05, BLH_TAN_08, BLH_TAN_09])):
            # for i, (datasetax, datasettan) in enumerate(zip([ BLH_AX_09,  BLH_AX_09, BLH_AX_09], [BLH_TAN_09, BLH_TAN_09, BLH_TAN_09])):

                datasetax = np.array(datasetax)
                datasettan = np.array(datasettan)

                kref = np.arange(len(datasetax))    
                ax[1, i].plot(kref, SCALE * np.abs(datasetax), color='k', linestyle='dashed',marker='x', label='Vella et al. 2026')
                ax[0, i].plot(kref, SCALE * np.abs(datasettan), color='k', linestyle='dashed',marker='x', label='Vella et al. 2026')

                if LIFT:
                    ax[2, i].plot(kref, SCALE * np.abs(np.sqrt(datasetax**2 + datasettan**2)), color='k', linestyle='dashed',marker='x', label='Vella et al. 2026')

                
        elif mode == 'beam':
            for i, (datasetax, datasettan) in enumerate(zip([BLH_BEAM_AX_05, BLH_BEAM_AX_08, BLH_BEAM_AX_09], [BLH_BEAM_TAN_05, BLH_BEAM_TAN_08, BLH_BEAM_TAN_09])):
                datasetax = np.array(datasetax)
                datasettan = np.array(datasettan)

                kref = np.arange(len(datasetax))    
                ax[1, i].plot(kref, SCALE * np.abs(datasetax), color='k', linestyle='dashed',marker='x', label='Vella et al. 2026')
                ax[0, i].plot(kref, SCALE * np.abs(datasettan), color='k', linestyle='dashed',marker='x', label='Vella et al. 2026')
                if LIFT:
                    ax[2, i].plot(kref, SCALE * np.abs(np.sqrt(datasetax**2 + datasettan**2)), color='k', linestyle='dashed',marker='x', label='Vella et al. 2026')
                    
            # ax[1, 1].plot(kref, np.abs(BLH_AX_08), color='k', linestyle='dashed')
            # ax[1, 2].plot(kref, np.abs(BLH_AX_09), color='k', linestyle='dashed')

        if LIFT:
            ax[2, 0].legend()
        else:
            ax[1, 0].legend()

        for axx in ax:
            for axxx in axx:
                axxx.grid(visible=True, which='major', color='k', linestyle='-')
                axxx.grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.5)

        fig.tight_layout()

    def plotPressureSpectrum(self, fig, ax, x:tuple, m:np.ndarray):

        R, theta, phi = x[0], x[1], x[2]

        # --- pressure / magnitude ---
        pmB1, _ = self.getBladeLoadingPressure(x, m)      # (Nx=1, Nm)
        pmB2, _ = self.getBeamLoadingPressure(x, m)

        pmB = pmB1 + pmB2 # shape (Nx=1, Nm)

        


        ax.plot(m, p_to_SPL(pmB[0]), label='total', marker='s', color='k')
        ax.plot(m, p_to_SPL(pmB1[0]), label='blade', marker='s', color='r')
        ax.plot(m, p_to_SPL(pmB2[0]), label='beam', marker='s', color='b')


        ax.minorticks_on()
        ax.grid(which='minor', alpha=0.5)
        ax.grid(which='major')
        ax.legend()

        ax.set_xscale('log')
        ax.set_xlabel('$f^+$')
        ax.set_ylabel('SPL w.r.t 20e-6 Pa')
        plt.tight_layout()


if __name__ == "__main__":
    import h5py
    NSEG = 20 # number of radial prop segments
    kmaxx = 64
    ROBS = 1.62

    # DATAPATH = './Validation/harmonics_ISAE2_D20_L20-1.h5'


    # datafile = h5py.File(DATAPATH, 'r')
    # r = np.array(datafile['r'][:, 0]) # discretization in the radial dir
    # ddr = r[5] - r[4]
    # r_bounds = np.concatenate(([r[0]-ddr], r))+ ddr/2
    # dr = np.diff(r_bounds)



    # VELLA ET AL. 2026
    HANSON_VELLA = HansonModel(twist_rad = np.deg2rad(10 * np.ones(NSEG+1)), chord_m = 0.025 * np.ones(NSEG+1),
                        radius_m=np.linspace(
                            0.016,
                            # 0.1 -0.03 * 0.1 * 20,
                                              0.1, NSEG+1, endpoint=True),
                        # radius_m = r_bounds,
                                                B=2, 
                        Dcylinder_m=0.02, Lcylinder_m=0.02, Omega_rads=8000/60 * 2 * np.pi, rho_kgm3=1.2, c_ms=340., kmax=kmaxx)
    

    # r_stat = np.array([0.5, 0.8, 0.9])
    # fig, ax = plt.subplots(nrows = 3, ncols = len(r_stat), sharey=True, sharex=True)
    # HANSON_VELLA.plotLoadingHarmonics(fig, ax, r_stat,
    #                             #  mode='beam'
    #                              mode='blade',
    #                              LIFT=True
    #                              )
    # plt.show()

    # fig, ax = plt.subplots()
    # HANSON_VELLA.plotPressureSpectrum(fig, ax, np.array([ROBS, np.pi/2, np.pi/2]).reshape(3, 1), np.arange(1, 16, 1))
    # ax.plot(FPLUS, SPL_BLADE_01, label='blade',  color='r', linestyle='dashed') #=> blade has incorrect magnitudes of modes, a "fatter tail"!
    # ax.plot(FPLUS, SPL_BEAM_01, label='beam',  color='b', linestyle='dashed')
    # ax.set_ylim(0, 70)
    # plt.show()
    # plt.close()

    # fig, ax = plt.subplots()
    # HANSON_VELLA.plotPressureSpectrum(fig, ax, np.array([ROBS, np.pi/2, np.pi]).reshape(3, 1), np.arange(1, 16, 1))
    # ax.plot(FPLUS, SPL_BLADE_02, label='blade',  color='r', linestyle='dashed')
    # ax.plot(FPLUS, SPL_BEAM_02, label='beam',  color='b', linestyle='dashed')
    # ax.set_ylim(0, 70)
    # plt.show()
    # plt.close()
    for m in [5]:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        HANSON_VELLA.plotDirectivity(fig, ax, m=m, R=ROBS,
                                valmax=65, valmin=10,
                                Nphi=36*2, Ntheta=18*2,
                                # mode='beam',
                                #   mode='total',
                                mode='blade'
                                )
        plt.show()
        plt.close(fig)

    for m in [5]:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        HANSON_VELLA.plotDirectivity(fig, ax, m=m, R=ROBS,
                                valmax=65, valmin=10,
                                Nphi=36*2, Ntheta=18*2,
                                mode='beam',
                                #   mode='total',
                                # mode='blade'
                                )
        plt.show()
        plt.close(fig)




    for m in [1, 4, 5]:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        HANSON_VELLA.plotDirectivity(fig, ax, m=m, R=ROBS,
                                valmax=65, valmin=10,
                                Nphi=36*2, Ntheta=18*2,
                                # mode='beam',
                                  mode='total',
                                # mode='blade'
                                )
        plt.show()
        plt.close(fig)