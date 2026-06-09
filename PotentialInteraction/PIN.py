import numpy as np
import numpy as np
from scipy.special import jv, fresnel
from scipy.interpolate import interp1d
from Constants.helpers import theodorsen, twoside_spectrum, ifft_periodic, fft_periodic, periodic_sum_interpolated, plot_directivity_contour, p_to_SPL
import matplotlib.pyplot as plt
import warnings


class PotentialInteraction:
    def __init__(self,
                twist_rad:np.ndarray, 
                chord_m:np.ndarray, 
                radius_m:np.ndarray,
                t_c:np.ndarray,
                Fzprime_Npm:np.ndarray,
                Fphiprime_Npm:np.ndarray,
                B,
                Dcylinder_m, Lcylinder_m, Omega_rads,
                rho_kgm3=1.0, c_mps = 340, kmax = 20, nb:float = 1,
                U0_mps:np.ndarray=None, # optional inflow velocity of shape (2, Nr), overwrites momentum theory computations
                numerics = {},
                ):
        self.name = numerics.get('name', None)
        self.B = B
        self.Dcylinder = Dcylinder_m
        self.Lcylinder = Lcylinder_m
        self.Omega=Omega_rads
        self.rho = rho_kgm3
        self.SoS = c_mps # speed of sound
        self.nbeam = nb
        self._numerics = numerics

        self.twist = twist_rad # Nr+1
        self.chord = chord_m # Nr+1
        self.radius = radius_m # Nr+1
        self.r0 = radius_m[0]
        self.r1 = radius_m[-1]

        self.seg_twist = (twist_rad[1:] + twist_rad[:-1]) / 2 # Nr
        self.seg_chord = (chord_m[1:] + chord_m[:-1]) / 2
        self.seg_radius = (radius_m[1:] + radius_m[:-1]) / 2
        self.seg_t_c = (t_c[1:] * chord_m[1:] + t_c[:-1] * chord_m[:-1]) / (chord_m[1:] + chord_m[:-1])

        self.dr = np.diff(radius_m) # (Nr)

        self.Nr = len(twist_rad) - 1
        self.Nk = kmax+1
        self.kmax = kmax
        self.k = np.arange(0, kmax+1, 1) # array of modal orders
    
        self.Fzprime = Fzprime_Npm # Nr
        self.Fphiprime = Fphiprime_Npm # Nr

        if U0_mps is not None:
            self.Ui = U0_mps # (2, Nr) x positive to the right, y positive upwards
        else:
            # TODO: Landgrebe inflow model!

            Uiz = - np.sqrt(self.Fzprime * self.B /  4 / np.pi / self.rho / self.seg_radius) # positive upwards, mind that this should include total loading: B * Fzprime
            Uiz[np.where(Uiz==0)] = 1e-12 # dont divide by zero!
            Uiphi = self.Fphiprime * self.B / 4 / np.pi / self.rho / self.seg_radius / np.abs(Uiz) # positive to the right, Note: dQ is the side force, not torque!
            # Uiphi = np.zeros_like(self.seg_radius) # ignore component!
            self.Ui = np.stack([Uiphi, Uiz]) # shape (2, Nr)
            self.Ui = np.real(self.Ui)

        # pre-compute common arrays
        Nphi = self._numerics.get('Nphi', 360)
        self.phi = np.linspace(-np.pi, np.pi, Nphi, endpoint=False)
        Nthetab = self._numerics.get('Nthetab', 36)
        self.theta_beam = np.linspace(0, 2 * np.pi, Nthetab, endpoint=False) # angles on the cylinder surface, measured from the prop. plane, size (Npoints)

        self.Ur = np.sqrt(
            (self.Omega * self.seg_radius - self.Ui[0])**2 + self.Ui[1]**2
        ) # relative velocity over the blade (mean?), shape Nr

    def updateUi(self, Ui):
        self.Ui = Ui
        self.Ur = np.sqrt(
            (self.Omega * self.seg_radius - self.Ui[0])**2 + self.Ui[1]**2
        )

    def getStrutLoading(self):
        pressure  = self.getStrutPressure() # Nthetab, Nphi, Nr

        thetab = self.theta_beam
        deltathetab = np.diff(thetab)[0]

        Fphi = self.Dcylinder / 2 * np.sum(
            pressure *
            np.cos(thetab)[:, None, None] *
            deltathetab,
            axis=0
        ) # (Nphi, Nr), drag, positive backwards


        Fz = -self.Dcylinder / 2 * np.sum(
            pressure *
            np.sin(thetab)[:, None, None] * 
            deltathetab,
            axis=0
        ) # (Nphi, Nr) # lift, positive upwards

        F_beam = np.zeros((3, self.phi.shape[0], self.seg_radius.shape[0]), dtype=np.complex128) # Note: in the time domain!, size (3, Nt, Nr)
        F_beam[1, :, :] = Fz
        F_beam[2, :, :] = Fphi
    
        return F_beam
    
    def getStrutLoadingHarmonics(self):
        """"
        strut loading in N/m along the strut radial stations
        returns: [Fr positive outwards, Fphi positive opposite to blade rotation, Fz positive upsteam]
        """
        
        F_beam = self.getStrutLoading() # shape (3, Nt, Nr)

        # period_vortex = 2 * np.pi / self.B / self.Omega # period with which a blade passes over the beam: i.e. T/B !
        # period_global = 2 * np.pi / self.Omega # rotation period, DIFFERENT FROM blade passage period!
        # N_periods = int(period_global / period_vortex)

        # phi_extended = np.linspace(self.phi[0], self.phi[0] + N_periods * np.pi * 2, self.phi.shape[0] * N_periods, endpoint=False) # extended phi array covering multiple periods, shape (Np_extended,)
        # F_beam_extended = np.tile(F_beam, (1, N_periods, 1)) # extend the time series to multiple periods, shape (3, Nt*N_periods, Nr)
        # dt = np.diff(phi_extended)[0] / self.Omega # timestep corresponding to phi array
        # F_beam_k_global = 1/period_global/N_periods * np.sum(F_beam_extended[:, None, :, :] * np.exp(+1j *
        #         self.k[None, :, None, None] * 2 * np.pi / period_global * 
        #          (phi_extended/self.Omega)[None, None, :, None]) * dt, axis=2)

        period_global = 2 * np.pi / self.Omega # rotation period, DIFFERENT FROM blade passage period!
        phi = self.phi
        dt = np.diff(self.phi)[0] / self.Omega # timestep corresponding to phi array
        F_beam_k_global = 1/period_global * np.sum(F_beam[:, None, :, :] * np.exp(+1j *
                self.k[None, :, None, None] * 2 * np.pi / period_global * 
                 (phi/self.Omega)[None, None, :, None]) * dt, axis=2)
        
        return F_beam_k_global
    
    def getGammaInPhi(self):
        """
        get QS-unsteady circulation as a function of phi
        """

        do_gamma_steady = self._numerics.get('gamma_steady', False) # do we consider QS or S gamma?

        gamma_k = self.getGammaHarmonics() #Nk, Nr
        gamma_kk, kk = twoside_spectrum(gamma_k, self.k) # get two-sided spectrum
        gamma_phi, phi = ifft_periodic(gamma_kk, 2 * np.pi, self.phi, kk) # Nphi, Nr

        if do_gamma_steady: 
            gamma_phi = np.ones_like(phi)[:, None] * np.mean(gamma_phi, axis=0)[None, :] # fill the gamma array with means along phi

        return gamma_phi

    def getGammaHarmonics(self):
        # quasi=steady loading: circulatory!
        BLH = self.getBladeLoadingHarmonics(QS=True) # 3, Nk, Nr
        Ur = self.Ur

        BLH_angle = np.arctan2(np.abs(BLH[2, :, :]), np.abs(BLH[1, :, :])) # phi / z 
        BLH_magnitude = BLH[1, :, :] / np.cos(BLH_angle) # magnitude of the loading harmonics, shape (Nk, Nr)

        gamma_k = BLH_magnitude / self.rho / Ur # Nk, Nr -  gamma in the frequency domain
        
        # testing
        # gamma_k[0, :] = self.Fzprime / self.rho / Ur # overwrite!
        # gamma_k[1:, :] = 0.0

        return gamma_k

    def getRankineParams(self):
        """
        get the source spacing b and source strength Lambda of the rankine oval approximating thickness noise

        Note: b could be extended to be a complex number, such that to account for the inflow being at an angle.
        Formally, the inflow should be normal to the oval axis at all times, currently we assume Omega * r >> Ui over the path
        """

        # source strength such that the oval extends t/c in the axis normal
        # Ur = np.sqrt((self.Omega * self.seg_radius - self.Ui[0])**2 + self.Ui[1]**2) # Nr
        Ur = self.Ur
        Lambda = Ur * self.seg_t_c * self.seg_chord # Nr

        # source spacing that sets the axial extent of the oval to self.seg_chord
        b = 0j + self.seg_chord / 2 * (
            np.sqrt(1 + (1 / np.pi * self.seg_t_c)**2)
            - (1 / np.pi * self.seg_t_c)
        ) # strictly < c/2!, size Nr
        b *= np.exp(1j * (np.angle((self.Omega * self.seg_radius - self.Ui[0]) - 1j * self.Ui[1]))) # source: i made it up: b such that relative inflow is parallel to the oval axis

        return Lambda, b

    def getDoubletParams(self):
        """
        parameters of a doublet flow corresponding to a cylinder of radius t/c/2
        """

        Ur = self.Ur # Nr
        # radius_doublet_squared = self.seg_t_c * self.seg_chord / 2 # Nr,  half of mean thickness: radius of the doublet in inflow!
        # + self.seg_chord / 2 

        # radius_doublet_squared = 1 / np.pi * self.seg_chord / 2 * (
        #     np.sqrt(1 + (1 / np.pi * self.seg_t_c)**2)
        #     - (1 / np.pi * self.seg_t_c)
        # ) * self.seg_t_c * self.seg_chord # consistent with the source-sink pair!

        radius_doublet_squared = self.seg_t_c * self.seg_chord ** 2 / 2 / np.pi #

        # mu = radius_doublet ** 2 * np.abs(Ur) # Nr
        # Ucomplex = self.Omega * self.seg_radius - self.Ui[0] - 1j * self.Ui[1]
        Ucomplex = self.Omega * self.seg_radius
        mu = radius_doublet_squared * Ucomplex # Nr, doublet strength, accounting for orientation of the inflow

        return mu

    def getStrutPressure(self,):
        """
        returns pressure distribution over the strut, as a function of r, phi, and theta (beam polar angle w.r.t rotor plane)
        """

        include_thickness_sources=self._numerics.get('include_thickness_sources', False)
        include_vortex_sources=self._numerics.get('include_vortex_sources', True)

        
        gamma = self.getGammaInPhi() # shape (Nphi, Nr) - quasi-steady-unsteady vortex strength



        thetab = self.theta_beam
        deltathetab = np.diff(thetab)[0]

        z = self.Dcylinder/2 * np.exp(1j * thetab) # positions along the cylinder surface, complex, size (Nthetab)
        zprime = self.Dcylinder**2 / 4 / z # circle conjugate

        # complex derivative of the complex potential dfdz = u - i * v (which follows from f being holomorphic on Z!=Zv), size (Npoints, Nt, Nr)
        # dfdz = -1j * gamma_e / 2 / np.pi / (Z_e-Zv_e) + 1j * gamma_e / 2 / np.pi / (Z_circ_conjugate_e - Zvbar_e) * (-Z_circ_conjugate_e / Z_e) + 1j * Uz_e * (1 + Z_circ_conjugate_e / Z_e)

        # inflow part
        Uimag = np.linalg.norm(self.Ui, axis=0) # Nr
        alpha0 = np.arctan2(self.Ui[0], -self.Ui[1]) # Nr

        # TODO: check for errors
        vortex_period = 2 * np.pi / self.B / self.Omega # vortex passage period
        pressure = np.zeros((thetab.shape[0], self.phi.shape[0], self.seg_radius.shape[0]), dtype=np.complex128) # Nthetab, Nphi, Nr
        dfdz = np.zeros((thetab.shape[0], self.phi.shape[0], self.seg_radius.shape[0]), dtype=np.complex128) # Nthetab, Nphi, Nr
        
        # add the mean flow term
        dfdz += 1j * Uimag[None, None, :] * (np.exp(-1j * alpha0[None, None, :]) + np.exp(1j * alpha0[None, None, :]
                    ) * zprime[:, None, None] / z[:, None, None]) 
        
        # thickness variables
        # Lambda, b = self.getRankineParams()
        mu = self.getDoubletParams()

        for vortex_index in range(-10, 10, 1): # sum an arbitrary amount of vortices, further ones should be negligible
            
            # vortex position, complex, size (Nphi, Nr), vortex is moving from negative x to positive with speed Omega * r
            # phased vortices: shift the passage time by vortex_index * T/B

            zv = self.seg_radius[None, :] * (self.phi[:, None] + vortex_index * vortex_period * self.Omega
                                             
                                             + self.seg_chord[None, :] / 4 / self.seg_radius[None, :] # position the vortex at quarter-chord!

                                             ) + 1j * self.Lcylinder 
            zvbar = np.conjugate(zv) # complex conjugate

            if include_vortex_sources:

                # add the linear contribution to dfdz
                phi = self.phi
                shift = vortex_index * vortex_period * self.Omega
                shifted_phi = (phi - shift) % (2 * np.pi)

                # sort once
                sort_idx = np.argsort(shifted_phi)
                phi_sorted = shifted_phi[sort_idx]

                gamma_shifted = np.apply_along_axis(
                    lambda g: np.interp(phi, phi_sorted, g[sort_idx], period=2*np.pi),
                    axis=0,
                    arr=gamma
                )
                
                dfdz_vortex = -1j * gamma_shifted[None, :, :] / 2 / np.pi / (z[:, None, None] -
                        zv[None, :, :]) + 1j * gamma_shifted[None, :, :] / 2 / np.pi / (zprime[:, None, None] - 
                        zvbar[None, :, :]) * (-zprime[:, None, None] / z[:, None, None]) # Nthetab, Nr, Nphi
                dfdz += dfdz_vortex

                pressure_vortex = self.rho * gamma_shifted[None, :, :] * self.Omega * self.seg_radius[None, None, :] / 2 / np.pi * np.real(
                    1j / zvbar[None, :, :] + 1j / (zv[None, :, :] - z[:, None, None]) - 1j / (zvbar[None, :, :] - zprime[:, None, None])
                ) # (Nthetab, Nphi, Nr)
                
                pressure += pressure_vortex # add the linear contribution to the pressure


            # thickness contribution
            # source: i made it up 
            if include_thickness_sources:

                # source-sink
                # zsp = zv + b # source location
                # zsn = zv - b # sink location
                # zspbar = np.conj(zsp)
                # zsnbar = np.conj(zsn)

                # ##### TODO: Lambda conj?
                # # dfdz due to a sum of source at zsp and sink at zsn of strength Lambda
                # dfdz_sourcesink = Lambda[None, None, :] / 2 / np.pi * ( 1 / (z[:, None, None] - zsp[None, :, :]) - 1 / (z[:, None, None] - zsn[None, :, :])
                # ) + Lambda[None, None, :] / 2 / np.pi * (1 / (zprime[:, None, None] - zspbar[None, :, :]) - 1 / (zprime[:, None, None] - zsnbar[None, :, :])
                # ) * (-zprime[:, None, None] / z[:, None, None])

                # dfdz += dfdz_sourcesink

                # pressure_sourcesink = np.real( self.Omega * self.seg_radius[None, None, :] * Lambda[None, None, :] / 2 / np.pi * (
                #     1 / zsp[None, :, :] - 1 / zsn[None, :, :] - 1 / (zsp[None, :, :] - z[:, None, None]) + 1 / (zsn[None, :, :] - z[:, None, None])
                #      - 1 / (zspbar[None, :, :] - zprime[:, None, None]) + 1 / (zsnbar[None, :, :] - zprime[:, None, None])
                # ) )
                # pressure += pressure_sourcesink

                #doublet
                zd = zv
                zdbar = np.conj(zd)

                dfdz_doublet = - mu[None, None, :] / ( z[:, None, None] - zd[None, :, :] ) ** 2 + ( zprime[:, None, None] / z[:, None, None] ) * ( 
                                np.conj( mu[None, None, :] ) / (( zprime[:, None, None]  - zdbar[None, :, :] ) ** 2) )

                dfdz += dfdz_doublet

                pressure_doublet = np.real( self.Omega * self.seg_radius[None, None, :] * ( mu[None, None, :] / ( z[:, None, None] - zd[None, :, :] ) ** 2 + np.conj(
                     mu[None, None, :] ) / ( ( zprime[:, None, None]  - zdbar[None, :, :] ) ** 2 ) ) )

                pressure += pressure_doublet

        u, v = np.real(dfdz), -np.imag(dfdz)
        U = np.sqrt(u**2 + v**2) # (Nthetab, Nphi, Nr)


        only_linear = self._numerics.get('only_linear', False)
        only_nonlinear = self._numerics.get('only_nonlinear', False)
        
        pressure_dynamic = 0.5 * self.rho * (Uimag**2 - U**2) # total!

        if only_linear:
            output = pressure
        elif only_nonlinear:
            output = pressure_dynamic
        else:
            output = pressure + pressure_dynamic

        return output # Nthetab, Nphi, Nr

    def getBladeLoadingHarmonics(self, QS=False):
        """
        returns loading acting ON the blade in the frequency domain, according to Sears
        result of shape (3, Nk, Nr)
        3: radial, axial, tangential
        k's correspond to frequencies k * Omega
        QS - if True, use quasi-steady assumption, else use full unsteady Sears function
        """

        wk = self.getBladeDownwashHarmonics()

        Ur = self.Ur # Nr
        k = self.k
        # --- Theodorsen arguments ---
        sigma = k[:, None] * self.Omega * self.seg_chord[None, :] / (2.0 * Ur[None, :])      # (Nk, Nr)
        beta = np.pi/2 - self.seg_twist # alpha in Wu et al. 2022, shape Nr

        mu = (
            1j * k[:, None]  * self.seg_chord[None, :] / (2.0 * self.seg_radius[None, :])
            * np.exp(-1j * beta[None, :]) # TODO: check which form correct: me or Riccardo's?
            )                                           # (Nk, Nr)
        Nk, Nr = wk.shape
        US_TERM = np.ones((Nk, Nr), dtype=np.complex128)
        US_TERM[0, :] = 1.0

        if not QS:
            C = theodorsen(sigma[1:, :])
            T1 = jv(0, mu[1:, :]) - 1j * jv(1, mu[1:, :])
            T2 = 1j * sigma[1:, :] / mu[1:, :] * jv(1, mu[1:, :])
        else:
            # quasi-steady
            C = 1.0
            T1 = jv(0, mu[1:, :]) - 1j * jv(1, mu[1:, :])
            T2 = 0.0


        US_TERM[1:, :] = np.conjugate(C * T1 + T2)
        
        Lkprime = (
            np.pi * self.rho * self.seg_chord[None, :] * Ur[None, :]
            * wk
            * US_TERM
        )                                       # (Nk, Nr)

        # --- allocate blade forces ---
        
        Fblade = np.zeros((3, Nk, Nr), dtype=np.complex128) # radial, axial, tangential
        Fblade[1, :, :] = Lkprime * np.cos(self.seg_twist[None, :]) # positive UPWARDS
        Fblade[2, :, :] = Lkprime * np.sin(self.seg_twist[None, :]) # DRAG, oriented BACKWARDS


        # steady loads.
        Fblade[1, 0, :] = self.Fzprime # axial, positive upwards, NOTE: Fzprime is PER BLADE, so is Fphiprime
        Fblade[2, 0, :] = self.Fphiprime # tangential, positive backwards

        return Fblade

    def getBladeLoading(self, QS=False):
        
        BLH = self.getBladeLoadingHarmonics(QS=QS) # 3, Nk, Nr

        period = 2 * np.pi / self.Omega
        time = np.linspace(-period/2, period/2, 1000)

        Nt = len(time)
        Nr = len(self.seg_radius)
        BL = np.zeros((3, Nt, Nr))
        for i in range(3):
            BLH_twoside, k_twoside = twoside_spectrum(BLH[i], self.k)
            BLH_twoside = BLH_twoside
            BLi, time = ifft_periodic(BLH_twoside, period, time, k_twoside)

            BL[i, :, :] = BLi

        return BL, time # 3, Nt, Nr and Nt

    def getBladeLoadingHarmonicsAmiet(self, chord_stations=None, dc=None):
        """
        return Amiet's prediction of US loading of a flat plate, assuming a spanwise gust k2 = 0, and thus THETA = infinity and the gust is supercritical
        
        result of shape (3, Nk, Nr, Nc) with chordwise stations of size Nc,
        optionally, the chordwise array (ranging from -1 to 1) can be provided as an input
        """

        Mach = np.sqrt(
         (self.Omega * self.seg_radius - self.Ui[0])**2 + 
         self.Ui[1]**2 
        ) / self.SoS # shape Nr

        beta = np.sqrt(1-Mach**2) # Nr

        k = self.k
        kprime = k[:, None] * self.seg_chord[None, :] / 2 # Nk, Nr

        # TODO: REWRITE THIS FULLY BASED ON MISH & DEVENPORT 2006
        
        mu = Mach[None, :] / beta[None, :]**2 * k[:, None] # Nk, Nr
        kappa = mu # assuming k2=0, shape Nk, Nr

        if chord_stations is not None and dc is not None:
            chord_stations = chord_stations 
            dc = dc # TODO: remove redundancy
        else:
            theta = np.linspace(0, np.pi, 101)
            chord_stations_outer = -np.cos(theta)
            chord_stations = (chord_stations_outer[1:] + chord_stations_outer[:-1]) / 2
            dc = np.diff(chord_stations_outer)

        x = (1+chord_stations) / 2 # ranging from 0 to 1

        wk = self.getBladeDownwashHarmonics() # harmonics of downwash at blade station, measured at half-chord only, shape Nk, Nr

        S, C = fresnel(2 * 1j * (1/np.pi * (2-x[None, None, :]) * (kappa[:, :, None])**(0.5))) # shape Nk, Nr, Nc each
        E = C + 1j * S # assemble the complex term

        # S, C = fresnel((2 * kappa[:, :, None] * (1-chord_stations[None, None, :])/ np.pi)**(0.5)) # shape Nk, Nr, Nc each
        # E = -1j * C + 1j * -1j * S # assemble the complex term

        # all of shape Nk, Nr, Nc
        fk = 1 - (x[None, None, :]/2)**(0.5) * (1 - (1-1j) * E) # supercritical case

        gk = -fk / np.pi / beta[None, :, None] * (np.pi * x[None, None, :] * (1j * kappa[:, :, None] + 1j * (Mach[None, :, None] * mu[:, :, None] + kprime[:, :, None])))**(-0.5)

        lift_per_unit_area_k = 2 * np.pi * self.rho * self.SoS * Mach[None, :, None] * wk[:, :, None] * gk

        # lift_per_unit_area_k = 2 * self.rho * self.SoS * Mach[None, :, None] * wk[:, :, None] * np.exp(1j *
        #         np.pi/4) / np.sqrt(2 * np.pi * kprime[:, :, None] + beta[None, :, None]**2 * kappa[:, :, None]) * (
        #         1 - np.sqrt(2 / (1 + chord_stations[None, None, :])) - (1-1j) * E
        #     ) * np.exp(-1j * (Mach[None, :, None] * mu[:, :, None] - kappa[:, :, None]) * (1+chord_stations[None, None, :]))

        loading_harmonics_per_unit_area = np.zeros((3, k.shape[0], self.seg_radius.shape[0], chord_stations.shape[0]), dtype=np.complex128)

        # same as above
        loading_harmonics_per_unit_area[1, 1:, :, :] = lift_per_unit_area_k[1:, :, :]  * np.cos(self.seg_twist[None, :, None])
        loading_harmonics_per_unit_area[2, 1:, :, :] = lift_per_unit_area_k[1:, :, :]  * np.sin(self.seg_twist[None, :, None])

        loading_harmonics_per_unit_area[1, 0, :, :] = self.getSearsLoadingDistribution(self.Fzprime, chord_stations) # map the net loads to "reasonable" distributions from Sears
        loading_harmonics_per_unit_area[2, 0, :, :] = self.getSearsLoadingDistribution(self.Fphiprime, chord_stations)

        # "net loading", used only for debugging
        # note this is not meaningfull for acoustics since for an acoustic source we should not be able to 

        # loading_harmonics_per_unit_area of shape 3, Nk, Nr, Nc
        loading_harmonics_per_unit_span = np.sum(loading_harmonics_per_unit_area * dc[None, None, None, :] * self.seg_chord[None, None, :, None] / 2, axis=-1) # shape 3, Nk, Nr, unit N/m

        return loading_harmonics_per_unit_area, loading_harmonics_per_unit_span, k, self.seg_radius, chord_stations # return array and all its inputs

    def getSearsLoadingDistribution(self, loading, chord_stations):
        """
        given the net load in units N/m, compute the loading distribution along the chord in units N/m^2

        loading - array of shape Nr of net sectional loading harmonics
        chord_stations - locations to compute the distributed load at of shape Nc, representing non-dimensional chordwise locations from -1 to 1

        note: chord stations should be "well conditioned" for the reverse (integration along the chord) to work properly
        consider a transform in the form x = -cos(theta) for theta in (0, pi) to cluster points near LE
        """

        # dc = np.diff(chord_stations)[0] * self.chord / 2 # assumed uniform!, shape Nr

        # loading_per_unit_area = 2 / self.seg_chord[:, None] / np.pi * np.sqrt((1-chord_stations[None, :])/(1+chord_stations[None, :])) * loading[:, None] # units of N/m^2, shape of Nk, Nr, Nc

        # return loading_per_unit_area

        theta_stations = np.arccos(chord_stations) # map chordwise stations to theta space, where Sears kernel is defined, shape Nc

        # Sears kernel in θ-space (no sqrt singularity)
        kernel = np.tan(0.5 * theta_stations)[None, :]

        loading_per_unit_area = (
            2 / self.seg_chord[:, None] / np.pi
            * kernel
            * loading[:, None]
        )

        return loading_per_unit_area

    def getBladeDownwash(self):
        """
        get downwash at the blade station due to uniform inflow over the cylinder, in m/s, in the time domain
        """

        Uimag = np.linalg.norm(self.Ui, axis=0) # Nr
        alpha0 = np.arctan2(self.Ui[0], -self.Ui[1]) # Nr
        # check that:
        # Ui[0] = Uimag * np.sin(alpha0)
        # Ui[1] = -Uimag * np.cos(alpha0)

        Ds = self.Dcylinder
        Ls = self.Lcylinder

        # correct for position of c/4 - the shift may be significant if 1) the chord is large 2) the twist is large
        # Leff = Ls + self.seg_chord / 4 * np.sin(self.seg_twist) # distance to quarter chord!
        Leff = Ls * np.ones_like(self.seg_radius)

        # complex variable
        # Note: here we need the assumption r * pi >> Lcylinder!
        Nphi = self._numerics.get('Nphi', 360)
        N = self._numerics.get('Nperiods', 10) # arbitrary, N>2 should be okay

        phi_long = np.linspace(-N * np.pi, N * np.pi, Nphi * N, endpoint=False)

        z = 1j * Leff[:, None] + self.seg_radius[:, None] * phi_long[None, :] # Nr, Nphi

        # CIRCLE conjugate
        zprime = Ds**2 / 4 / z

        # COMPLEX conjugate
        # zbar = np.conj(z)

        # potential due to uniform inflow over a circle at an angle
        # f = 1j * Uimag * (z * np.exp(-1j * alpha0) - Ds**2 / 4 / z * np.exp(1j * alpha0))

        # derivative of potential: df/dz = u - 1j * v, explicit because why wouldn't we
        dfdz = 1j * Uimag[:, None] * (np.exp(-1j * alpha0[:, None]) + np.exp(1j * alpha0[:, None]) * zprime / z) # Nr, Nphi

        u, v = np.real(dfdz), -np.imag(dfdz)

        w_vec = np.stack([
            # u - Uimag[:, None] * np.sin(alpha0)[:, None], # u should approach Uimag * sin(alpha0) at infinity
            # v + Uimag[:, None] * np.cos(alpha0)[:, None]  # v sgould approach -Uimag * sin(alpha0) at infinity
            u - self.Ui[0, :, None], 
            v - self.Ui[1, :, None] 
        ])

        # sum periodically to extract the periodic downwash!
        # NEED A LONGER ARRAY   
        _, w_periodic = periodic_sum_interpolated(np.swapaxes(w_vec, 1, 2), period=2 * np.pi, time=phi_long, kind='cubic', t_new=self.phi)
        w_periodic = np.swapaxes(w_periodic, 1, 2)

        # alpha = self.seg_twist[:, None]

        alpha = np.arctan2(-self.Ui[1], self.Omega * self.seg_radius - self.Ui[0])[:, None] # Nr, None
        w_normal = w_periodic[0] * (-np.sin(alpha)) + w_periodic[1] * np.cos(alpha) # Nr, Nphi


        return w_normal
    
    def getBladeDownwashHarmonics(self):

        w = self.getBladeDownwash() # Nr, Nphi
        k = self.k # Nk
        phi = self.phi # Nphi

        dphi = np.diff(self.phi)[0]
        wk = 1 / 2 / np.pi * np.sum(
            dphi * np.exp(1j * k[None, None, :] * self.phi[None, :, None])*
            w[:, :, None], axis=1 # reduce the phi axis, result of shape Nr, Nk
        )
        wk = wk.T # Nk, Nr for consistency

        return wk

    def plotDownwashInRotorPlane(self, fig=None, ax=None):
        w_normal = self.getBladeDownwash() # Nr, Nphi
        r = self.seg_radius
        phi = self.phi

        # plot a polar contour of the downwash magnitude
        # also plot lines at y = +- self.Dcylinder/2 for x>0 to show cylinder location
        # if fig, ax not specified, create new ones
        # return fig, ax
            # Create meshgrid for polar plotting
        Phi, R = np.meshgrid(phi, r)

        # Create figure/axis if not provided
        if fig is None or ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

        # Plot contour (polar)
        Nlevels = 51
        wmax = np.max(np.abs(w_normal))
        levels = np.linspace(-wmax, wmax, Nlevels)
        contour = ax.contourf(Phi, R, w_normal, levels=levels, cmap='seismic')
        fig.colorbar(contour, ax=ax, label="Downwash [m/s]")

        # --- Plot cylinder projection lines ---
        # Convert to Cartesian for overlay logic
        x = R * np.cos(Phi)
        y = R * np.sin(Phi)

        # Cylinder half-height
        y_cyl = self.Dcylinder / 2

        # Create mask for x > 0
        mask = x > 0

        # Plot y = +Dcylinder/2
        ax.plot(
            np.arctan2(y_cyl, x[mask]),
            np.sqrt(x[mask]**2 + y_cyl**2),
            'k--',
            linewidth=1
        )

        # Plot y = -Dcylinder/2
        ax.plot(
            np.arctan2(-y_cyl, x[mask]),
            np.sqrt(x[mask]**2 + y_cyl**2),
            'k--',
            linewidth=1
        )

        # Formatting
        # ax.set_title("Downwash in Rotor Plane")
        ax.set_ylim(0, np.max(r))

        return fig, ax
        
    def plotStrutLoading3D(self, fig=None, ax=None):
        F_beam = self.getStrutLoading()  # shape (3, Nt=Nphi, Nr)
        phi = self.phi
        r_rt = self.seg_radius / self.r1  # r/rtip

        component_labels = ['$F_r$ (radial)', '$F_z$ (axial)', '$F_\phi$ (tangential)']
        # component_cmaps  = ['RdBu_r', 'RdBu_r', 'RdBu_r']
        component_cmaps  = ['viridis', 'viridis', 'viridis']


        # Build meshgrid in polar coords
        PHI, R = np.meshgrid(phi, r_rt, indexing='ij')   # (Nphi, Nr)
        # X = R * np.cos(PHI)
        # Y = R * np.sin(PHI)

        X = R
        Y = PHI


        if fig is None or ax is None:
            fig, axes = plt.subplots(
                1, 3,
                figsize=(18, 5),
                subplot_kw={'projection': '3d'}
            )
        else:
            # ax should be a list/array of 3 Axes3D
            axes = ax

        for i, (axi, label, cmap) in enumerate(zip(axes, component_labels, component_cmaps)):
            Z = np.real(F_beam[i])                        # (Nphi, Nr)
            magnitude = np.abs(Z)

            vmax = np.max(np.abs(Z))

            norm = plt.Normalize(vmin=-vmax, vmax=vmax)
            colors = plt.get_cmap(cmap)(norm(Z))  # (Nphi, Nr, 4) RGBA

            surf = axi.plot_surface(
                X, Y, Z,
                facecolors=colors,
                rstride=1, cstride=1,
                linewidth=0,
                antialiased=True,
                shade=False
            )

            # Colorbar
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array(Z)
            fig.colorbar(mappable, ax=axi, shrink=0.55, aspect=12, pad=0.1,
                        label='Force [N]')

            axi.set_title(label, fontsize=11)
            # axi.set_xlabel('x/r$_{tip}$')
            # axi.set_ylabel('y/r$_{tip}$')
            axi.set_xlabel('$r/r_{tip}$')
            axi.set_ylabel('$\phi=t\Omega$')
            
            axi.set_zlabel('F [N/m]')
            axi.view_init(elev=15, azim=75-60+180)

        fig.suptitle('Strut beam loading — polar surface', fontsize=13, y=1.01)
        plt.tight_layout()
        return fig, axes

    def plotStrutLoading2D(self, r_query, fig=None, ax=None):
        """
        Plot strut loading vs azimuth at a given radial station (r/r_tip),
        using scipy interpolation in the radial direction.

        r_query - non-dimensional radius (r/rt)
        """

        F_beam = self.getStrutLoading()  # (3, Nphi, Nr)
        phi = self.phi                   # (Nphi,)
        r_rt = self.seg_radius / self.r1 # (Nr,)

        component_labels = ['$F_r$ (radial)', '$F_z$ (axial)', '$F_\\phi$ (tangential)']

        # --- Bounds check ---
        r_min, r_max = np.min(r_rt), np.max(r_rt)
        if not (r_min <= r_query <= r_max):
            warnings.warn(
                f"Requested r/r_tip={r_query:.3f} outside [{r_min:.3f}, {r_max:.3f}] → extrapolating",
                RuntimeWarning
            )

        # --- Interpolator (vectorized over phi) ---
        # axis=2 → interpolate along radial direction
        interp_fun = interp1d(
            r_rt,
            F_beam,
            axis=2,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )

        # Result: (3, Nphi)
        F_interp = interp_fun(r_query)

        # --- Plotting ---
        if fig is None or ax is None:
            fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
        else:
            axes = ax

        for i, (axi, label) in enumerate(zip(axes, component_labels)):
            axi.plot(phi, np.real(F_interp[i]), label='Real', color='k', linestyle='dashed')
            # axi.plot(phi, np.imag(F_interp[i]), '--', label='Imag')

            axi.set_ylabel('F [N/m]')
            axi.set_title(label)
            axi.grid(True)

            # if i == 0:
                # axi.legend()

        axes[-1].set_xlabel(r'$\phi = t\Omega$')

        fig.suptitle(f'Strut loading at r/r_tip = {r_query:.3f}', fontsize=13)
        plt.tight_layout()

        return fig, axes
    
    def plotStrutLoadingHarmonics2D(self, r_query, fig=None, ax=None):
        """
        Plot strut loading vs wavenumber at a given radial station (r/r_tip),
        using scipy interpolation in the radial direction.

        Parameters
        ----------
        r_query : float
            Non-dimensional radius (r/r_tip)
        """

        F_beam = self.getStrutLoadingHarmonics()   # (3, Nk, Nr)
        k = self.k                        # (Nk,)
        r_rt = self.seg_radius / self.r1  # (Nr,)

        component_labels = [
            '$F_r^k$ (radial)',
            '$F_z^k$ (axial)',
            '$F_\\phi^k$ (tangential)'
        ]

        # --- Ensure monotonic radius for interpolation ---
        if not np.all(np.diff(r_rt) > 0):
            idx = np.argsort(r_rt)
            r_rt = r_rt[idx]
            F_beam = F_beam[:, :, idx]

        # --- Bounds check ---
        r_min, r_max = np.min(r_rt), np.max(r_rt)
        if not (r_min <= r_query <= r_max):
            warnings.warn(
                f"Requested r/r_tip={r_query:.3f} outside [{r_min:.3f}, {r_max:.3f}] → extrapolating",
                RuntimeWarning
            )

        # --- Interpolation along radial direction ---
        interp_fun = interp1d(
            r_rt,
            F_beam,
            axis=2,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )

        # Result: (3, Nk)
        F_interp = interp_fun(r_query)

        # --- Plotting ---
        if fig is None or ax is None:
            fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
        else:
            axes = ax

        for i, (axi, label) in enumerate(zip(axes, component_labels)):
            axi.plot(k, np.abs(F_interp[i]), label='|F|', color='k', marker='x')
            axi.plot(k, np.real(F_interp[i]), '--', label='Real', color='r')
            axi.plot(k, np.imag(F_interp[i]), ':', label='Imag', color='b')

            axi.set_ylabel('F [N/m]')
            axi.set_title(label)
            axi.grid(True)

            if i == 0:
                axi.legend()

        axes[-1].set_xlabel('Wavenumber $k$')

        fig.suptitle(f'Strut loading harmonics at r/r_tip = {r_query:.3f}', fontsize=13)
        plt.tight_layout()

        return fig, axes

    def getStrutPressureHarmonics(self):

        pressure = self.getStrutPressure() # Nthetab, Nphi, Nr

        # period_vortex = 2 * np.pi / self.B / self.Omega # period with which a blade passes over the beam: i.e. T/B !

        period_global = 2 * np.pi / self.Omega # rotation period, DIFFERENT FROM blade passage period!

        # N_periods = int(period_global / period_vortex)
        # phi_extended = np.linspace(self.phi[0], self.phi[0] + N_periods * np.pi * 2, self.phi.shape[0] * N_periods, endpoint=False) # extended phi array covering multiple periods, shape (Np_extended,)
        # pressure_extended = np.tile(pressure, (1, N_periods, 1)) # extend the time series to multiple periods, shape (3, Nt*N_periods, Nr)
        # dt = np.diff(phi_extended)[0] / self.Omega # timestep corresponding to phi array
        # pressure_global = 1/period_global/N_periods * np.sum(pressure_extended[:, None, :, :] * np.exp(+1j *
        #         self.k[None, :, None, None] * 2 * np.pi / period_global * 
        #          (phi_extended/self.Omega)[None, None, :, None]) * dt, axis=2)

        phi = self.phi
        dt = (self.phi[1] - self.phi[0]) / self.Omega
        pressure_k_global = 1/period_global * np.sum(pressure[:, None, :, :] * np.exp(+1j *
        self.k[None, :, None, None] * 2 * np.pi / period_global * 
        (phi/self.Omega)[None, None, :, None]) * dt, axis=2) # shape Nthetab, Nk, Nr

        return pressure_k_global
    
    def plotSurfacePressureContour(self, m:int, fig=None, ax=None, show_discretization=False, levels=21):
        p = self.getStrutPressureHarmonics() # p_k of shape (Ntheta, Nk, Nr)

        thetab = self.theta_beam
        radius = self.seg_radius

        index = np.where(self.k == m)[0][0] # index of the desired mode 
        # if does not exist, will throw an error, which is fine
        pk = p[:, index, :]

        # # --- shift theta from [0, 2π) → [-π, π) ---
        # th_shifted = (thetab + np.pi) % (2*np.pi) - np.pi

        # # --- sort theta so it increases from -π to π ---
        # sort_idx = np.argsort(th_shifted)

        # th_sorted = th_shifted[sort_idx]
        th_sorted = thetab

        # pk = pk.reshape(len(thetab), len(radius))
        # pk = pk[sort_idx,]
        # pk = pk.flatten()

        TH, PHI = np.meshgrid(th_sorted, radius, indexing='ij')

        fig, ax, mappable = plot_directivity_contour(Theta=np.rad2deg(TH), Phi=PHI, magnitudes=pk, fig=fig, ax=ax, ylabel=r'$\theta$ [deg]',
                                            xlabel='$z$ [m]', title=f'Surface Pressure $p_{{{m}}}$ (dB)', levels=levels)
        
        if show_discretization:
            ax.scatter(PHI, np.rad2deg(TH), color='k', marker='x',alpha=0.25)
        
        print(f'maximum surface SPL: {np.max(p_to_SPL(pk))} dB')
        return fig, ax, mappable

    def plotBladeLoadingPerUnitArea(self, m, fig=None, ax=None, chord_stations=None, dc=None):
        """
        Plot the loading at harmonic k=m along the radial and chord
        directions for each component of loading as contour plots.

        Parameters
        ----------
        m : int
            Harmonic index/value to plot.
        fig : matplotlib.figure.Figure, optional
            Existing figure handle.
        ax : array-like of matplotlib.axes.Axes, optional
            Existing axes handle(s). Must contain 3 axes.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : ndarray of matplotlib.axes.Axes
        """

        import numpy as np
        import matplotlib.pyplot as plt

        # loading_harmonics_per_unit_area shape:
        # (3, Nk, Nr, Nc)
        (
            loading_harmonics_per_unit_area,
            loading_harmonics_per_unit_span,
            k,
            radius,
            chord_stations,
        ) = self.getBladeLoadingHarmonicsAmiet(chord_stations=chord_stations, dc=dc)

        # Find harmonic index
        ik = np.argmin(np.abs(k - m))

        # Create figure/axes if needed
        if fig is None or ax is None:
            fig, ax = plt.subplots(
                1, 3,
                figsize=(15, 4),
                constrained_layout=True
            )

        ax = np.atleast_1d(ax)

        if len(ax) != 3:
            raise ValueError("ax must contain 3 axes.")

        # Radius mesh
        R = np.tile(radius[:, None], (1, len(chord_stations)))  # (Nr, Nc)

        # Chord positions varying with radius
        chord_positions = (
            self.seg_chord[:, None] / 2
            * chord_stations[None, :]
        )  # (Nr, Nc)

        component_names = ["r", "ax", "\phi"]

        from matplotlib.colors import LogNorm

        # ------------------------------------------------------------------
        # Compute common logarithmic color scale across all 3 components
        # ------------------------------------------------------------------

        all_data = np.abs(loading_harmonics_per_unit_area[:, ik, :, :])

        # Avoid zeros for logarithmic scaling
        positive_data = all_data[all_data > 0]

        if positive_data.size == 0:
            vmin, vmax = 1e-12, 1.0
        else:
            vmin = np.min(positive_data)
            vmax = np.max(positive_data)

        # Optional dynamic range limiting
        vmax = min(vmax, 2 * np.mean(positive_data))

        # Ensure valid bounds
        vmin = max(vmin, vmax * 1e-6)

        norm = LogNorm(vmin=vmin, vmax=vmax)

        # Log-spaced contour levels
        levels = np.logspace(
            np.log10(vmin),
            np.log10(vmax),
            51
        )

        # ------------------------------------------------------------------
        # Plot
        # ------------------------------------------------------------------

        for i in range(3):

            # Select component and harmonic
            data = np.abs(
                loading_harmonics_per_unit_area[i, ik, :, :]
            )  # (Nr, Nc)

            # Avoid zeros in plotted field
            data = np.maximum(data, vmin)

            cf = ax[i].contourf(
                R,
                chord_positions,
                data,
                levels=levels,
                norm=norm,
                cmap="viridis",
            )

            ax[i].set_title(f"$F_{{{component_names[i]}}}^{{k={k[ik]}}}$")
            ax[i].set_xlabel("Radius")
            ax[i].set_ylabel("Chord")

            # Keep equal scaling
            ax[i].set_aspect(1)

            fig.colorbar(cf, ax=ax[i])

        return fig, ax

