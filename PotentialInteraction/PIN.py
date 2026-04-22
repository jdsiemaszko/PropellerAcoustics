import numpy as np
import numpy as np
from scipy.special import jv
from Constants.helpers import theodorsen, twoside_spectrum, ifft_periodic, fft_periodic, periodic_sum_interpolated
import matplotlib.pyplot as plt

class PotentialInteraction:
    def __init__(self,
                twist_rad:np.ndarray, 
                chord_m:np.ndarray, 
                radius_m:np.ndarray,
                Fzprime_Npm:np.ndarray,
                Fphiprime_Npm:np.ndarray,
                B,
                Dcylinder_m, Lcylinder_m, Omega_rads,
                rho_kgm3=1.0, c_mps = 340, kmax = 20, nb:float = 1,
                U0_mps:np.ndarray=None, # optional inflow velocity of shape (2, Nr), overwrites momentum theory computations
                numerics = {},
                ):
        self.B = B
        self.Dcylinder = Dcylinder_m
        self.Lcylinder = Lcylinder_m
        self.Omega=Omega_rads
        self.rho = rho_kgm3
        self.c = c_mps # speed of sound
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

            Uiz = -np.sqrt(self.Fzprime /  4 / np.pi / self.rho / self.seg_radius) # positive upwards
            Uiphi = -self.Fphiprime / 4 / np.pi / self.rho / self.seg_radius / Uiz # positive to the right, Note: dQ is the side force, not torque!
            self.Ui = np.stack([Uiphi, Uiz]) # shape (2, Nr)

        # pre-compute common arrays
        Nphi = self._numerics.get('Nphi', 360)
        self.phi = np.linspace(-np.pi, np.pi, Nphi, endpoint=False)
        Nthetab = self._numerics.get('Nthetab', 360)
        self.theta_beam = np.linspace(0, 2 * np.pi, Nthetab, endpoint=False) # angles on the cylinder surface, measured from the prop. plane, size (Npoints)

        self.Ur = np.sqrt(
            (self.Omega * self.seg_radius - self.Ui[0])**2 + self.Ui[1]**2
        ) # relative velocity over the blade (mean?), shape Nr

    def getStrutLoading(self):
        pressure  = self.getStrutPressure() # Nthetab, Nphi, Nr

        thetab = self.theta_beam
        deltathetab = np.diff(thetab)[0]

        Fphi = self.Dcylinder / 2 * np.sum(
            pressure *
            np.cos(thetab)[:, None, None] *
            deltathetab,
            axis=0
        ) # (Nphi, Nr), drag, oriented backwards


        Fz = -self.Dcylinder / 2 * np.sum(
            pressure *
            np.sin(thetab)[:, None, None] * 
            deltathetab,
            axis=0
        ) # (Nphi, Nr) # lift, oriented upwards

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

        # go to the frequency domain
        # points_per_period = self._numerics.get('points_per_period', 20)
        # k_local_max = np.ceil(self.kmax / self.B) # only resolve up to ceil(kmax/B) multiples of B*Omega
        # k_local = np.arange(1, k_local_max+1, 1)

        # T_periodic = np.linspace(-period/2, period/2, points_per_period * int(np.max(k_local)), endpoint=False) # Np
        #TODO: fix - this in incorrect as F_beam is not linear with number of vortices!
        # Need to apply the linear addition earlier, at the potential computation
        # T_periodic, F_beam = periodic_sum_interpolated(F_beam, period=period, time=self.phi / self.Omega, kind='cubic', t_new=T_periodic)

        # Np = T_periodic.shape[0] # should be equal to points_per_period * max(k_local)!
        # dt = np.diff(T_periodic)[0]

        # # shape (3, Nk, Nr)
        # F_beam_k = 1/period * np.sum(F_beam[:, None, :, :] * np.exp(+1j *
        #          k_local[None, :, None, None] * 2 * np.pi / period * 
        #          T_periodic[None, None, :, None]) * dt, axis=2)

        # corrected
        period_vortex = 2 * np.pi / self.B / self.Omega # period with which a blade passes over the beam: i.e. T/B !

        period_global = 2 * np.pi / self.Omega # rotation period, DIFFERENT FROM blade passage period!

        N_periods = int(period_global / period_vortex)

        # TODO: fix! some bs here
        phi_extended = np.linspace(self.phi[0], self.phi[0] + N_periods * np.pi * 2, self.phi.shape[0] * N_periods, endpoint=False) # extended phi array covering multiple periods, shape (Np_extended,)
        F_beam_extended = np.tile(F_beam, (1, N_periods, 1)) # extend the time series to multiple periods, shape (3, Nt*N_periods, Nr)
        dt = np.diff(phi_extended)[0] / self.Omega # timestep corresponding to phi array
        F_beam_k_global = 1/period_global/N_periods * np.sum(F_beam_extended[:, None, :, :] * np.exp(+1j *
                self.k[None, :, None, None] * 2 * np.pi / period_global * 
                 (phi_extended/self.Omega)[None, None, :, None]) * dt, axis=2)
        
        # correct the contributions k!=mB (should be small anyway)
        for i, k_val in enumerate(self.k):
            if k_val % self.B != 0:
                F_beam_k_global[:, i, :] = 0.0 # zero out contributions that are not multiples of B, should be inconsequential
        F_beam_k_global[:, 0, :] = 0.0 # zero-out mean loading on the beam, should be inconsequential

        # Note: indez k corresponds to frequency k*B*Omega => need to map onto global k array with multiples of k*Omega!
        # this is DIFFERENT from the propeller computation!

        # # fill the array with zeros where k!= multiple of nb
        # k_global = self.k
        # F_beam_k_global = np.zeros((3, len(k_global), self.Nr), dtype=np.complex128)

        # # find where self.k==k_global
        # # fill these entries with value of Fblade
        # # leave the rest at zero
        # # find where self.k == k_global and fill
        # for i, k_val in enumerate(k_local):
        #     idx = np.where(k_global == k_val*self.B)[0] # should be EXACTLY one entry!
        #     if idx.size > 0:
        #          F_beam_k_global[:, idx[0], :] =  F_beam_k[:, i, :]

        return F_beam_k_global
    
    def getGammaInPhi(self):
        """
        get QS-unsteady circulation as a function of phi
        """
        gamma_k = self.getGammaHarmonics() #Nk, Nr
        gamma_kk, kk = twoside_spectrum(gamma_k, self.k) # get two-sided spectrum
        gamma_phi, phi = ifft_periodic(gamma_kk, 2 * np.pi, self.phi, kk) # Nphi, Nr

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

    def getStrutPressure(self):
        """
        returns pressure distribution over the strut, as a function of r, phi, and theta (beam polar angle w.r.t rotor plane)
        """

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
        
        for vortex_index in range(-12, 12, 1): # sum an arbitrary amount of vortices, further ones should be negligible
            # vortex position, complex, size (Nphi, Nr), vortex is moving from negative x to positive with speed Omega * r
            # phased vortices: shift the passage time by vortex_index * T/B
            zv = self.seg_radius[None, :] * (self.phi[:, None] + vortex_index * vortex_period * self.Omega) + 1j * self.Lcylinder 
            zvbar = np.conjugate(zv) # complex conjugate

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
        
        u, v = np.real(dfdz), -np.imag(dfdz)
        U = np.sqrt(u**2 + v**2) # (Nthetab, Nphi, Nr)
        pressure_dynamic = 0.5 * self.rho * (Uimag**2 - U**2) # total!
        pressure += pressure_dynamic # add the nonlinear contribution

        return pressure # Nthetab, Nphi, Nr

    def getBladeLoadingHarmonics(self, QS=False):
        """
        returns loading ON the blade in the fourier domain
        result of shape (3, Nk, Nr)
        3: radial, axial, tangential
        k's correspond to frequencies k * Omega
        QS - if True, use quasi-steady assumption, else use full unsteady Sears function
        """

        w = self.getBladeDownwash() # Nr, Nphi
        k = self.k # Nk
        phi = self.phi # Nphi

        dphi = np.diff(self.phi)[0]
        wk = 1 / 2 / np.pi * np.sum(
            dphi * np.exp(1j * k[None, None, :] * self.phi[None, :, None])*
            w[:, :, None], axis=1 # reduce the phi axis, result of shape Nr, Nk
        )
        wk = wk.T # Nk, Nr for consistency

        Ur = self.Ur # Nr

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

        Fblade[1, :, :] = Lkprime * np.cos(self.seg_twist[None, :]) # positive upwards, but Lkprime is oriented downwards for positive wk!
        Fblade[2, :, :] = Lkprime * np.sin(self.seg_twist[None, :]) # DRAG, oriented BACKWARDS


        # steady loads. Note: phase shift
        Fblade[1, 0, :] = self.Fzprime # axial, positive upwards, NOTE: Fzprime is PER BLADE, so is Fphiprime
        Fblade[2, 0, :] = self.Fphiprime # tangential, positive backwards

        return Fblade

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

        # complex variable
        # Note: here we need the assumption r * pi >> Lcylinder!

        z = 1j * Ls + self.seg_radius[:, None] * self.phi[None, :] # Nr, Nphi

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
            u - Uimag[:, None] * np.sin(alpha0)[:, None], # u should approach Uimag * sin(alpha0) at infinity
            v + Uimag[:, None] * np.cos(alpha0)[:, None]  # v sgould approach -Uimag * sin(alpha0) at infinity
        ])


        alpha = self.seg_twist[:, None]
        w_normal = w_vec[0] * (-np.sin(alpha)) + w_vec[1] * np.cos(alpha) # Nr, Nphi


        return w_normal
    
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

