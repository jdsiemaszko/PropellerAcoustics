from .PIN import PotentialInteraction
import numpy as np
from scipy.optimize import root, least_squares, fsolve, minimize
import matplotlib.pyplot as plt
from Constants.helpers import theodorsen, twoside_spectrum, ifft_periodic, fft_periodic, periodic_sum_interpolated

class ConformalPIN(PotentialInteraction):
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
                numerics = {},):
        
        # run the __init__ of the superclass
        super().__init__(twist_rad, 
                            chord_m, 
                            radius_m,
                            Fzprime_Npm,
                            Fphiprime_Npm,
                            B,
                            Dcylinder_m, Lcylinder_m, Omega_rads,
                            rho_kgm3=rho_kgm3, c_mps = c_mps, kmax = kmax, nb = nb,
                            U0_mps=U0_mps, # optional inflow velocity of shape (2, Nr), overwrites momentum theory computations
                            numerics = numerics,
                            )

        self.Rd = Dcylinder_m/2 # cylinder radius in the comp domain!
        self.zs, self.zeta_s = self.getSurfacePoints()
        
    def getZ(self, zeta):
        """
        transform from the computational domain to the physical domain, this should be overwritten in subclasses
        """
        
        return zeta
    
    def getDzetaDz(self, zeta):
        """
        get the IMPLICIT derivative dzeta/dz w.r.t. the computational coordinate zeta,
        this should be overwritten in subclasses
        """

        return np.ones_like(zeta)
    
    def getSurfacePoints(self):
        # xt = (self.Rd - self.Rr) * np.cos(self.theta_beam + self.theta0) + self.rho_corner * np.cos((self.Rd-self.Rr)/self.Rr * self.theta_beam - self.theta0)
        # yt = (self.Rd - self.Rr) * np.sin(self.theta_beam + self.theta0) - self.rho_corner * np.sin((self.Rd-self.Rr)/self.Rr * self.theta_beam - self.theta0)

        # return xt + 1j * yt

        # assumption about surface location: should be overwritten if different!
        zeta_s = self.Rd * np.exp(1j * self.theta_beam)
        z_s = self.getZ(zeta_s) 
        return z_s, zeta_s
    
    def getZeta(self, z):
        """
        inverse transform from physical to computational. 
        this should be overwritten in subclasses
        """
        return z
    

    def getStrutPressure(self):
        """"
        surface pressure accounting for conformal mapping
        mind the derivatives!

        returns pressure distribution over the strut, as a function of r, phi, and theta (beam polar angle w.r.t rotor plane)
        """

        gamma = self.getGammaInPhi() # shape (Nphi, Nr) - quasi-steady-unsteady vortex strength

        thetab = self.theta_beam
        deltathetab = np.diff(thetab)[0]

        zetas = self.zeta_s
        zetasprime = self.Rd**2 / zetas # circle conjugate

        # inflow part
        Uimag = np.linalg.norm(self.Ui, axis=0) # Nr
        alpha0 = np.arctan2(self.Ui[0], -self.Ui[1]) # Nr

        alpha0_zeta = alpha0 - self.theta0 # ROTATE the inflow in the computational plane!

        # TODO: check for errors
        vortex_period = 2 * np.pi / self.B / self.Omega # vortex passage period
        pressure = np.zeros((thetab.shape[0], self.phi.shape[0], self.seg_radius.shape[0]), dtype=np.complex128) # Nthetab, Nphi, Nr
        dfdzeta = np.zeros((thetab.shape[0], self.phi.shape[0], self.seg_radius.shape[0]), dtype=np.complex128) # Nthetab, Nphi, Nr
        
        # add the mean flow term
        dfdzeta += 1j * Uimag[None, None, :] * (np.exp(-1j * alpha0_zeta[None, None, :]) + np.exp(1j * alpha0_zeta[None, None, :]
                    ) * zetasprime[:, None, None] / zetas[:, None, None]) 
        
        for vortex_index in range(-12, 12, 1): # sum an arbitrary amount of vortices, further ones should be negligible
            # vortex position, complex, size (Nphi, Nr), vortex is moving from negative x to positive with speed Omega * r
            # phased vortices: shift the passage time by vortex_index * T/B
            zv = self.seg_radius[None, :] * (self.phi[:, None] + vortex_index * vortex_period * self.Omega) + 1j * self.Lcylinder 

            zetav = self.getZeta(zv) # get vortex path in COMPUTATIONAL DOMAIN

            zetavbar = np.conjugate(zetav) # complex conjugate

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
            
            dfdzeta_vortex = -1j * gamma_shifted[None, :, :] / 2 / np.pi / (zetas[:, None, None] -
                    zetav[None, :, :]) + 1j * gamma_shifted[None, :, :] / 2 / np.pi / (zetasprime[:, None, None] - 
                    zetavbar[None, :, :]) * (-zetasprime[:, None, None] / zetas[:, None, None]) # Nthetab, Nr, Nphi
            dfdzeta += dfdzeta_vortex

            dfdt_dzetavdzv= self.rho * gamma_shifted[None, :, :] * self.Omega * self.seg_radius[None, None, :] / 2 / np.pi * (1j / zetavbar[None, :, :] +
                     1j / (zetav[None, :, :] - zetas[:, None, None]) - 1j / (zetavbar[None, :, :] - zetasprime[:, None, None]))
             # (Nthetab, Nphi, Nr)
            

            # add the linear contribution to the pressure, including the dzetadz mapping at zetav.
            # note that at |zeta| -> infinity we have dzeta/dz = 1
            pressure += np.real(dfdt_dzetavdzv * self.getDzetaDz(zetav)[None, :, :]) # apply the real over the entire expression!
        
        dfdz = dfdzeta * self.getDzetaDz(zetas)[:, None, None] # apply the mapping

        u, v = np.real(dfdz), -np.imag(dfdz)
        U = np.sqrt(u**2 + v**2) # (Nthetab, Nphi, Nr)
        pressure_dynamic = 0.5 * self.rho * (Uimag**2 - U**2) # total!
        pressure += pressure_dynamic # add the nonlinear contribution

        return pressure # Nthetab, Nphi, Nr

    def getBladeDownwash(self):
        """
        get downwash at the blade station due to uniform inflow over the cylinder, in m/s, in the time domain
        """

        Uimag = np.linalg.norm(self.Ui, axis=0) # Nr
        alpha0 = np.arctan2(self.Ui[0], -self.Ui[1]) # Nr

        alpha0_zeta = alpha0 - self.theta0 # ROTATE the inflow in the computational plane!

        # check that:
        # Ui[0] = Uimag * np.sin(alpha0)
        # Ui[1] = -Uimag * np.cos(alpha0)

        Ds = self.Dcylinder
        Ls = self.Lcylinder
        Nphi = self._numerics.get('Nphi', 360)
        N = 10 # arbitrary, N>2 should be okay
        phi_long = np.linspace(-N * np.pi, N * np.pi, Nphi * N, endpoint=False)

        z = 1j * Ls + self.seg_radius[:, None] * phi_long[None, :] # Nr, Nphi

        zeta = self.getZeta(z) # map to computational domain

        # CIRCLE conjugate
        zetaprime = self.Rd**2 / zeta

        # COMPLEX conjugate
        # zbar = np.conj(z)

        # potential due to uniform inflow over a circle at an angle
        # f = 1j * Uimag * (z * np.exp(-1j * alpha0) - Ds**2 / 4 / z * np.exp(1j * alpha0))

        # derivative of potential: df/dz = u - 1j * v, explicit because why wouldn't we
        dfdzeta = 1j * Uimag[:, None] * (np.exp(-1j * alpha0_zeta[:, None]) + np.exp(1j * alpha0_zeta[:, None]) * zetaprime / zeta) # Nr, Nphi

        dfdz = dfdzeta * self.getDzetaDz(zeta)

        u, v = np.real(dfdz), -np.imag(dfdz)
        
        # downwash in the physical domain!
        w_vec = np.stack([
            u - Uimag[:, None] * np.sin(alpha0)[:, None], # u should approach Uimag * sin(alpha0) at infinity
            v + Uimag[:, None] * np.cos(alpha0)[:, None]  # v sgould approach -Uimag * sin(alpha0) at infinity
        ])

        # NEED A LONGER ARRAY   
        _, w_periodic = periodic_sum_interpolated(np.swapaxes(w_vec, 1, 2), period=2 * np.pi, time=phi_long, kind='cubic', t_new=self.phi)
        w_periodic = np.swapaxes(w_periodic, 1, 2)


        # alpha = self.seg_twist[:, None]
        alpha = np.arctan2(-self.Ui[1], self.Omega * self.seg_radius - self.Ui[0])[:, None] # Nr, None

        w_normal = w_periodic[0] * (-np.sin(alpha)) + w_periodic[1] * np.cos(alpha) # Nr, Nphi

        return w_normal
    
    def getStrutLoading(self):
        pressure  = self.getStrutPressure() # Nthetab, Nphi, Nr

        thetab = self.theta_beam
        deltathetab = np.diff(thetab)[0]

        # compute forces by integrating over thetab,
        # mapping the surface position!

        dzsdtheta = self.getDzetaDz(self.zeta_s)**(-1) * 1j * self.zeta_s # assuming zeta = Rd * e^(i*theta)

        Fphi = np.sum(
            pressure *
            np.imag(dzsdtheta)[:, None, None] * 
            deltathetab,
            axis=0
        ) # (Nphi, Nr), drag, positive backwards

        Fz = -np.sum(
            pressure *
            -np.real(dzsdtheta)[:, None, None] * 
            deltathetab,
            axis=0
        ) # (Nphi, Nr) # lift, positive upwards

        F_beam = np.zeros((3, self.phi.shape[0], self.seg_radius.shape[0]), dtype=np.complex128) # Note: in the time domain!, size (3, Nt, Nr)
        F_beam[1, :, :] = Fz
        F_beam[2, :, :] = Fphi
    
        return F_beam

    def plotZ(self, fig=None, ax=None):
        zs = self.zs

        zs = np.append(zs, zs[0]) # close the circle

        # Create figure/axes if not provided
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        # Plot the mapped points
        ax.plot(np.real(zs), np.imag(zs), label="Beam Surface", color='k')

        # # Plot circle of radius Rd
        # theta = self.theta_beam
        # circle = self.Rd * np.exp(1j * theta)
        # ax.plot(np.real(circle), np.imag(circle), linestyle='--', label="|z| = Rd", color='k')

        # Formatting
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("$Re(z_s)$")
        ax.set_ylabel("$Im(z_s)$")
        # ax.set_title("Physical Domain, N={}".format(self.Nsides))
        ax.legend()
        ax.grid(True)

        return fig, ax
    
    def plotZeta(self, fig=None, ax=None):
        zetas = self.zeta_s

        zetas = np.append(zetas, zetas[0]) # close the circle

        # Create figure/axes if not provided
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        # Plot the mapped points
        ax.plot(np.real(zetas), np.imag(zetas), label="Beam Surface", color='k')

        # Formatting
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("$Re(\zeta_s)$")
        ax.set_ylabel("$Im(\zeta_s)$")
        ax.set_title("Computational Domain, N={}".format(self.Nsides))
        ax.legend()
        ax.grid(True)

        return fig, ax

    def plotMap(self, fig=None, ax=None):
        radii = np.linspace(self.Rd, self.Rd * 3, 50)

        # Create figure/axes if not provided
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        for index, radius in enumerate(radii):
            zs = self.getZ(radius * np.exp(1j * self.theta_beam))
            zs = np.append(zs, zs[0]) # close the circle

            # Plot the mapped points
            if index == 0:
                ax.plot(np.real(zs), np.imag(zs), label="Beam Surface", color='blue', marker='x')
            else:
                ax.plot(np.real(zs), np.imag(zs), color='k', linestyle='dashed')

        # Formatting
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("$Re(z_s)$")
        ax.set_ylabel("$Im(z_s)$")
        # ax.set_title("Strut Cross Section, N={}".format(self.Nsides))
        ax.legend()
        ax.grid(True)

        return fig, ax
    
class HypotrochoidalPIN(ConformalPIN):
    def __init__(self, Nsides:int, theta0:float, rho_corner:float,
                twist_rad:np.ndarray, 
                chord_m:np.ndarray, 
                radius_m:np.ndarray,
                Fzprime_Npm:np.ndarray,
                Fphiprime_Npm:np.ndarray,
                B,
                Dcylinder_m, Lcylinder_m, Omega_rads,
                rho_kgm3=1.0, c_mps = 340, kmax = 20, nb:float = 1,
                U0_mps:np.ndarray=None, # optional inflow velocity of shape (2, Nr), overwrites momentum theory computations
                numerics = {},):
        
        self.Rd = Dcylinder_m/2
        self.Nsides = Nsides
        self.Rr = self.Rd / self.Nsides
        self.rho_corner = rho_corner * self.Rr # define relative to maximum allowed rho!
        self.theta0 = theta0
        # run the __init__ of the superclass ConformalPIN
        super().__init__(twist_rad, 
                            chord_m, 
                            radius_m,
                            Fzprime_Npm,
                            Fphiprime_Npm,
                            B,
                            Dcylinder_m, Lcylinder_m, Omega_rads,
                            rho_kgm3=rho_kgm3, c_mps = c_mps, kmax = kmax, nb = nb,
                            U0_mps=U0_mps, # optional inflow velocity of shape (2, Nr), overwrites momentum theory computations
                            numerics = numerics,
                            )
        
    def getZ(self, zeta):
        """
        transform from the computational domain (cylinder flow) to the physical domain (Nsides-gonal flow)
        """
        # z = self.Rd * ((1-1/self.Nsides) * zeta + self.rho_corner / self.Rd *
        #                 zeta **(-self.Nsides+1)) * np.exp(1j * self.theta0)

        z = zeta + self.rho_corner / (1-1/self.Nsides) * (zeta/self.Rd) ** (-self.Nsides + 1)
        z *= np.exp(1j * self.theta0)
        return z
    
    def getDzetaDz(self, zeta):
        """
        get the IMPLICIT derivative dzeta/dz w.r.t. the computational coordinate zeta
        """

        # dzetadz = (np.exp(1j * self.theta0) * self.Rd * ((1-1/self.Nsides)-self.rho_corner/self.Rd * (self.Nsides-1) * zeta**(-self.Nsides)))**(-1) 
        dzetadz = (np.exp(1j * self.theta0) * (1 - self.rho_corner * self.Nsides / self.Rd  * (zeta/self.Rd)**(-self.Nsides)))**(-1) 

        return dzetadz
    
    def getZeta(self, z, MAXITER=int(1e3), epstol=1e-8):
        """
        Vectorized implicit solve z = self.getZ(zeta)
        using polar parametrization: zeta = r * exp(i phi)
        with bounds on r and phi.
        """

        # zeta = 2.0 * self.Rd * np.exp(1j * np.angle(z)) # initial guess
        zeta = z
        for iter in range(MAXITER):
            dzeta = -(self.getZ(zeta) - z) * self.getDzetaDz(zeta) # Newton!

            zeta += dzeta
            error = np.abs(self.getZ(zeta) - z) / self.Rd

            if np.all(error) < epstol:
                break

        return zeta
    

# class JoukowskiPIN(HypotrochoidalPIN):

