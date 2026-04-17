import numpy as np
import numpy as np
from scipy.special import jv
from Constants.helpers import theodorsen
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
        self.phi = np.linspace(-np.pi, np.pi, Nphi)
        

    def getStrutLoading(self):
        """"
        strut loading in N/m along the strut radial stations
        returns: [Fr positive outwards, Fphi positive opposite to blade rotation, Fz positive upsteam]
        """
        pass
    

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


        Ur = np.sqrt(
            (self.Omega * self.seg_radius - self.Ui[0])**2 + self.Ui[1]**2
        ) # relative velocity over the blade (mean?), shape Nr

        # --- Theodorsen arguments ---
        sigma = k[:, None] * self.Omega * self.seg_chord[None, :] / (2.0 * Ur[None, :])      # (Nk, Nr)
        beta = np.pi/2 - self.seg_twist # alpha in Wu et al. 2022, shape Nr

        mu = (
            1j * k[:, None]  * self.seg_chord[None, :] / (2.0 * self.seg_radius[None, :])
            * np.exp(-1j * beta[None, :])
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

        Fblade[1, :, :] = -Lkprime * np.cos(self.seg_twist[None, :]) # positive upwards, but Lkprime is oriented downwards for positive wk!
        Fblade[2, :, :] = -Lkprime * np.sin(self.seg_twist[None, :]) # DRAG, oriented BACKWARDS


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
        Nlevels = 50
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