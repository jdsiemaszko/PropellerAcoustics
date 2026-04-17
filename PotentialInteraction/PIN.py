import numpy as np
import numpy as np
from scipy.special import jv
from Constants.helpers import periodic_sum, plot_directivity_contour, p_to_SPL, periodic_sum_interpolated, fft_periodic, ifft_periodic, continuous_log, fft_periodic, ifft_periodic, twoside_spectrum
import matplotlib.pyplot as plt

class PotentialInteraction:
    def __init__(self,
                twist_rad:np.ndarray, 
                chord_m:np.ndarray, 
                radius_m:np.ndarray,
                Tprime_Npm:np.ndarray,
                Qprime_Npm:np.ndarray,
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
    
        self.Tprime = Tprime_Npm # Nr
        self.Qprime = Qprime_Npm # Nr

        if U0_mps is not None:
            self.Ui = U0_mps # (2, Nr) x positive to the right, y positive upwards
        else:

            Uiz = -np.sqrt(self.Tprime /  4 / np.pi / self.rho / self.seg_radius) # positive upwards
            Uiphi = -self.Qprime / 4 / np.pi / self.rho / self.seg_radius / Uiz # positive to the right, Note: dQ is the side force, not torque!
            self.Ui = np.stack([Uiphi, Uiz]) # shape (2, Nr)

        # pre-compute common arrays
        Nphi = 100
        self.phi = np.linspace(-np.pi, np.pi, Nphi)
        

    def getStrutLoading(self):
        """"
        strut loading in N/m along the strut radial stations
        returns: [Fr positive outwards, Fphi positive opposite to blade rotation, Fz positive upsteam]
        """
        pass
    

    def getBladeLoading(self):
        """
        loading on the blades in N/m along the radial direction

        """
        pass


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