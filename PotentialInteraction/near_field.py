from .main import HansonModel
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .placeholder import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.special import hankel2, jve, jv

def distance_in_polar(x, y):
    """
    x: array of shape (3, Nx1, Nx2, ...)
    y: array of shape (3, Ny1, Ny2, ...)
    
    returns:
        distances of shape (Nx1, Nx2, ..., Ny1, Ny2, ...)
    """
    R1, theta1, phi1 = x
    R2, theta2, phi2 = y

    # number of spatial dimensions (excluding the coordinate axis)
    nx = R1.ndim
    ny = R2.ndim

    # reshape for broadcasting
    R1 = R1.reshape(R1.shape + (1,) * ny)
    theta1 = theta1.reshape(theta1.shape + (1,) * ny)
    phi1 = phi1.reshape(phi1.shape + (1,) * ny)

    R2 = R2.reshape((1,) * nx + R2.shape)
    theta2 = theta2.reshape((1,) * nx + theta2.shape)
    phi2 = phi2.reshape((1,) * nx + phi2.shape)

    distance = np.sqrt(
        R1**2 + R2**2
        - 2 * R1 * R2 * (
            np.cos(theta1) * np.cos(theta2)
            + np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2)
        )
    )

    return distance 


class NearFieldHansonModel(HansonModel):

    def __init__(self, N_points_per_period=20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Nt = self.kmax * N_points_per_period
        self.time = np.linspace(0, 2 * np.pi / self.Omega, self.Nt, endpoint=False)
        self.dt = self.time[1] - self.time[0]


        # Broadcast seg_radius to shape (Nt, Nr)
        seg_radius_broadcasted = np.broadcast_to(self.seg_radius[None, :], (self.Nt, self.Nr))

        # Broadcast pi/2 array (already (Nt, Nr))
        theta_array = np.ones((self.Nt, self.Nr)) * (np.pi/2)

        # Broadcast Omega*time to shape (Nt, Nr)
        phi_array = np.broadcast_to(self.Omega * self.time[:, None], (self.Nt, self.Nr))

        # Stack along new first axis to get shape (3, Nt, Nr)
        self.seg_positions = np.stack([seg_radius_broadcasted, theta_array, phi_array], axis=0)
        # position vector in POLAR at self.time

    def getHansonPressure(self, x, m, B, Omega, loading, nb, multiplier):
        """
        Generic near-field formulation of the Hanson model for rotor noise

        multiplier is an overall multiplier for total the pressure mode. For B blades it should be B, for one stator/beam it should be 1.
        """
        assert np.all(m != 0), "m=0 (steady loading) is not supported"

        c0 = self.c # SoS
        R, theta, phi = x[0], x[1], x[2]

        Rprime = distance_in_polar(x, self.seg_positions) # distance vector between observation and source points, size (Nx, Nt, Nr)

        # matrix of size (Nx, Nm, Nt, Nk, Nr)

        mB = m * B # Nm
        wavenumber = mB * Omega / c0 # Nm, issue if mb = 0?

        k = self.k 
        Fblade = loading

        k = np.concatenate((-k[-1:0:-1], k)) # add the minus part!, shape (2Nk-1 -> Nk)
        Fblade = np.concatenate((np.conjugate(Fblade[:, -1:0:-1]), Fblade), axis=1) # minus loadings are conjugates of positive!


        # p_mB (Nx, Nm)


        # --- explicit broadcasting (with time) ---      
        theta = theta[:, None, None, None, None]       # (Nx, 1, 1, 1, 1)
        phi   = phi[:, None, None, None, None]         # (Nx, 1, 1, 1)

        mB    = mB[None, :, None, None, None]           # (1, Nm, 1, 1, 1)
        k     = k[None, None, None, :, None]            # (1, 1, 1, Nk, 1)
        mB_minus_k = mB - k # (1, Nm, 1, Nk, 1)
 
        radius = self.seg_radius[None, None, None, None, :]      # (1, 1, 1, 1, Nr)
        dr     = self.dr[None, None, None, None, :]          # (1, 1, 1, 1, Nr)
        time = self.time[None, None, :, None, None]
        
        # dt = np.array([self.dt])[None, None, None, None, None]
        dt = self.dt #scalar!
        wavenumber = wavenumber[None, :, None, None, None]  # (1, Nm, 1, 1, 1)

        Rprime = Rprime[:, None, :, None, :] # (Nx, 1, Nt, 1, Nr)


        Fphi = Fblade[2][None, None, None, :, :]             # (1, 1, 1, Nk, Nr)
        Fz = Fblade[1][None, None, None, :, :]             # (1, 1, 1, Nk, Nr)

        G1 = np.exp(1j * wavenumber * Rprime) / (Rprime ** 2) * (1 - 1/ 1j / wavenumber / Rprime) # (Nx, Nm, Nt, 1, Nr)
        G2 = G1 * np.sin(self.Omega * time - phi) # (Nx, Nm, Nt, 1, Nr)

        # # full structure to be reduced, of size (Nx, Nm, Nt, Nk, Nr)
        # matrix = Fz * np.cos(theta) * G1 * np.exp(1j * mB_minus_k * self.Omega * time) * dt * self.Omega / np.pi / 2 * dr
        # matrix += Fphi * np.sin(theta) * G2 * np.exp(1j * mB_minus_k * self.Omega * time) * dt * self.Omega / np.pi / 2 * dr
        # matrix *= 1j * wavenumber / 4 / np.pi * Rprime 

        # REWRITE: reduce instantaneous memory requirement!
        # NOTE: this is the expensive step!
        # G_mb__k_1 = np.sum(G1 * np.exp(1j * mB_minus_k * self.Omega * time) * dt * self.Omega / np.pi / 2, axis=2) # (Nx, Nm, Nt, Nk, Nr) -> (Nx, Nm, 1, Nr)
        # G_mb__k_2 = np.sum(G2 * np.exp(1j * mB_minus_k * self.Omega * time) * dt * self.Omega / np.pi / 2, axis=2) # (Nx, Nm, Nt, Nk, Nr) -> (Nx, Nm, 1, Nr)

        G_mb__k_1, G_mb__k_2 = self.get_G1_G2_lowmemory(dt, time, mB_minus_k, G1, G2)

        matrix = Fz * np.cos(theta) * G_mb__k_1[:, :, None, :, :] * dr # (Nx, Nm, 1, Nk, Nr)
        matrix += Fphi * np.sin(theta) * G_mb__k_2[:, :, None, :, :] * dr
        matrix *= 1j * wavenumber / 4 / np.pi * R[:, None, None, None, None] # check if correct!
        
        pmb = np.sum(matrix, axis=(-3, -2, -1)) # reduce along Nt, Nk, Nr axes by summing/integrating
        pmb *= multiplier # should be times the number of blades #TODO: check that this actually works, i.e. results in the same pressure as summing 2-3 blades separately!

        return pmb, x
    

    def get_G1_G2_lowmemory(self, dt, time, mB_minus_k, G1, G2):
        prefac = dt * self.Omega / (2 * np.pi)

        Nx, Nm, Nt, _, Nr = G1.shape
        _, _, _, Nk, _ = mB_minus_k.shape
        out_shape = (Nx, Nm, Nk, Nr)

        G_mb__k_1 = np.zeros(out_shape, dtype=complex)
        G_mb__k_2 = np.zeros(out_shape, dtype=complex)

        print('running time integration loop:')
        for it, t in enumerate(time[0, 0, :, 0, 0]):
            print(f'timestep {it+1} of {Nt}')
            # TIME AXIS IS AXIS=1 IN mB_minus_k
            phase = np.exp(
                    -1j * self.Omega * t * mB_minus_k
                )  # (Nx, Nm, 1, Nk, Nr)
            

            G_mb__k_1 += G1[:, :, it, :, :] * phase[:, :, 0, :, :]
            G_mb__k_2 += G2[:, :, it, :, :] * phase[:, :, 0, :, :]

        G_mb__k_1 *= prefac
        G_mb__k_2 *= prefac



        return G_mb__k_1, G_mb__k_2
    