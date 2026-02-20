from scipy.special import hankel2, jve, jv
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from placeholder import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def getHansonPressure(self, x, m, B, Omega, loading, nb, mutliplier):
    """
    Generic function for computing the hanson formulation of noise for rotors
    loading should be: radial, axial, tangential
    x should be: radius, polar, azimuth
    """
    assert np.all(m != 0), "m=0 (steady loading) is not supported"

    c0 = self.c

    # todo: make the transformation explicit
    R, theta, phi = x[0], x[1], x[2] # all of size Nx

    radius, twist, chord = self.seg_radius, self.seg_twist, self.seg_chord # all of size # Nr
    dr = self.dr # Nr, size of segment

    mB = m * B # Nm

    wavenumber = mB * Omega / c0 # Nm, issue if mb = 0?

    k = self.k * nb # Nk mutliplied by the number of beams!

    # Fblade = self.getLoadingHarmonics() # (3, Nr, Nk)
    Fblade = loading
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

    Fphi = Fblade[2][None, None, :, :]             # (1, 1, Nk, Nr)
    Fz = Fblade[1][None, None, :, :]             # (1, 1, Nk, Nr)

    # --- matrix construction ---
    matrix = (
        Fphi * (mB_m - k_k) / radius_r / (wavenumber_m+1e-12) # issue for m=0 or Omega=0
        + np.cos(theta_x) * Fz
    )

    # --- apply condition: wavenumber=0 and m!=k ---
    # Create a mask that broadcasts correctly to (Nx, Nm, Nk, Nr)
    # mask = (wavenumber_m == 0) & (mB_m != k_k)  # shapes: (1, Nm, 1, 1) & (1, Nm, Nk, 1) -> broadcast to (Nx, Nm, Nk, Nr)
    # matrix = np.where(mask, 0, matrix)

    matrix *= jv(mB_m - k_k, mB_m * Omega * radius_r / c0 * np.sin(theta_x))

    matrix = matrix.astype(np.complex128)

    matrix *= np.exp(
        1j * (mB_m - k_k) * (phi_x - np.pi / 2)
        + 1j * mB_m * Omega * R_x / c0
    )
    # matrix shape: (Nx, Nm, Nk, Nr)

    # reduce by summing along Nk and Nr axes
    pmb = np.sum (
        matrix * dr_r,
        axis=-1
    ) # integrate along the r axis
    pmb = np.sum(pmb, axis=-1) # sum along the k axis
    
    
    # np.sum(matrix, axis=(-1, -2))  # -> (Nx, Nm)

    # scaling
    pmb *= -1j * wavenumber * mutliplier / (4 * np.pi * R[:, None])

    return pmb, x



def getHansonPressure_POINTWISE(x, m, radius, dr, B, Omega, loading, nb, karr, mutliplier, c0=340):
    """
    Generic function for computing the hanson formulation of noise for rotors
    loading should be: radial, axial, tangential
    x should be: radius, polar, azimuth
    """

    # todo: make the transformation explicit
    R, theta, phi = x[0], x[1], x[2] # all of size Nx

    mB = m * B # Nm

    wavenumber = mB * Omega / c0 # Nm, issue if mb = 0?

    k = karr * nb # Nk mutliplied by the number of beams!

    Fblade = loading

    # --- explicit broadcasting ---
    R_x      = R[None, None]        
    theta_x = theta[None, None]       
    phi_x   = phi[None, None]     

    mB_m    = mB[None, None]         
    k_k     = k[:, None]          

    radius_r = radius[None, :]  
    dr_r     = dr[None, :]         

    wavenumber_m = wavenumber[None, None] 

    Fphi = Fblade[2][:, :]          
    Fz = Fblade[1][:, :]         

    # --- matrix construction ---
    matrix = (
        Fphi * (mB_m - k_k) / radius_r / (wavenumber_m+1e-12) # issue for m=0 or Omega=0
        + np.cos(theta_x) * Fz
    )

    # --- apply condition: wavenumber=0 and m!=k ---
    # Create a mask that broadcasts correctly to (Nx, Nm, Nk, Nr)
    # mask = (wavenumber_m == 0) & (mB_m != k_k)  # shapes: (1, Nm, 1, 1) & (1, Nm, Nk, 1) -> broadcast to (Nx, Nm, Nk, Nr)
    # matrix = np.where(mask, 0, matrix)

    matrix *= jv(mB_m - k_k, mB_m * Omega * radius_r / c0 * np.sin(theta_x))

    matrix = matrix.astype(np.complex128)

    matrix *= np.exp(
        1j * (mB_m - k_k) * (phi_x - np.pi / 2)
        + 1j * mB_m * Omega * R_x / c0
    )
    # matrix shape: (Nx, Nm, Nk, Nr)

    # reduce by summing along Nk and Nr axes
    pmb = np.sum (
        matrix * dr_r,
        axis=-1
    ) # integrate along the r axis
    pmb = np.sum(pmb, axis=-1) # sum along the k axis
    
    
    # np.sum(matrix, axis=(-1, -2))  # -> (Nx, Nm)

    # scaling
    pmb *= -1j * wavenumber * mutliplier / (4 * np.pi * R[:, None])

    return pmb, x
