import h5py
from scipy.special import hankel2, jve, jv
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PotentialInteraction.beam_to_blade import BladeLoadings
from PotentialInteraction.blade_to_beam import BeamLoadings
from PotentialInteraction.placeholder import *
from Hanson.far_field import HansonModel
from Hanson.near_field import NearFieldHansonModel

import numpy as np

# ------------------------------------------------------------------
# Define shared single instances (arrays & scalars)
# ------------------------------------------------------------------

NSEG = 20
radius_m = np.linspace(0.016, 0.1, NSEG+1)
radius_c = (radius_m[1:] + radius_m[:-1]) / 2
twist_rad = np.ones(NSEG+1) * np.deg2rad(10)
chord_m = np.ones(NSEG+1) * 0.025
Uz0_mps = -np.interp(radius_c / 0.1, R_RT_UDW, UDW_EXACT) # interpolated from data
Tprime_Npm = np.interp(radius_c / 0.1, R_RT_EXACT, DT_EXACT)
Qprime_Npm = np.interp(radius_c / 0.1, R_RT_EXACT, DQ_EXACT)

B = 2
Dcylinder_m = 0.02
Lcylinder_m = 0.02
Omega_rads = 8000 / 60 * 2 * np.pi
rho_kgm3 = 1.2
c_mps = 340
kmax = 40
nb = 1

NPHI = 36
NTHETA = 18

axis = np.array([0.0, 0.0, 1.0])
origin = np.array([0.0, 0.0, 0.0])

# ------------------------------------------------------------------
# Class initializations using shared instances
# ------------------------------------------------------------------

blade_l = BladeLoadings(
    twist_rad=twist_rad,
    chord_m=chord_m,
    radius_m=radius_m,
    Uz0_mps=Uz0_mps,
    Tprime_Npm=Tprime_Npm,
    Qprime_Npm=Qprime_Npm,
    B=B,
    Dcylinder_m=Dcylinder_m,
    Lcylinder_m=Lcylinder_m,
    Omega_rads=Omega_rads,
    rho_kgm3=rho_kgm3,
    c_mps=c_mps,
    kmax=kmax,
    nb=nb
)

beam_l = BeamLoadings(
    twist_rad=twist_rad,
    chord_m=chord_m,
    radius_m=radius_m,
    Uz0_mps=Uz0_mps,
    Tprime_Npm=Tprime_Npm,
    Qprime_Npm=Qprime_Npm,
    B=B,
    Dcylinder_m=Dcylinder_m,
    Lcylinder_m=Lcylinder_m,
    Omega_rads=Omega_rads,
    rho_kgm3=rho_kgm3,
    c_mps=c_mps,
    kmax=kmax,
    nb=nb
)

han = HansonModel(
    axis=axis,
    origin=origin,
    twist_rad=twist_rad,
    chord_m=chord_m,
    radius_m=radius_m,
    B=B,
    Omega_rads=Omega_rads,
    rho_kgm3=rho_kgm3,
    c_mps=c_mps,
    nb=nb
)

# han_nf = NearFieldHansonModel(
#     axis=axis,
#     origin=origin,
#     twist_rad=twist_rad,
#     chord_m=chord_m,
#     radius_m=radius_m,
#     B=B,
#     Omega_rads=Omega_rads,
#     rho_kgm3=rho_kgm3,
#     c_mps=c_mps,
#     nb=nb
# )

BLH = blade_l.getBladeLoadingMagnitude() # shape Nk, Nr
BeamLH = beam_l.getBeamLoadingHarmonics() # shape 3, Nk, Nr


han.plot3DdirectivityRotor(m=5, loadings=BLH, R=1.62, Nphi=NPHI, Ntheta=NTHETA, valmax=65, valmin=40)
han.plot3DdirectivityStator(m=5, loadings=BeamLH, R=1.62, Nphi=NPHI, Ntheta=NTHETA, valmax=65, valmin=40)
han.plot3DdirectivityTotal(m=1, loadings=BLH, loadings_2=BeamLH, R=1.62, Nphi=NPHI, Ntheta=NTHETA, valmax=65, valmin=40)
han.plot3DdirectivityTotal(m=4, loadings=BLH, loadings_2=BeamLH, R=1.62, Nphi=NPHI, Ntheta=NTHETA, valmax=65, valmin=40)
han.plot3DdirectivityTotal(m=5, loadings=BLH, loadings_2=BeamLH, R=1.62, Nphi=NPHI, Ntheta=NTHETA, valmax=65, valmin=40)




