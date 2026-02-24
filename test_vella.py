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

blade_l = BladeLoadings(
    twist_rad=np.array([0.0, 0.0]), # Nr+1
    chord_m=np.array([0.025, 0.025]), # Nr+1
    radius_m=np.array([0.75, 1.25]), # Nr+1
    Uz0_mps=np.array([10.0]), # Nr
    Tprime_Npm=np.array([100.0]), # Nr
    Qprime_Npm=np.array([50.0]), # Nr
    B=2,
    Dcylinder_m=0.02,
    Lcylinder_m=0.02,
    Omega_rads=8000/60 * 2 * np.pi,
    rho_kgm3=1.225,
    c_ms=340,
    kmax=20,
    nb=1
)

beam_l = BeamLoadings(
    twist_rad=np.array([0.0, 0.0]), # Nr+1
    chord_m=np.array([0.025, 0.025]), # Nr+1
    radius_m=np.array([0.75, 1.25]), # Nr+1
    Uz0_mps=np.array([10.0]), # Nr
    Tprime_Npm=np.array([100.0]), # Nr
    Qprime_Npm=np.array([50.0]), # Nr
    B=2,
    Dcylinder_m=0.02,
    Lcylinder_m=0.02,
    Omega_rads=8000/60 * 2 * np.pi,
    rho_kgm3=1.225,
    c_ms=340,
    kmax=20,
    nb=1
)

han = HansonModel(
    
)

