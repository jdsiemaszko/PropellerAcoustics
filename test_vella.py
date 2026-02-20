import h5py
from scipy.special import hankel2, jve, jv
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PotentialInteraction.main import HansonModel
from PotentialInteraction.placeholder import *

NSEG = 20 # number of radial prop segments
kmaxx = 64
ROBS = 1.62


# rad = np.linspace(0.017, 0.1, NSEG, endpoint=True)
# dr = rad[1] - rad[0]
# rad = np.concatenate((
#     [rad[0] - dr/2],
#     rad+dr/2,
#     # [rad[-1] + dr/2]
# )) 

rad = np.linspace(0.017, 0.1, NSEG+1, endpoint=True)


# VELLA ET AL. 2026
HANSON_VELLA = HansonModel(twist_rad = np.deg2rad(10 * np.ones(NSEG+1)), chord_m = 0.025 * np.ones(NSEG+1),
                    
                    radius_m=rad
                                            
                                            ,
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

# r_stat = np.array([0.5, 0.8, 0.9])
# fig, ax = plt.subplots(nrows = 2, ncols = len(r_stat), sharey=True, sharex=True)
# HANSON_VELLA.plotLoadingHarmonics(fig, ax, r_stat,
#                              mode='beam'
#                             #  mode='blade',
#                             #  LIFT=True
#                              )
# plt.show()

fig, ax = plt.subplots()
HANSON_VELLA.plotPressureSpectrum(fig, ax, np.array([ROBS, np.pi/2, np.pi/2]).reshape(3, 1), np.arange(1, 16, 1))
ax.plot(FPLUS, SPL_BLADE_01, label='blade',  color='r', linestyle='dashed') #=> blade has incorrect magnitudes of modes, a "fatter tail"!
ax.plot(FPLUS, SPL_BEAM_01, label='beam',  color='b', linestyle='dashed')
ax.set_ylim(0, 70)
plt.show()
plt.close()

fig, ax = plt.subplots()
HANSON_VELLA.plotPressureSpectrum(fig, ax, np.array([ROBS, np.pi/2, np.pi]).reshape(3, 1), np.arange(1, 16, 1))
ax.plot(FPLUS, SPL_BLADE_02, label='blade',  color='r', linestyle='dashed')
ax.plot(FPLUS, SPL_BEAM_02, label='beam',  color='b', linestyle='dashed')
ax.set_ylim(0, 70)
plt.show()
plt.close()

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