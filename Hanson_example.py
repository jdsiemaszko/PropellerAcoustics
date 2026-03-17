import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from Hanson.far_field import HansonModel

NSEG = 20 # number of radial prop segments
kmaxx = 64 # number of harmonics k*Omega covered
ROBS = 10. # observation radius, in meters

rad = np.linspace(0.016, 0.1, NSEG+1, endpoint=True) # radial stations (segment edges)
loadings = np.zeros((3, kmaxx, NSEG), dtype=np.complex128) # loadings of size (3, Nk, Nr) in order: radial, axial, tangential.
# sign convention: axial positive towards upstream, tangential positive opposite to the direction of rotation
loadings[1, 0, :] = (1.0 + 0j) * np.cos(np.deg2rad(10)) # example loading
loadings[2, 0, :] = (1.0 + 0j) * np.sin(np.deg2rad(10)) # example loading


# Initialize Module
hm = HansonModel(
                radius_m=rad, # blade radius stations [m] of size Nr + 1
                axis=np.array([0, 0, 1]), origin=np.array([0, 0, 0]), radial=np.array([1, 0, 0]), # coordinate system (not needed here)
                B=2, # number of blades
                Omega_rads=8000/60 * 2 * np.pi, # rotation speed [rad/s]
                rho_kgm3=1.2, # fluid density [kg/m^3]
                c_mps= 340., # speed of sound [m/s]
                nb = 0 # number of beams (irrelevant)
                )

# 1) Plot directivity (3D)
fig = plt.figure(figsize=(7, 7))
ax1 = fig.add_subplot(111, projection="3d")
hm.plot3Ddirectivity(
    fig=fig,
    ax=ax1,
    m=1, # harmonic to plot
    R=ROBS, # observation radius
    Nphi=36*2, # plotting params
    Ntheta=18*2,
    valmin=40,
    valmax=65,
    title='far-field',
    mode='rotor', # 'rotor' or 'stator'
    loadings=loadings # blade loading harmonics
)
plt.tight_layout()
plt.show()

# 2) Plot directivity (2D polar plot)
hm.plot2Ddirectivity(
    m=1, # harmonic to plot
    R=ROBS, # observation radius
    Ntheta=360,
    mode='rotor', # 'rotor' or 'stator' or 'total'
    loadings=loadings, # blade loading harmonics
    plane='xz'
)
plt.tight_layout()
plt.show()

# 3) Plot spectrum at a point
fig, ax = plt.subplots()
hm.plotPressureSpectrum(fig=fig, ax=ax, 
                        x =np.array([1.0, 0.0, 0.0]).T, # position to plot the spectrum at
                        m = np.arange(1, 10, 1), # modes to compute
                        loadings=loadings # blade loading harmonics
                            )
plt.tight_layout()
plt.show()
