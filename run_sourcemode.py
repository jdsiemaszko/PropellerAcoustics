from TailoredGreen.CylinderGreen import CylinderGreen
from TailoredGreen.TailoredGreen import TailoredGreen
from SourceMode.SourceMode import SourceMode
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from Hanson.far_field import HansonModel

# FREE-FIELD GREEN'S FUNCTION
gff = TailoredGreen() # free-field

# CYLINDER GREEN'S FUNCTION
caxis = np.array([1.0, 0.0, 0.0])
D_prop = 0.2
D = 20 / 1000
L = 20 / 1000
corigin = np.array([0.0, 0.0, -L])

cg = CylinderGreen(radius=D/2, axis=caxis, origin=corigin, dim=3, 
                           numerics={
                        'mmax': 16, # mind that increasing this increases the chance of overflows, ***should*** be handled by the safe Bessel functions, but beware
                        'Nq_prop': 32,
                        'Nq_evan': 32,
                        'eps_k' : 1e-24,
                    }) # cylinder

#SOURCE MODE
TWIST = np.deg2rad(00)
loadings = np.array([0.0, 1.0 * np.cos(TWIST), 1.0 * np.sin(TWIST)], dtype=np.complex_).reshape(3, 1, 1)
loading_magnitudes = np.linalg.norm(loadings) * 1.0
source = SourceMode(BLH=loading_magnitudes.reshape(1), B=2, gamma=TWIST,
                     axis=np.array([0., 0.0, 1.]), origin=np.array([0, 0.0,0.0]), radius=D_prop/2,
                    #    green=gff  # green's functions are interchangable
                      green = cg
                       )
OMEGA = 8000/60 * 2 * np.pi
ROBS = 1.62
hm = HansonModel(
                radius_m=np.array([0.5, 1.5]), # blade radius stations [m] of size Nr + 1
                axis=np.array([0, 0, 1]), origin=np.array([0, 0, 0]), radial=np.array([1, 0, 0]), # coordinate system (not needed here)
                B=2, # number of blades
                Omega_rads=OMEGA, # rotation speed [rad/s]
                rho_kgm3=1.2, # fluid density [kg/m^3]
                c_mps= 340., # speed of sound [m/s]
                nb = 2 # number of beams (irrelevant)
                )

# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(111, projection="3d")
# source.plotSelf(fig, ax)
# # ax.set_box_aspect([1, 1, 1])
# ax.set_aspect('equal')
# plt.show()
# plt.close()

fig = plt.figure(figsize=(7, 7))
ax1 = fig.add_subplot(111, projection="3d")
hm.plot3Ddirectivity(
    fig=fig,
    ax=ax1,
    m=1, # harmonic to plot
    R=ROBS, # observation radius
    Nphi=36, # plotting params
    Ntheta=18,
    title='far-field',
    valmax=80,
    valmin=0,
    mode='rotor', # 'rotor' or 'stator'
    loadings=loadings # blade loading harmonics
)
plt.tight_layout()
plt.show()

source.plotFarFieldPressure(m=np.array([1]), Omega = OMEGA, Nphi=36, Ntheta=18, R=ROBS,
                                valmax=80,
    valmin=0, 
                            # valmin=10, valmax=65 
                            mode ='direct'
                            )
plt.show()
plt.close()

# source.plotFarFieldPressure(m=np.array([1]), Omega = OMEGA, Nphi=36, Ntheta=18, R=ROBS, 
#                                 valmax=80,
#     valmin=0,
#                             # valmin=10, valmax=65 
#                             mode='scattered'
#                             )
# plt.show()
# plt.close()

# # source.plotGeometry()
# source.plotFarFieldPressure(m=np.array([1]), Omega = OMEGA, Nphi=36, Ntheta=18, R=ROBS, 
#                                 valmax=80,
#     valmin=0,
#                             # valmin=10, valmax=65 
#                             )
# plt.show()
# plt.close()



