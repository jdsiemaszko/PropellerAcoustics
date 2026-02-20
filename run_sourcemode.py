from TailoredGreen.CylinderGreen import CylinderGreen
from TailoredGreen.TailoredGreen import TailoredGreen
from SourceMode.SourceMode import SourceMode
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# FREE-FIELD GREEN'S FUNCTION
# gff = TailoredGreen() # free-field

# CYLINDER GREEN'S FUNCTION
caxis = np.array([1.0, 0.0, 0.0])
corigin = np.array([0.0, 0.0, 0.0])

cg = CylinderGreen(radius=0.5, axis=caxis, origin=corigin, dim=3, 
                           numerics={
                        'nmax': 32, # mind that increasing this increases the chance of overflows, ***should*** be handled by the safe Bessel functions, but beware
                        'Nq_prop': 128,
                        'eps_k' : 1e-6,
                    }) # cylinder

#SOURCE MODE
source = SourceMode(BLH=np.array([1.0 + 1j, 0.5, 0.1]), B=2, gamma=np.deg2rad(30),
                     axis=np.array([0.,1., 1.]), origin=np.array([0, 0.0,1.0]), radius=0.1,
                      #  green=gff  # green's functions are interchangable
                      green = cg
                       )

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")
source.plotSelf(fig, ax)
# ax.set_box_aspect([1, 1, 1])
ax.set_aspect('equal')
plt.show()
plt.close()

# source.plotGeometry()
source.plotFarFieldPressure(m=np.array([1]), Omega=100., Nphi=36, Ntheta=18, R=20.0, valmin=40, valmax=65 )