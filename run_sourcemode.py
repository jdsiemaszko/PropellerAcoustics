from TailoredGreen.CylinderGreen import CylinderGreen
from TailoredGreen.TailoredGreen import TailoredGreen
from SourceMode.SourceMode import SourceMode
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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
                        'mmax': 32, # mind that increasing this increases the chance of overflows, ***should*** be handled by the safe Bessel functions, but beware
                        'Nq_prop': 128,
                        'Nq_evan': 128,
                        'eps_k' : 1e-24,
                    }) # cylinder

#SOURCE MODE
source = SourceMode(BLH=np.array([1.0 + 1j, 0.5, 0.1]), B=2, gamma=np.deg2rad(00),
                     axis=np.array([0.,0.0, 1.]), origin=np.array([0, 0.0,0.0]), radius=D_prop/2,
                    #    green=gff  # green's functions are interchangable
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
source.plotFarFieldPressure(m=np.array([1]), Omega=100., Nphi=36, Ntheta=18, R=20.0, 
                            # valmin=10, valmax=65 
                            )
plt.show()