from TayloredGreen.CylinderGreen import CylinderGreen
from TayloredGreen.TayloredGreen import TayloredGreen
from SourceMode.SourceMode import SourceMode
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# green = CylinderGreen(axis=np.array([1.,0,0]), radius=0.5, origin=np.array([0,0,0]))

gff = TayloredGreen() # free-field
caxis = np.array([1.0, 0.0, 0.0])
corigin = np.array([0.0, 0.0, 0.0])
cg = CylinderGreen(radius=0.5, axis=caxis, origin=corigin, dim=3, 
                           numerics={
                        'nmax': 16, # mind that increasing this increases the chance of overflows (numbers get big!)
                        'Nq_prop': 128,
                        'eps_k' : 1e-6,
                    }) # cylinder
source = SourceMode(BLH=np.array([1.0 + 1j, 0.5, 0.1]), B=2, gamma=0.05,
                     axis=np.array([0.,1., 1.]), origin=np.array([0, 0.0,1.0]), radius=0.1,
                       green=gff
                       )

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")
source.plotSelf(fig, ax)
# ax.set_box_aspect([1, 1, 1])
ax.set_aspect('equal')
plt.show()
plt.close()

# source.plotGeometry()
source.plotFarFieldPressure(m=np.array([1]), Omega=100., Nphi=36, Ntheta=18, R=2.0)