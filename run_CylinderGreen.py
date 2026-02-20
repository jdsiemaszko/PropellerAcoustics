from TailoredGreen.CylinderGreen import CylinderGreen
from TailoredGreen.TailoredGreen import TailoredGreen
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

axis = np.array([1.0, 0.0, 0.0])
origin = np.array([0.0, 0.0, 0.0])
cg = CylinderGreen(radius=0.5, axis=axis, origin=origin, dim=3, 
                        numerics={
                    'nmax': 32,
                    'Nq_prop': 128,
                    # 'eps_k' : 1e-6,
                    'eps_radius' : 1e-12
                })


fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")
cg.plotSelf(fig, ax)
ax.set_aspect('equal')
# ax.set_axis_off()
plt.show()
plt.close()

cg.plot3Ddirectivity(k=np.array([10.0]), y=np.array([[0], [1.0], [0.]]), R=5.0, Nphi=36, Ntheta=18,
                   valmin=40, valmax=65)