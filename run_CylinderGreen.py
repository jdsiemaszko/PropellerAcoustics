from TailoredGreen.CylinderGreen import CylinderGreen
from TailoredGreen.TailoredGreen import TailoredGreen
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

axis = np.array([0.0, 0.0, 1.0])
origin = np.array([0.0, 0.0, 0.0])
cg_3D = CylinderGreen(radius=0.5, axis=axis, origin=origin, dim=3, 
                        numerics={
                    'mmax': 32,
                    'Nq_prop': 128,
                    'Nq_evan': 128,
                    # 'eps_k' : 1e-6,
                    'eps_radius' : 1e-24
                })

cg_2D = CylinderGreen(radius=0.5, axis=axis[:2], origin=origin[:2], dim=2, 
                        numerics={
                    'mmax': 32,
                    'eps_radius' : 1e-24
                })


# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(111, projection="3d")
# cg_3D.plotSelf(fig, ax)
# ax.set_aspect('equal')
# # ax.set_axis_off()
# plt.show()
# plt.close()

# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(111, projection="3d")
# cg_2D.plotSelf(fig, ax)
# ax.set_aspect('equal')
# # ax.set_axis_off()
# plt.show()
# plt.close()

# cg_3D.plot3Ddirectivity(k=np.array([10.0]), y=np.array([[0], [1.0], [0.]]), R=5.0, Nphi=36, Ntheta=18,
#                    valmin=10, valmax=65)
# plt.show()

fig, ax = cg_2D.plotGreenOnPlane(k=np.array([10.0]), y=np.array([[0], [1.0]]), rmin = 0.51, rmax= 1.5, Nr=30, Ntheta=36*2, alpha=0.1)
plt.show()
plt.clf()

fig, ax = cg_3D.plotGreenOnPlane(k=np.array([10.0]), y=np.array([[0], [1.0], [0.0]]), rmin = 0.51, rmax= 1.5, Nr=30, Ntheta=36*2, mode='xy', alpha=0.005)
plt.show()
plt.clf()
