from TayloredGreen.CylinderGreen import CylinderGreen
from TayloredGreen.TayloredGreen import TayloredGreen
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

axis = np.array([1.0, 0.0, 0.0])
origin = np.array([0.0, 0.0, 0.0])
cg = CylinderGreen(radius=0.5, axis=axis, origin=origin, dim=3, 
                        numerics={
                    'nmax': 64,
                    'Nq_prop': 128,
                    # 'eps_k' : 1e-6,
                    'eps_radius' : 1e-12
                })


# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(111, projection="3d")
# # source.plotSelf(fig, ax)
# cg.plotSelf(fig, ax)
# # ax.set_box_aspect([1, 1, 1])
# ax.set_aspect('equal')
# ax.set_axis_off()
# plt.show()
# plt.close()

# cg.plotScatteringYZ(y=np.array([[0.0], [1.0], [0.0]]), k=np.array([10.0]), rmin=0.5, rmax=5.0)
cg.plotDirectivity(k=np.array([10.0]), y=np.array([[0], [1.0], [0.]]), R=5.0, Nphi=36*2, Ntheta=18*2,
                   valmin=40, valmax=65)
# cg.plotFarFieldGradient(k=np.array([10.0]), y=np.array([[0], [1.0], [0.]]), R=2.0, Nphi=36, Ntheta=18, )

# wavy in the x-y plane? - why?