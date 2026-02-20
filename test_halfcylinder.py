from TayloredGreen.HalfCylinderGreen import HalfCylinderGreen, SF_FullCylinderGreen
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

axis = np.array([1.0, 0.0, 0.0])
origin = np.array([0.0, 0.0, 0.0])
cg = HalfCylinderGreen(radius=0.5, axis=axis, origin=origin, dim=3, 
# cg = SF_FullCylinderGreen(radius=0.5, axis=axis, origin=origin, dim=3, 
                        numerics={
                    'nmax': 64,
                    'Nq_prop': 128,
                    'eps_radius' : 1e-12, # must be lower than eps_eval!
                    'Nazim' : 36, # discretization of the boundary in the azimuth
                    'Nax': 64, # in the axial direction
                    'RMAX': 10, # max radius!
                    'mode': 'geometric', # uniform or geometric, defines the spacing of the surface panels!
                    'geom_factor': 1.025, # geometric stretching factor, only used if mode is 'geometric'
                    'eps_eval' : 1e-6 # evaluation distance from the actual surface, as a fraction of cylinder radius!
                    # Note: the function is currently NOT checking if the panels are compact!
                 })

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")
cg.plotSelf(fig, ax, show_normals=False)
ax.set_aspect('equal')
# ax.set_axis_off()
plt.show()
plt.close()

cg.plotDirectivity(k=np.array([10.0]), y=np.array([[0], [1.0], [0.0]]), R=5.0, Nphi=36*2, Ntheta=18*2,
                   valmin=40, valmax=65)



fig, ax = plt.subplots()

# cg.plotSurfaceGreen(fig=fig, ax=ax, k=np.array([1.0]),  y=np.array([[0], [1.0], [0.]]), levels=20, cmap='viridis')
# # ax.set_aspect('equal')
# # ax.set_axis_off()
# plt.show()
# plt.close()
# cg.plotFarFieldGradient(k=np.array([10.0]), y=np.array([[0], [1.0], [0.]]), R=2.0, Nphi=36, Ntheta=18)


# TODO: plot comparision cylinder full with the two formulations!