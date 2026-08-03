from TailoredGreen.HalfCylinderGreen import HalfCylinderGreen, SF_FullCylinderGreen
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

axis = np.array([1.0, 0.0, 0.0])
origin = np.array([0.0, 0.0, 0.0])
radial = np.array([0.0, 0.0, -1.0])
tangential = np.cross(axis, radial)

RAD = 0.5
# cg = HalfCylinderGreen(radius=RAD, axis=axis, origin=origin, radial=radial, dim=3, 
cg = SF_FullCylinderGreen(radius=RAD, axis=axis, origin=origin,radial=radial,  dim=3, 
                        numerics={
                    'nmax': 32,
                    'Nq_prop': 128,
                    'eps_radius' : 1e-12, # must be lower than eps_eval!
                    'Nazim' : 18, # discretization of the boundary in the azimuth
                    'Nax': 32, # in the axial direction
                    'RMAX': 5, # max radius!
                    'mode': 'uniform', # uniform or geometric, defines the spacing of the surface panels!
                    'geom_factor': 1.025, # geometric stretching factor, only used if mode is 'geometric'
                    'eps_eval' : 1e-6 # evaluation distance from the actual surface, as a fraction of cylinder radius!
                    # Note: the function is currently NOT checking if the panels are compact!
                 })




z = np.linspace(-RAD, RAD, 10)
phi = np.linspace(0, 2 * np.pi, 18)
eps = 1e-6
Z, PHI = np.meshgrid(z, phi)

N = radial[:, None, None]  * np.cos(PHI[None, :, :] ) + tangential[:, None, None]  * np.sin(PHI[None, :, :] )
X = axis[:, None, None] * Z[None, :, :] + N * (RAD+eps) # shape 3, Nz, Nphi

zz = Z.flatten()
x = X.reshape(3,zz.shape[0])
n = N.reshape(3, zz.shape[0])

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")
cg.plotSelf(fig, ax, show_normals=False)
ax.scatter(x[0], x[1], x[2], color='m')
ax.quiver(x[0], x[1], x[2], n[0], n[1], n[2], color='k')


ax.set_aspect('equal')
# ax.set_axis_off()
plt.show()
plt.close()
gradG = cg.getGradientGreenAnalytical(x,y=np.array([[0], [0.0], [1.0]]), k=np.array([10.0]))



dGdn = np.einsum(
    'dn, dmno-> n',
    n,
    gradG
)

pass
# cg.plot3Ddirectivity(k=np.array([10.0]), y=np.array([[0], [0.0], [RAD1]]), R=5.0, Nphi=36*2, Ntheta=18*2,
#                    valmin=40, valmax=65)

# plt.show()


# fig, ax = plt.subplots()

# cg.plotSurfaceGreen(fig=fig, ax=ax, k=np.array([1.0]),  y=np.array([[0], [0.0], [RAD1]]), levels=20, cmap='viridis')
# # ax.set_aspect('equal')
# # ax.set_axis_off()
# plt.show()
# plt.close()
# cg.plotFarFieldGradient(k=np.array([10.0]), y=np.array([[0], [1.0], [0.]]), R=2.0, Nphi=36, Ntheta=18)


# TODO: plot comparision cylinder full with the two formulations!