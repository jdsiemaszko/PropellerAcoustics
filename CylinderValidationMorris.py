from TailoredGreen.HalfCylinderGreen import HalfCylinderGreen, SF_FullCylinderGreen
from TailoredGreen.CylinderGreen import CylinderGreen
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors



a = 0.5

# Values from Morris .P 1995, figure 7.
L__a = 3.0
Rstar__a = 1.0
ka = 10.0

L = L__a * a
Rstar = Rstar__a * a
k = ka / a

# source locations: circle at (L, 0) of radius Rstar

NPth = 9
NPr  = 5
th = np.linspace(0, 2 * np.pi, NPth)
r = np.linspace(Rstar, 0, NPr, endpoint=False) # avoid point at the center
R, TH = np.meshgrid(r,th, indexing='ij')

rr = R.flatten()
tthh = TH.flatten()

y = np.array([
    L + rr * np.cos(tthh),
    rr * np.sin(tthh)
]) # source positions of shape (2, NPth * NPr)

# Cylinder Module
axis = np.array([1.0, 0.0, 0.0])
origin = np.array([0.0, 0.0, 0.0])
radial = np.array([0.0, 0.0, -1.0])
tangential = np.cross(axis, radial)
cg = CylinderGreen(
    radius=a, axis=axis, origin=origin,radial=radial,  dim=3, 
                     numerics={
                    'nmax': 16,
                    'Nq_prop': 32*8, #discretization of the propagating part
                    'Nq_evan': 32*2, # discretization of the evanescent part
                    'eps_radius' : 1e-32, # cut-off distance
                 })

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")
cg.plotSelf(fig, ax) # plot the cylinder
ax.scatter(y[0], y[1], np.zeros(y.shape[1]), color='m', marker='^') # and the sources
ax.set_aspect('equal')
# ax.set_axis_off()
plt.show()
plt.close()


