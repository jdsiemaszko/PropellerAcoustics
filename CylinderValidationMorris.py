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

NPth = 18
NPr  = 20
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
# TODO: set this up in 2 dimensions!
axis = np.array([0.0, 0.0, 1.0])
origin = np.array([0.0, 0.0])
radial = np.array([1.0, 0.0])
tangential = np.cross(axis, radial)
cg = CylinderGreen(
    radius=a, axis=axis, origin=origin,radial=radial,  dim=2, 
                     numerics={
                    'nmax': 32,
                    'eps_radius' : 1e-32, # cut-off distance
                 })

# fig, ax = plt.subplots()
# cg.plotSelf(fig, ax) # plot the cylinder
# ax.scatter(y[0], y[1], color='m', marker='^') # and the sources
# ax.set_aspect('equal')
# # ax.set_axis_off()
# ax.grid()
# plt.show()
# plt.close()

cg.plot2DDirectivity(k, y, R=5*a)
plt.show()

Ntheta= 36 * 4
Nr = 50 * 4
rmin = a * (1+1e-6)
rmax = a * 5
theta = np.linspace(0, 2 * np.pi, Ntheta, endpoint=False)
R = np.linspace(rmin, rmax, Nr)
Theta, R_grid = np.meshgrid(theta, R)
val1 = R_grid * np.cos(Theta)
val2 = R_grid * np.sin(Theta)
x = np.vstack([val1.ravel(), val2.ravel()])  # shape (2, N)

G = cg.getGreenFunction(x, y, k) # Nk, Nx, Ny
p = np.sum(G, axis=-1)[0] # pressure due to Ny unit monopoles

index1_th = np.argmin(np.abs(theta-1.39228872121))
index0_th = np.argmin(np.abs(theta-np.pi))
index_r = np.argmin(np.abs(R-a))

ind_global0 = index_r * Ntheta + index0_th
ind_global1 = index_r * Ntheta + index1_th

# pref = (p[ind_global1]+p[ind_global0])/2 # pick the mean around the cylinder as the reference pressure
pref = p[ind_global1]

p /= pref # normalize



fig, ax = plt.subplots()


im0 = ax.pcolormesh(
    val1/a, val2/a,
    np.real(p.reshape(Nr, Ntheta)),
    shading='auto',
    cmap='viridis',
    # norm=colors.LinNorm(vmin=eps0, vmax=np.abs(G0_reshaped).max())
    norm=colors.CenteredNorm(halfrange = np.max(p)),
    # edgecolor='k',
)

im1 = ax.contour(
    val1/a, val2/a,
    np.real(p.reshape(Nr, Ntheta)),
    levels=[0.5],
    colors='k'   # choose contour color
)

ax.set_xlabel('y/a')
ax.set_ylabel('z/a')
ax.set_xlim((-5, 5))
ax.set_ylim((-5, 5))

cbar = fig.colorbar(im0, ax=ax)
cbar.set_label('Real(p)')
ax.axis('equal')

plt.show()
plt.close()