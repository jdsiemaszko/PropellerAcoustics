from TailoredGreen.HalfCylinderGreen import HalfCylinderGreen, SF_FullCylinderGreen
from TailoredGreen.CylinderGreen import CylinderGreen
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors



axis = np.array([1.0, 0.0, 0.0])
origin = np.array([0.0, 0.0, -0.02])
radial = np.array([0.0, 0.0, -1.0])
tangential = np.cross(axis, radial)

RAD = 0.01
RAD_PROP = 0.1
# cg = HalfCylinderGreen(radius=RAD, axis=axis, origin=origin, radial=radial, dim=3, 
# cg_sf = SF_FullCylinderGreen(radius=RAD, axis=axis, origin=origin,radial=radial,  dim=3, 
#                         numerics={
#                     'nmax': 32,
#                     # 'Nq_prop': 128,
#                     'Nq_prop': 64,
#                     'Nq_evan': 32,
#                     'eps_radius' : 1e-24, # must be lower than eps_eval!
#                     'Nazim' : 9, # discretization of the boundary in the azimuth
#                     'Nax': 32, # in the axial direction
#                     'RMAX': 2, # max radius!
#                     'mode': 'uniform', # uniform or geometric, defines the spacing of the surface panels!
#                     'geom_factor': 1.025, # geometric stretching factor, only used if mode is 'geometric'
#                     'eps_eval' : 1e-6 # evaluation distance from the actual surface, as a fraction of cylinder radius!
#                     # Note: the function is currently NOT checking if the panels are compact!
#                  })

nums = {
        'mmax': 16,
        'Nq_prop': 32,
        'Nq_evan': 16,
        'eps_radius' : 1e-32, # cut-off distance
        }

outfile = f'./Data/current/validation/green_on_surface_{nums['mmax']}_{nums['Nq_prop']}_{nums["Nq_evan"]}'
print(f'saving results to output file: {outfile}')

cg = CylinderGreen(
    radius=RAD, axis=axis, origin=origin, radial=radial,  dim=3, 
                     numerics=nums
)



K = np.array([5, 10, 15, 20, 25])
# K = np.array([10])

YPOS = np.array([[RAD_PROP], [0.0], [0]])
Nk = len(K)
# cg.plotScatteringYZ(K, YPOS, rmin = 0.51, rmax= 1.5, Nr=30, Ntheta=36*2)
# # plt.tight_layout()
# plt.show()
# fig, ax = cg.plotSelf()
# ax.scatter(*YPOS)
# plt.show()

# for k in [1, 2, 3, 4, 5]:
er_ar = []
eps_ar = [
    1e-0, 
    1e-1,
           1e-2, 1e-3, 1e-6,
             1e-12
           ]
for eps in eps_ar:
# [1, 2, 3, 4, 5]
    # eps = 1e-3
    #


    z = np.linspace(0, 2 * RAD_PROP, 32)
    phi = np.linspace(0, 2 * np.pi, 18)
    Z, PHI = np.meshgrid(z, phi, indexing='ij')

    N = radial[:, None, None]  * np.cos(PHI[None, :, :] ) + tangential[:, None, None]  * np.sin(PHI[None, :, :] )
    X = origin[:, None, None] + axis[:, None, None] * Z[None, :, :] + N * RAD * (1+eps) # shape 3, Nz, Nphi

    zz = Z.flatten()
    x = X.reshape(3, zz.shape[0]) # Nd, Nx
    n = N.reshape(3, zz.shape[0]) # Nd, Nx

    # fig = plt.figure(figsize=(7, 7))
    # ax = fig.add_subplot(111, projection="3d")
    # cg.plotSelf(fig, ax)
    # ax.scatter(x[0], x[1], x[2], color='m')
    # ax.quiver(x[0], x[1], x[2], n[0], n[1], n[2], color='k')
    # ax.set_aspect('equal')
    # # ax.set_axis_off()
    # plt.show()
    # plt.close()


    ############## DIRECT - INCORRECT!!!!
    # gradG0 = cg.getFreeSpaceGreenGradient(x,y=YPOS, k=K) # nabla_y G(x|y) = nabla_y G(y|x), evaluated explicitly
    # gradGs = cg.getScatteringGreenGradient(x,y=YPOS, k=K) # shape 3, Nk, Nx, Ny 

    # # would need: y on the boundary and x somewhere in the domain

    # # construct d/dn_x G(x|y) = n_x * nabla_x G(x|y)
    # dG0dn = np.einsum(
    #     'dx, dxyo-> x',
    #     n,
    #     gradG0
    # )

    # dGsdn = np.einsum(
    #     'dx, dxyo-> x',
    #     n,
    #     gradGs
    # ) 


    ############# USING RECIPROCITY
    gradG0 = cg.getFreeSpaceGreenGradient(YPOS, x, k=K)  # nabla_x G(y|x) = nabla_x G(x|y) by reciprocity, x approx. on del Omega
    gradGs = cg.getScatteringGreenGradient(YPOS, x, k=K) # shape 3, Nk, Ny, Nx 


    # construct d/dn_x G(x|y) = n_x * nabla_x G(x|y)
    # x approx on the boundary!
    dG0dn = np.einsum(
        'dx, dkyx-> kx',
        n,
        gradG0
    )

    dGsdn = np.einsum(
        'dx, dkyx-> kx',
        n,
        gradGs
    ) 

    error = dGsdn + dG0dn # shape Nk, Nx

    error2D = error.reshape((Nk, z.shape[0], phi.shape[0]))
    # plt.plot(np.real(dG0dn[0]))
    # plt.plot(np.real(dGsdn[0]))
    # plt.plot(np.real(error[0]), color='k', linestyle='dashed')
    # plt.show()
    # plt.close()
    # plt.contourf(Z, np.rad2deg(PHI),np.abs(error2D[0]))
    # plt.colorbar()
    # plt.show()

    print(np.max(np.abs(dG0dn), axis=-1))
    print(np.max(np.abs(dGsdn), axis=-1))
    print(np.max(np.abs(error), axis=-1))
    L2_error = error.std(axis=-1) # same as ||error||_L2 / ||1||_L2
    er_ar.append(L2_error)
    print(L2_error)

fig, ax = plt.subplots()

ax.plot(eps_ar, er_ar, marker='s')


np.savez(
    outfile,
    L2 = np.array(er_ar), # shape Neps, Nk
    eps = np.array(eps_ar) # shape Neps
)
ax.set_xscale('log')
ax.set_yscale('log')

ax.grid()

ax.set_xlabel('$\delta r / a$')
ax.set_ylabel('$\|\epsilon \|_2$')

plt.tight_layout()
plt.show()


