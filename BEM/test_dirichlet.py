import bempp_cl.api
import numpy as np
from bempp_cl.api.linalg import gmres
from bempp_cl.api.operators.potential import helmholtz as helmholtz_potential
from matplotlib import pyplot as plt

k = 10.0
RCYL = 0.5
DZ = 10 # extent in the z extrusion direction

grid = bempp_cl.api.shapes.cylinders(r=[RCYL], z=DZ)
piecewise_const_space = bempp_cl.api.function_space(grid, "DP", 0)

identity = bempp_cl.api.operators.boundary.sparse.identity(
    piecewise_const_space, piecewise_const_space, piecewise_const_space
)
adlp = bempp_cl.api.operators.boundary.helmholtz.adjoint_double_layer(
    piecewise_const_space, piecewise_const_space, piecewise_const_space, k
)
slp = bempp_cl.api.operators.boundary.helmholtz.single_layer(
    piecewise_const_space, piecewise_const_space, piecewise_const_space, k
)

lhs = 0.5 * identity + adlp - 1j * k * slp

@bempp_cl.api.complex_callable
def combined_data(x, n, domain_index, result):
    result[0] = 1j * k * np.exp(1j * k * x[0]) * (n[0] - 1)


grid_fun = bempp_cl.api.GridFunction(piecewise_const_space, fun=combined_data)
neumann_fun, info = gmres(lhs, grid_fun, tol=1e-5)

Nx = 200
Ny = 200
xmin, xmax, ymin, ymax = [-3, 3, -3, 3]
plot_grid = np.mgrid[xmin : xmax : Nx * 1j, ymin : ymax : Ny * 1j]
points = np.vstack((plot_grid[0].ravel(), plot_grid[1].ravel(), np.ones(plot_grid[0].size) * DZ/2))
u_evaluated = np.zeros(points.shape[1], dtype=np.complex128)
u_evaluated[:] = np.nan

x, y, z = points
idx = np.sqrt(x**2 + y**2) > RCYL

slp_pot = helmholtz_potential.single_layer(piecewise_const_space, points[:, idx], k)
res = np.real(np.exp(1j * k * points[0, idx]) - slp_pot.evaluate(neumann_fun))
u_evaluated[idx] = res.flat

try:
    get_ipython().run_line_magic("matplotlib", "inline")
    ipython = True
except NameError:
    ipython = False

u_evaluated = u_evaluated.reshape((Nx, Ny))
print('finished BEM routine, plotting')

fig = plt.figure(figsize=(10, 8))
plt.imshow(np.imag(u_evaluated.T), extent=[-3, 3, -3, 3])
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.title("Scattering from a cylinder")
if not ipython:
    plt.savefig(f"CYLINDER_IMAG_k={k:.1f}_R={RCYL:.1f}.png")
plt.show()
plt.close()

fig = plt.figure(figsize=(10, 8))
plt.imshow(np.real(u_evaluated.T), extent=[-3, 3, -3, 3])
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.title("Scattering from a cylinder")
if not ipython:
    plt.savefig(f"CYLINDER_REAL_k={k:.1f}_R={RCYL:.1f}.png")
plt.show()
plt.close()

fig = plt.figure(figsize=(10, 8))
plt.imshow(np.abs(u_evaluated.T), extent=[-3, 3, -3, 3])
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.title("Scattering from a cylinder")
if not ipython:
    plt.savefig(f"CYLINDER_ABS_k={k:.1f}_R={RCYL:.1f}.png")
plt.show()
plt.close()