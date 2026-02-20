import bempp_cl.api
from bempp_cl.api.operators.boundary import helmholtz, sparse
from bempp_cl.api.operators.potential import helmholtz as helmholtz_potential
from bempp_cl.api.linalg import gmres
import numpy as np
from matplotlib import pyplot as plt

k = 10.
RCYL = 0.5
DZ = 10 # extent in the z extrusion direction

grid = bempp_cl.api.shapes.cylinders(r=[RCYL], z=DZ)

space = bempp_cl.api.function_space(grid, "DP", 0)

identity = sparse.identity(space, space, space)
double_layer = helmholtz.double_layer(space, space, space, k)

@bempp_cl.api.complex_callable
def p_inc_callable(x, n, domain_index, result):
    result[0] = np.exp(1j * k * x[0])


p_inc = bempp_cl.api.GridFunction(space, fun=p_inc_callable)
p_total, info = gmres(double_layer - 0.5 * identity, -p_inc, tol=1E-5)

Nx = 200
Ny = 200
xmin, xmax, ymin, ymax = [-3, 3, -3, 3]
plot_grid = np.mgrid[xmin:xmax:Nx * 1j, ymin:ymax:Ny * 1j]
points = np.vstack((plot_grid[0].ravel(),
                    plot_grid[1].ravel(),
                    np.ones(plot_grid[0].size) * DZ/2))

p_inc_evaluated = np.real(np.exp(1j * k * points[0, :]))
p_inc_evaluated = p_inc_evaluated.reshape((Nx, Ny))

vmax = max(np.abs(p_inc_evaluated.flat))

try:
    get_ipython().run_line_magic("matplotlib", "inline")
    ipython = True
except NameError:
    ipython = False

fig = plt.figure(figsize=(10, 8))
plt.imshow(np.real(p_inc_evaluated.T), extent=[-3, 3, -3, 3], cmap=plt.get_cmap("bwr"), vmin=-vmax, vmax=vmax)
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.title("Incident Wave")
if not ipython:
    plt.savefig(f"NEUMANN/CYLINDER_INCIDENT_k={k:.1f}_R={RCYL:.1f}.png")
plt.show()
plt.close()

double_pot = helmholtz_potential.double_layer(space, points, k)
p_s = np.real(double_pot.evaluate(p_total))
p_s = p_s.reshape((Nx, Ny))

vmax = max(np.abs(p_s.flat))

fig = plt.figure(figsize=(10, 8))
plt.imshow(np.real(p_s.T), extent=[-3, 3, -3, 3], cmap=plt.get_cmap("bwr"), vmin=-vmax, vmax=vmax)
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.title("Scattered Wave")
if not ipython:
    plt.savefig(f"NEUMANN/CYLINDER_SCATTERED_k={k:.1f}_R={RCYL:.1f}.png")
plt.show()
plt.close()


vmax = max(np.abs((p_inc_evaluated + p_s).flat))

fig = plt.figure(figsize=(10, 8))
plt.imshow(np.real((p_inc_evaluated + p_s).T), extent=[-3, 3, -3, 3],
           cmap=plt.get_cmap("bwr"), vmin=-vmax, vmax=vmax)
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.title("Total Wave")
if not ipython:
    plt.savefig(f"NEUMANN/CYLINDER_TOTAL_k={k:.1f}_R={RCYL:.1f}.png")
plt.show()
plt.close()