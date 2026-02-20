from matplotlib import axis
from .TailoredGreen import TailoredGreen
from .CylinderGreen import CylinderGreen
import numpy as np
from numpy.polynomial.legendre import leggauss
from Constants.helpers import p_to_SPL
import matplotlib.pyplot as plt

class SurfacePotentialGreen(TailoredGreen): # Note: subclass the main object, not the CylinderGreen!

    def __init__(self, radius:float, axis:np.ndarray, origin:np.ndarray, radial:np.ndarray=None, dim=3,
                 numerics={
                    'nmax': 16,
                    'Nq_prop': 100,
                    'Nazim' : 18, # discretization of the boundary in the azimuth
                    'Nax': 64, # in the axial direction
                    'RMAX': 5, # max radius!
                    'mode': 'uniform', # uniform or geometric, defines the spacing of the surface panels!
                    'geom_factor': 1.01, # geometric stretching factor, only used if mode is 'geometric'
                    'eps_eval':1e-3 # distance from the boundary for the boundary evaluation (as a fraction of cylinder radius!)
                 }
                 
                 ):
        super().__init__(dim=dim)
        self.radius = radius
        self._numerics = numerics
        self.axis = axis / np.linalg.norm(axis) # axis vector of the cylinder
        self.origin = origin # a point on the cylinder axis, taken as the origin of the cylindrical coordinate system
        if radial is None:
            # choose an arbitrary radial direction perpendicular to the axis
            if np.allclose(self.axis, np.array([1,0,0])):
                temp_vec = np.array([0,1,0])
            else:
                temp_vec = np.array([1,0,0])
            self.radial = temp_vec - np.dot(temp_vec, self.axis) * self.axis
            self.radial /= np.linalg.norm(self.radial)
        else:
            self.radial = radial / np.linalg.norm(radial) # radial vector of the cylinder, taken as the zero azimuth direction
        self.normal = np.cross(self.axis, self.radial)
        self.free_field_green = TailoredGreen(
            dim=dim
        ) # also store the free-field solution!

        self.panel_positions, self.panel_normals, self.panel_areas, self.panel_z_edges, self.panel_th_edges = self.getBoundaryDiscretization()

    def getBoundaryDiscretization(self):
        # boundary discretization for the geometry
        # leave arbitrary in the parent class
        pos = np.array([[0], [0], [0]]) # shape (3, N)
        normals = np.array([[0], [0], [1]]) # shape (3, N)
        areas = np.array([1]) # shape (N,)
        z_edges = np.array([0, 1]) # shape (Nax+1,)
        th_edges = np.array([0, np.pi]) # shape (Nazim+1,)
        return pos, normals, areas, z_edges, th_edges
    
    def getBoundaryEvaluationPoints(self):
        eps_eval = self._numerics.get('eps_eval', 1e-3)
        panel_centers = self.panel_positions
        normals = self.panel_normals
        eval_centers = panel_centers + eps_eval * self.radius * normals # shape (3, N)!
        return eval_centers

    def _getScatteringGreen(self, x, y, k, green_at_surface):
        
        eval_points = self.getBoundaryEvaluationPoints() # (3, Nz)

        # ____________prediction from Roger et al. 2023______________________
        # effectively, we're solving G = G0 + G0 @ V @ G for V defining the surface potential
        # or equivalently G|f> = |p> = G0 |f> + G0 @ V @ G |f> = p0 + G0 @ V |p>, see "Born Series" in QM
        # caveat: G on the RHS is replaced by a predictor, in this case coming from the full cylinder

        # use the full cylinder as a predictor 
        # green_at_surface = self.full_cylinder_green.getGreenFunction(eval_points, y, k) # (Nk, Nz, Ny)
        # compute the gradient of ff green on the surface (NOTE: could use exact positions instead of eval!)
        ff_green_gradient = self.free_field_green.getGradientGreenAnalytical(x, eval_points, k) # (3, Nk, Nx, Nz)
        eval_areas = self.panel_areas # Nz
        panel_normals = self.panel_normals # (3, Nz)

        ff_green_normal = -np.einsum(
            'dz, dkxz -> kxz',
            panel_normals,
            ff_green_gradient
        ) # reduce to normal derivative only
    
        # reduce to (Nk, Nx, Ny) (integrating over the surface!)
        Gs = -np.einsum(
            'kzy, kxz, z -> kxy',  # Einstein summation
            green_at_surface,
            ff_green_normal,
            eval_areas,
        )

        return Gs

    def getScatteringGreen(self, x, y, k):
        green_at_surface = self.free_field_green.getGreenFunction(self.getBoundaryEvaluationPoints(), y, k) # (Nk, Nz, Ny)
        return self.__getScatteringGreen(x, y, k, green_at_surface)
    
    def _getScatteringGreenGradient(self, x, y, k, green_grad_at_surface):
        
        eval_points = self.getBoundaryEvaluationPoints() # (3, Nz)

        # ____________prediction from Roger et al. 2023______________________
        # effectively, we're solving ∇yG = ∇yG0 + ∇y(G0 @ V @ G) = (G0 @ V @ ∇yG) for V defining the surface potential

        # use the full cylinder as a predictor 
        # green_grad_at_surface = self.full_cylinder_green.getGradientGreenAnalytical(eval_points, y, k) # (3, Nk, Nz, Ny)
        # compute the gradient of ff green on the surface (NOTE: could use exact positions instead of eval!)
        ff_green_gradient = self.free_field_green.getGradientGreenAnalytical(x, eval_points, k) # (3, Nk, Nx, Nz)
        eval_areas = self.panel_areas # Nz
        panel_normals = self.panel_normals # (3, Nz)

        ff_green_normal = -np.einsum(
            'dz, dkxz -> kxz',
            panel_normals,
            ff_green_gradient
        ) # reduce to normal derivative only
    
        # reduce to (Nk, Nx, Ny) (integrating over the surface!)
        nablaGs = -np.einsum(
            'dkzy, kxz, z -> dkxy',  # Einstein summation
            green_grad_at_surface,
            ff_green_normal,
            eval_areas,
        ) 

        return nablaGs #shape (3, Nk, Nx, Ny)

    def getScatteringGreenGradient(self, x, y, k):
        green_grad_at_surface = self.free_field_green.getGradientGreenAnalytical(self.getBoundaryEvaluationPoints(), y, k) # (3, Nk, Nz, Ny)
        return self.__getScatteringGreenGradient(x, y, k, green_grad_at_surface)
    
    def plotSelf(self, fig, ax, normal_scale=None, stride=1, show_mesh=True, show_normals=True):
        """
        High-level plotting function for the discretized geometry.
        """

        self.plotCenters(fig, ax)

        if show_normals:
            self.plotNormals(fig, ax, normal_scale=None, stride=1)
        if show_mesh:
            self.plotMesh(fig, ax)
        self._setEqualAspect(fig, ax)

        return fig, ax
    
    def _setEqualAspect(self,fig, ax):
        pos = self.panel_positions

        max_range = np.ptp(pos, axis=1).max()
        mid = pos.mean(axis=1)

        ax.set_xlim(mid[0]-max_range/2, mid[0]+max_range/2)
        ax.set_ylim(mid[1]-max_range/2, mid[1]+max_range/2)
        ax.set_zlim(mid[2]-max_range/2, mid[2]+max_range/2)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    def plotCenters(self, fig, ax):
        """
        Plot panel center locations.
        """
        pos = self.panel_positions

        ax.scatter(pos[0], pos[1], pos[2], marker='o', color='r')

        return fig, ax

    def plotMesh(self, fig, ax):
        """
        Plot panel mesh edges.
        """
        return fig, ax

    def plotNormals(self, fig, ax, stride = 1, normal_scale=None):
        """
        Plot normal vectors.
        """
        pos = self.panel_positions
        normals = self.panel_normals

        if normal_scale is None:
            normal_scale = 0.5 * self.radius

        idx = np.arange(0, pos.shape[1], stride)

        ax.quiver(
            pos[0, idx],
            pos[1, idx],
            pos[2, idx],
            normals[0, idx],
            normals[1, idx],
            normals[2, idx],
            length=normal_scale,
            normalize=True,
            color='b'
        )

        return fig, ax

    def _plotSurfaceSolution(self, fig, ax, sol, levels=20, cmap='viridis', title=None):
        # plot the solution against two surface coordinates z and th
        z_edges = self.panel_z_edges
        th_edges = self.panel_th_edges
        z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
        th_centers = 0.5 * (th_edges[:-1] + th_edges[1:])

        Nax = len(z_edges) - 1
        Nazim = len(th_edges) - 1

        # --- reshape consistent with construction ---
        sol_2d = sol.reshape(Nax, Nazim)

        # --- shift theta from [0, 2π) → [-π, π) ---
        th_shifted = (th_centers + np.pi) % (2*np.pi) - np.pi

        # --- sort theta so it increases from -π to π ---
        sort_idx = np.argsort(th_shifted)

        th_sorted = th_shifted[sort_idx]
        sol_2d = sol_2d[:, sort_idx]   # reorder along azimuth axis

        # --- meshgrid ---
        Z, TH = np.meshgrid(z_centers, th_sorted, indexing='ij')

        cf = ax.contourf(Z, np.rad2deg(TH), sol_2d, levels=levels, cmap=cmap)

        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label("Directivity [dB]")

        ax.set_ylabel("Theta [deg]")
        ax.set_xlabel("Z [-]")
        if title is not None:
            ax.set_title(title)

        return fig, ax
    
    def plotSurfaceGreen(self, fig, ax, y, k, levels=20, cmap='viridis', title=None):
        gr = self.getGreenFunction(self.getBoundaryEvaluationPoints(), y, k)
        self._plotSurfaceSolution(fig, ax, p_to_SPL(gr), levels=levels, cmap=cmap, title=title)
        return fig, ax
    
    def plotSurfaceFFGreen(self, fig, ax, y, k, levels=20, cmap='viridis', title=None):
        gr = self.getFreeSpaceGreen(self.getBoundaryEvaluationPoints(), y, k)
        self._plotSurfaceSolution(fig, ax, p_to_SPL(gr), levels=levels, cmap=cmap, title=title)
        return fig, ax

class HalfCylinderGreen(SurfacePotentialGreen):
    def __init__(self, radius, axis, origin, radial = None, dim=3,
                  numerics={ 'nmax': 16,'Nq_prop': 100,'Nazim': 18,'Nax': 64,
                            'RMAX': 5,'mode': 'uniform','geom_factor': 1.01,'eps_eval': 0.001 }):
        super().__init__(radius, axis, origin, radial, dim, numerics)
        self.full_cylinder_green = CylinderGreen(
            radius, axis, origin, radial, dim=dim,
                 numerics=numerics) # store the full cylinder module as a helper function!

    def getBoundaryDiscretization(self):
        # construct boundary discretization
        # return: panel positions (3, Npanels), panel normals (3, Npanels), panel areas (Npanels), and edges of the two surface

        Nazim = self._numerics.get('Nazim', 18)
        Nax   = self._numerics.get('Nax', 64)
        Rmax  = self._numerics.get('RMAX', 5) * self.radius
        mode  = self._numerics.get('mode', 'uniform')
        geom_ratio = self._numerics.get('geom_factor', 1.01)

        # --- axial discretization: from 0 → Rmax ---
        if mode == 'geometric':
            g = geom_ratio

            if np.isclose(g, 1.0):
                dz = np.full(Nax, Rmax / Nax)
            else:
                a = Rmax * (g - 1.0) / (g**Nax - 1.0)
                dz = a * g**np.arange(Nax)

            z_edges = np.concatenate(([0.0], np.cumsum(dz)))

        elif mode == 'uniform':
            z_edges = np.linspace(0.0, Rmax, Nax + 1)
            # z_edges = np.linspace(-Rmax, Rmax, Nax + 1) # FULL CYLINDER INSTEAD


        else:
            raise ValueError(f'mode {mode} not recognized, see docstring')

        z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
        dz = np.diff(z_edges)

        # --- radial discretization (FULL circumference) ---
        th_edges = np.linspace(0.0, 2.0 * np.pi, Nazim + 1)
        th_centers = 0.5 * (th_edges[:-1] + th_edges[1:])
        dth = np.diff(th_edges)

        # --- allocate arrays ---
        Npan = Nazim * Nax
        pos = np.zeros((3, Npan))
        normals = np.zeros((3, Npan))
        areas = np.zeros(Npan)

        k = 0
        for zc, dzi in zip(z_centers, dz):
            for thc, dthi in zip(th_centers, dth):

                radial = np.cos(thc) * self.radial + np.sin(thc) * self.normal

                pos[:, k] = (
                    self.origin
                    + zc * self.axis
                    + self.radius * radial
                )

                normals[:, k] = radial
                areas[k] = self.radius * dthi * dzi

                k += 1

        return pos, normals, areas, z_edges, th_edges
    
    def getScatteringGreen(self, x, y, k):
        # use cylinder green as the predictor!
        green_at_surface = self.full_cylinder_green.getGreenFunction(self.getBoundaryEvaluationPoints(), y, k) # (Nk, Nz, Ny)
        return self._getScatteringGreen(x, y, k, green_at_surface)
    
    def getScatteringGreenGradient(self, x, y, k):
        # use cylinder green as the predictor!
        green_grad_at_surface = self.full_cylinder_green.getGradientGreenAnalytical(self.getBoundaryEvaluationPoints(), y, k) # (3, Nk, Nz, Ny)
        return self._getScatteringGreenGradient(x, y, k, green_grad_at_surface)
    
    def plotMesh(self, fig, ax):

        """
        Plot panel mesh edges.
        """

        z_edges=self.panel_z_edges
        th_edges=self.panel_th_edges


        # axial rings
        for z in z_edges:
            circle = (
                self.origin[:, None]
                + z * self.axis[:, None]
                + self.radius * (
                    np.cos(th_edges)[None, :] * self.radial[:, None]
                    + np.sin(th_edges)[None, :] * self.normal[:, None]
                )
            )
            ax.plot(circle[0], circle[1], circle[2], color='r', linestyle='dashed')

        # radial lines
        for th in th_edges:
            line = (
                self.origin[:, None]
                + z_edges[None, :] * self.axis[:, None]
                + self.radius * (
                    np.cos(th) * self.radial[:, None]
                    + np.sin(th) * self.normal[:, None]
                )
            )
            ax.plot(line[0], line[1], line[2], color='r', linestyle='dashed')

        return fig, ax
    
    def plotSurfaceGreen(self, fig, ax, y, k, levels=20, cmap='viridis', title=None):
        gr = self.full_cylinder_green.getGreenFunction(self.getBoundaryEvaluationPoints(), y, k)
        self._plotSurfaceSolution(fig, ax, p_to_SPL(gr), levels=levels, cmap=cmap, title=title)
        return fig, ax

class SF_FullCylinderGreen(HalfCylinderGreen):
    def getBoundaryDiscretization(self):
        # construct boundary discretization
        # return: panel positions (3, Npanels), panel normals (3, Npanels), panel areas (Npanels), and edges of the two surface

        Nazim = self._numerics.get('Nazim', 18)
        Nax   = self._numerics.get('Nax', 64)
        Rmax  = self._numerics.get('RMAX', 5) * self.radius
        mode  = self._numerics.get('mode', 'uniform')
        geom_ratio = self._numerics.get('geom_factor', 1.01)

        # --- axial discretization: from 0 → Rmax ---
        if mode == 'geometric':
            g = geom_ratio

            if np.isclose(g, 1.0):
                dz = np.full(Nax, Rmax / (Nax//2))
            else:
                a = Rmax * (g - 1.0) / (g**(Nax//2) - 1.0)
                dz = a * g**np.arange(Nax//2)

            cs = np.cumsum(dz)
            z_edges = np.concatenate((-cs[::-1], [0.0], cs)) # FULL CYLINDER

        elif mode == 'uniform':
            z_edges = np.linspace(-Rmax, Rmax, Nax + 1) # FULL CYLINDER


        else:
            raise ValueError(f'mode {mode} not recognized, see docstring')

        z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
        dz = np.diff(z_edges)

        # --- radial discretization (FULL circumference) ---
        th_edges = np.linspace(0.0, 2.0 * np.pi, Nazim + 1)
        th_centers = 0.5 * (th_edges[:-1] + th_edges[1:])
        dth = np.diff(th_edges)

        # --- allocate arrays ---
        Npan = Nazim * Nax
        pos = np.zeros((3, Npan))
        normals = np.zeros((3, Npan))
        areas = np.zeros(Npan)

        k = 0
        for zc, dzi in zip(z_centers, dz):
            for thc, dthi in zip(th_centers, dth):

                radial = np.cos(thc) * self.radial + np.sin(thc) * self.normal

                pos[:, k] = (
                    self.origin
                    + zc * self.axis
                    + self.radius * radial
                )

                normals[:, k] = radial
                areas[k] = self.radius * dthi * dzi

                k += 1

        return pos, normals, areas, z_edges, th_edges

class HalfCylinderGreem_Iterative(HalfCylinderGreen):
    def __init__(self, radius, axis, origin, radial = None, dim=3,
                  numerics={ 'nmax': 16,'Nq_prop': 100,'Nazim': 18,'Nax': 64,
                            'RMAX': 5,'mode': 'uniform','geom_factor': 1.01,'eps_eval': 0.001, 'iter_max':3, 'error_tol':1e-3}):
        super().__init__(radius, axis, origin, radial, dim, numerics)

    def getScatteringGreen(self, x, y, k):
        iter_green = self.full_cylinder_green.getGreenFunction(self.getBoundaryEvaluationPoints(), y, k) # (Nk, Nz, Ny)
        ITER_COUNT = self._numerics.get('MAXITER', 3)
        for iter in range(ITER_COUNT):
            print(f'Iteration {iter+1}/{ITER_COUNT}')
            # iterate the boundary solution using the previous iteration as the surface solution (note: this is hypersingular in principle)
            iter_green = self._getScatteringGreen(self.getBoundaryEvaluationPoints(), y, k, iter_green)
            
            norm_L_infty = np.max(np.abs(iter_green))
            error_rel = np.linalg.norm
        return self._getScatteringGreen(x, y, k, iter_green)
    