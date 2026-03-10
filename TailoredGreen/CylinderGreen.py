from matplotlib import axis
from .TailoredGreen import TailoredGreen
import numpy as np
from scipy.special import hankel1, jv, iv, kv, h1vp, hankel1e, jvp
from numpy.polynomial.legendre import leggauss
from Constants.const import PREF
from Constants.helpers import p_to_SPL, getCylindricalCoordinates, getPolarCoordinates
import matplotlib.pyplot as plt

def CylinderGreen2D(x, y, k, radius:float, mmax:int=32, eps_radius:float=1e-24):
    """
    2D Cylinder Green's function 
    solving (nabla^2 + k^2)G(x|y) = -delta(x-y)

    x - r, phi - observer positions in POLAR of size Nx each
    y - same as above for source positions of size Ny each
    k - wave numbers of size Nk
    radius - cylinder radius

    All computations done in parallel. Should be used with discretion if applied to large input arrays

    returns: G(x|y; k) of size Nk, Nx, Ny
    """

    x = np.array(x)
    y = np.array(y)
    k = np.array(k)

    k = np.atleast_1d(k)
    Nk = k.size
    Nx = x.shape[1]
    Ny = y.shape[1]

    obs_r, obs_phi = x[0], x[1] # Nx each
    src_r, src_phi = y[0], y[1] # Ny each

    m = np.arange(0, mmax) # Nm
    epsm = np.ones(mmax) 
    epsm[1:] = 2

    # --- beta using recurrence Hankels ---
    Hm_minus = hankel1(m[None,:] - 1, k[:, None] * radius) # shape Nk, Nm
    Hm_plus  = hankel1(m[None,:] + 1, k[:, None] * radius)

    Jm_minus = jv(m[None,:] - 1, k[:, None] * radius)
    Jm_plus  = jv(m[None,:] + 1, k[:, None] * radius)

    denom = Hm_minus - Hm_plus
    num = Jm_minus - Jm_plus
    # beta = (Jm_minus - Jm_plus) / (Hm_minus - Hm_plus)
    safe  = np.logical_and(np.abs(denom) > 1e-100, np.abs(denom) < 1e100)
    beta  = np.where(safe, num / np.where(safe, denom, 1.0), 0.0) # Nk, Nm

    # Nk, Nm, Ny
    H_src = hankel1(m[None, :, None], k[:, None, None] * src_r[None, None, :]) * (src_r[None, None, :] > radius * (1+eps_radius)) # ignore any part inside the cylinder!

    dphi = obs_phi[:, None] - src_phi[None, :] # Nx, Ny

    cosm = np.cos(m[:, None, None] * dphi[None, :, :])  # Nm, Nx, Ny

    # Nk, Nm, Nx
    H_obs = hankel1(m[None, :, None], k[:, None, None]  * obs_r[None, None, :]) * (obs_r[None, None, :] > radius * (1+eps_radius)) # ignore any part inside the cylinder!

    # A = beta[:, :, None, None] * H_obs[:, :, :, None] * H_src[:, :, None, :] # Nk, Nm, Nx, Ny

    G = np.einsum(
        "km,kmx,kmy,mxy,m->kxy",
        beta,
        H_obs,
        H_src,
        cosm,
        epsm,
        optimize=True
    )

    if np.any(np.isnan(G)):
        print('WARNING: NaN values detected in the computation')

    G *= (-1j / 4) #mind the convention is different by (-1) from Zamponi et al. 2024 and Gloerfelt et al. 2005

    return G

def CylinderGreenGradient2D(x, y, k, radius:float, mmax:int=32, eps_radius:float=1e-24, return_G=True):
    """
    2D Cylinder Green's function's GRADIENT W.R.T SOURCE COORDINATES (y) 
    solving (nabla^2 + k^2)G(x|y) = -delta(x-y)

    x - r, phi - observer positions in POLAR of size Nx each
    y - same as above for source positions of size Ny each
    k - wave numbers of size Nk
    radius - cylinder radius

    All computations done in parallel. Should be used with discretion if applied to large input arrays
    returns: nabla_y * G(x|y; k) of size 2, Nk, Nx, Ny
    also returns G(x|y; k) of size Nk, Nx, Ny if flag "return_G" is passed as True
    """

    x = np.array(x)
    y = np.array(y)
    k = np.array(k)

    k = np.atleast_1d(k)
    Nk = k.size
    Nx = x.shape[1]
    Ny = y.shape[1]

    obs_r, obs_phi = x[0], x[1] # Nx each
    src_r, src_phi = y[0], y[1] # Ny each

    m = np.arange(0, mmax) # Nm
    epsm = np.ones(mmax) 
    epsm[1:] = 2

    # --- beta using recurrence Hankels ---
    Hm_minus = hankel1(m[None,:] - 1, k[:, None] * radius) # shape Nk, Nm
    Hm_plus  = hankel1(m[None,:] + 1, k[:, None] * radius)

    Jm_minus = jv(m[None,:] - 1, k[:, None] * radius)
    Jm_plus  = jv(m[None,:] + 1, k[:, None] * radius)

    denom = Hm_minus - Hm_plus
    num = Jm_minus - Jm_plus
    # beta = (Jm_minus - Jm_plus) / (Hm_minus - Hm_plus)
    safe  = np.logical_and(np.abs(denom) > 1e-100, np.abs(denom) < 1e100)
    beta  = np.where(safe, num / np.where(safe, denom, 1.0), 0.0) # Nk, Nm

    # Nk, Nm, Ny
    H_src = hankel1(m[None, :, None], k[:, None, None] * src_r[None, None, :]) * (src_r[None, None, :] > radius * (1+eps_radius)) # ignore any part inside the cylinder!
    
    # derivative via recurrence
    H_src_minus = hankel1(
        m[None, :, None] - 1, k[:, None, None] * src_r[None, None, :]
    )
    H_src_plus = hankel1(
        m[None, :, None] + 1, k[:, None, None] * src_r[None, None, :]
    )

    dH_src_dry = k[:, None, None] * 0.5 * (
        H_src_minus - H_src_plus
    ) * (src_r[None, None, :] > radius * (1+eps_radius))


    dphi = obs_phi[:, None] - src_phi[None, :] # Nx, Ny

    cosm = np.cos(m[:, None, None] * dphi[None, :, :])  # Nm, Nx, Ny
    dcosm_dphiy = m[:, None, None] * np.sin(m[:, None, None] * dphi[None, :, :])  # Nm, Nx, Ny

    # Nk, Nm, Nx
    H_obs = hankel1(m[None, :, None], k[:, None, None]  * obs_r[None, None, :]) * (obs_r[None, None, :] > radius * (1+eps_radius)) # ignore any part inside the cylinder!

    # A = beta[:, :, None, None] * H_obs[:, :, :, None] * H_src[:, :, None, :] # Nk, Nm, Nx, Ny

    dG_dry = np.einsum( # dG/dr
        "km,kmx,kmy,mxy,m->kxy",
        beta,
        H_obs,
        dH_src_dry,
        cosm,
        epsm,
        optimize=True
    )

    dG_dphiy_ry = np.einsum( # 1/r dG/dphi_y
        "km,kmx,kmy,mxy,m->kxy",
        beta,
        H_obs,
        H_src,
        dcosm_dphiy / src_r[None, None, :], 
        epsm,
        optimize=True
    )    #mind the convention is different by (-1) from Zamponi et al. 2024 and Gloerfelt et al. 2005

    gradG  = np.stack([
        dG_dry, dG_dphiy_ry
    ])  # shape 2, Nk, Nx, Ny

    gradG *= (-1j / 4) # NOTE: IMPORTANT: Python is buggy when applying this pre-factor directly on the einsum, do this explicitly afterwards


    if return_G:
        G = np.einsum(
        "km,kmx,kmy,mxy,m->kxy",
        beta,
        H_obs,
        H_src,
        cosm,
        epsm,
        optimize=True
        ) 
        G *= (-1j / 4)

    if np.any(np.isnan(gradG)):
        print('WARNING: NaN values detected in the computation')
    
    if return_G:
        return gradG, G
    return gradG

def get_quadrature_in_kz(k0:float, Nq_prop=128, Nq_evan=128):
    """
    quadrature in the kz integral
    returns: kz, kk, w
    kz - integration variable, distributed over 0->infinity
    w - integration weights
    kk - sqrt(k0^2-kz^2), distributed over k->0 and 0j->infinity * 1j (helper variable)

    integration of function f(*) on 0->infty can then be approximated as sum(f(kz) * w)
    """

    # propagating part
    theta0, w_theta0 = arcsin_dist(Nq_prop)
    theta0 = 0.25 * np.pi * (theta0 + 1.0)
    w_theta0 *= 0.25 * np.pi

    sin_t0 = np.sin(theta0)
    cos_t0 = np.cos(theta0)

    kz0 = k0 * sin_t0
    kk0 = k0 * cos_t0
    w0  = k0 * cos_t0 * w_theta0

    # evanescent part
    theta1, w_theta1 = uniform_dist(Nq_evan)
    theta1  = 0.25 * np.pi * (theta1 + 1.0)
    w_theta1 *= 0.25 * np.pi

    sin_t1 = np.sin(theta1)
    cos_t1 = np.cos(theta1)

    kz1    = k0 / cos_t1
    kk1 = 1j * k0 * sin_t1 / cos_t1  
    jac   = k0 * sin_t1 / cos_t1**2 
    w1    = w_theta1 * jac                    

    # concatenate to size Nq_prop + Nq_evan
    kz = np.concatenate((kz0, kz1))
    kk = np.concatenate((kk0, kk1))
    w = np.concatenate((w0, w1))

    return kz, w, kk

def gradCylindricalToCartesian(gradient, r, phi, z, axis, origin, radial, normal):
    """
    Convert gradient from cylindrical to Cartesian coordinates.

    gradient - gradient in cylindrical coordinates of size (3, N) d/dr, 1/r*d/dphi, d/dz
    r, phi, z - cylindrical coordinates of size (N,) at evaluation points
    axis - cylinder axis vector of size (3,)
    origin - point on the cylinder axis of size (3,)
    radial - radial direction vector of size (3,)

    returns:
    grad_cart - gradient in Cartesian coordinates of size (3, N)
    """
    # Normalize axis and radial vectors
    axis = axis / np.linalg.norm(axis)
    radial = radial / np.linalg.norm(radial)
    normal = normal / np.linalg.norm(normal)

    # reshape phi for broadcasting
    phi_ = phi[None, None, None, :]   # (1,1,1,Ny)

    e_r = (
        np.cos(phi_) * radial[:, None, None, None]
    + np.sin(phi_) * normal[:, None, None, None]
    )

    e_phi = (
    -np.sin(phi_) * radial[:, None, None, None]
    + np.cos(phi_) * normal[:, None, None, None]
    )

    e_z = axis[:, None, None, None]

    grad_cart = (
    e_r   * gradient[0]
    + e_phi * gradient[1]   # assuming term is already (1/r) d/dphi
    + e_z   * gradient[2]
    ) # 3, Nk, Nx, Ny

    return grad_cart

def gradPolarToCartesian(gradient, r, phi, origin, radial, normal):
    """
    Convert gradient from cylindrical to Cartesian coordinates.

    gradient - gradient in cylindrical coordinates of size (2, N) d/dr, 1/r*d/dphi,
    r, phi - polaar coordinates of size (N,) at evaluation points
    origin - point on the cylinder axis of size (2,)
    radial - radial direction vector of size (2,)
    normal - phi direction vector of size (2, )

    returns:
    grad_cart - gradient in Cartesian coordinates of size (2, N): dG/dx, dG/dy
    """
    # Normalize axis and radial vectors
    axis = axis / np.linalg.norm(axis)
    radial = radial / np.linalg.norm(radial)
    normal = normal / np.linalg.norm(normal)

    # reshape phi for broadcasting
    phi_ = phi[None, None, None, :]   # (1,1,1,Ny)

    e_r = (
        np.cos(phi_) * radial[:, None, None, None]
    + np.sin(phi_) * normal[:, None, None, None]
    )

    e_phi = (
    -np.sin(phi_) * radial[:, None, None, None]
    + np.cos(phi_) * normal[:, None, None, None]
    )

    grad_cart = (
    e_r   * gradient[0]
    + e_phi * gradient[1]   # assuming term is (1/r) d/dphi
    ) # 3, Nk, Nx, Ny

    return grad_cart

def beta_safe(m_qm, x):
    # TODO: make safe(r)?
    Jm0 = jv(m_qm - 1, x)
    Jm1 = jv(m_qm + 1, x) 
    Hm0 = hankel1(m_qm - 1, x)
    Hm1 = hankel1(m_qm + 1, x)

    beta = (Jm0 - Jm1) / (Hm0 - Hm1)             # (q, m)

    return beta

def uniform_dist(N, eps=1e-6):
    """
    uniformly spaced datapoints in the range [1-, 1] with weight density 1
    """

    dx = 2 / N # spacing
    points = np.arange(-1, 1, dx)+eps # points dx/2 - 1, ..., 1-dx/2
    weights = np.ones(N)/N * (1-(-1))

    return points, weights

def arcsin_dist(N):

    x = np.linspace(0, 1, N+1, endpoint=True) # segment edges
    t = np.arcsin(x) * 4 / np.pi - 1 # transform to clump around -1
    t_c = (t[1:]+t[:-1])/2 # segment centers
    w = np.diff(t) # weights prop to length segment, should sum to 2 = |[-1, 1]|
     
    return t_c, w

def arccos_inv_dist(N, xmax = 10):

    x = np.linspace(1, xmax, N+1, endpoint=True) # segment edges
    t = np.arccos(1/x) * 4 / np.pi - 1 # transform to clump around -1
    t_c = (t[1:]+t[:-1])/2 # segment centers
    w = np.diff(t) # weights prop to length segment, should sum to 2 = |[-1, 1]|
     
    return t_c, w

class CylinderGreen(TailoredGreen):
    """
    Class for computing and plotting the Tailored Green's function
    for a cylinder scatterer.
    """
    def __init__(self, radius:float, axis:np.ndarray, origin:np.ndarray, radial:np.ndarray=None, dim=3,
                 numerics={
                    'mmax': 32,
                    'Nq_prop': 128,
                    'Nq_evan': 128,
                    'eps_radius':1e-24
                 }
                 
                 ):
        super().__init__(dim=dim)
        self.radius = radius
        self._numerics = numerics
        self.origin = origin # a point on the cylinder axis, taken as the origin of the cylindrical coordinate system

        if dim==3:
            self.axis = axis / np.linalg.norm(axis) # axis vector of the cylinder

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

            self.normal = np.cross(self.axis, self.radial) # normal vector of the cylinder, completing the right-handed coordinate system

        elif dim==2:
            self.axis=0.0 # should not be used
            if radial is None:
                # choose an arbitrary radial direction perpendicular to the axis
                if np.allclose(self.axis, np.array([1,0])):
                    temp_vec = np.array([0,1])
                else:
                    temp_vec = np.array([1,0])
                self.radial = temp_vec - np.dot(temp_vec, self.axis) * self.axis
                self.radial /= np.linalg.norm(self.radial)
            else:
                self.radial = radial / np.linalg.norm(radial) # radial vector of the cylinder, taken as the zero azimuth direction

            self.normal = np.stack([-self.radial[1], self.radial[0]]) # cross product in 2D between [0,0,1] and self.radial!
        else:
            raise ValueError(f'dimension {self.dim} not implemented!')
        
        self.G_cyl_base_2D = CylinderGreen2D
        self.G_grad_cyl_base_2D = CylinderGreenGradient2D

    def getScatteringGreen(self, x, y, k):
        mmax  = self._numerics.get("mmax",   32)
        eps_radius = self._numerics.get("eps_radius", 1e-24)   
        Nq_prop  = self._numerics.get("Nq_prop",   128)
        Nq_evan  = self._numerics.get("Nq_evan",   128)

        k = np.atleast_1d(k)

        if self.dim == 2: # in 2D, use the functions directly!
            obs_r, obs_phi = getPolarCoordinates(
            x, self.origin, self.radial, self.normal
            )
            src_r, src_phi= getPolarCoordinates(
            y, self.origin, self.radial, self.normal
            )

            return self.G_cyl_base_2D([obs_r, obs_phi], [src_r, src_phi], k, self.radius, mmax=mmax, eps_radius=eps_radius)

        elif self.dim == 3: # in 3D, integrate over the extrusion direction!
            obs_r, obs_phi, obs_x = getCylindricalCoordinates(
            x, self.axis, self.origin, self.radial, self.normal
            )
            src_r, src_phi, src_x = getCylindricalCoordinates(
                y, self.axis, self.origin, self.radial, self.normal
            )

            Nk = k.size
            Nx = x.shape[1]
            Ny = y.shape[1]

            G = np.zeros((Nk, Nx, Ny), dtype=np.complex128)


            for ik, k0 in enumerate(k): # integration loop to save on inst. memory requirement...
                print(f"[k-loop] {ik+1}/{Nk}  k={k0:.4e}")
                kz, w, kk = get_quadrature_in_kz(k0, Nq_prop=Nq_prop, Nq_evan=Nq_evan)

                for kzval, wval, kkval in zip(kz, w, kk): # second loop, necessary for large arrays....
                    
                    G_2D = self.G_cyl_base_2D([obs_r, obs_phi], [src_r, src_phi], kkval, self.radius, mmax=mmax, eps_radius=eps_radius) # shape 1, Nx, Ny
                    cosz = np.cos(kzval * (obs_x[:, None] -src_x[None, :])) # Nx, Ny

                    G[ik, :, :] += G_2D[0, :, :] * cosz * wval # increment
            
            G *= 1/np.pi # rescale the integral (see appendix on cylinder green's function)

            return G
    
    def getScatteringGreenGradient(self, x, y, k):
        mmax  = self._numerics.get("mmax",   32)
        eps_radius = self._numerics.get("eps_radius", 1e-24)   
        Nq_prop  = self._numerics.get("Nq_prop",   128)
        Nq_evan  = self._numerics.get("Nq_evan",   128)

        k = np.atleast_1d(k)


        if self.dim == 2: # in 2D, use the functions directly!

            obs_r, obs_phi = getPolarCoordinates(
            x, self.origin, self.radial, self.normal
            )
            src_r, src_phi= getPolarCoordinates(
            y, self.origin, self.radial, self.normal
            )

            gradG_polar = self.G_grad_cyl_base_2D([obs_r, obs_phi], [src_r, src_phi], k, self.radius, mmax=mmax, eps_radius=eps_radius, return_G=False) # shape 2, Nk, Nx, Ny
            gradG_cart = gradPolarToCartesian(
                gradG_cylindrical,
                src_r,
                src_phi,
                src_x,
                self.origin,
                self.radial,
                self.normal
            )

            return gradG_cart

        elif self.dim == 3: # in 3D, integrate over the extrusion direction!

            obs_r, obs_phi, obs_x = getCylindricalCoordinates(
            x, self.axis, self.origin, self.radial, self.normal
            )
            src_r, src_phi, src_x = getCylindricalCoordinates(
            y, self.axis, self.origin, self.radial, self.normal
            )

            Nk = k.size
            Nx = x.shape[1]
            Ny = y.shape[1]

            gradG_cylindrical = np.zeros((3, Nk, Nx, Ny), dtype=np.complex128)


            for ik, k0 in enumerate(k): # integration loop to save on inst. memory requirement...
                print(f"[k-loop] {ik+1}/{Nk}  k={k0:.4e}")
                kz, w, kk = get_quadrature_in_kz(k0, Nq_prop=Nq_prop, Nq_evan=Nq_evan)

                for kzval, wval, kkval in zip(kz, w, kk): # second loop, necessary for large arrays....
                    
                    grad_2D_polar, G_2D = self.G_grad_cyl_base_2D([obs_r, obs_phi], [src_r, src_phi], kkval, self.radius,
                                                                   mmax=mmax, eps_radius=eps_radius, return_G=True) # shapes 2, 1, Nx, Ny and 1, Nx, Ny
                    cosz = np.cos(kzval * (obs_x[:, None] - src_x[None, :])) # Nx, Ny
                    dcosz_dz = kzval * np.sin(kzval * (obs_x[:, None] - src_x[None, :]))

                    gradG_cylindrical[0, ik, :, :] += grad_2D_polar[0, 0, :, :] * cosz * wval # dG/dr - only integrating the 2D gradient
                    gradG_cylindrical[1, ik, :, :] += grad_2D_polar[1, 0, :, :] * cosz * wval # 1/r dG/dphi - same as above
                    gradG_cylindrical[2, ik, :, :] += G_2D[0, :, :] * dcosz_dz * wval # dG/dz -  actually computing the z gradient based on G_2D
            
            gradG_cylindrical *= 1/np.pi # rescale the integral (see appendix on cylinder green's function)

            gradG_cart = gradCylindricalToCartesian(
                gradG_cylindrical,
                src_r,
                src_phi,
                src_x,
                self.axis,
                self.origin,
                self.radial,
                self.normal
            )

            return gradG_cart
    
    def plot3Ddirectivity(
        self, k, y, R=None, Nphi=36, Ntheta=18,
        blending=0.1,
    valmin = None, valmax=None):
        super().plot3Ddirectivity(
            k, y, R=R, Nphi=Nphi, Ntheta=Ntheta,
            extra_script=self.plotAxis,
            blending=blending,
            valmin = valmin, valmax=valmax
        )

    def plotSelf(self, fig, ax):
        """
        Plot a cylindrical outline for reference.
        """
        axis = self.axis
        e0 = self.radial 
        e1 = self.normal   # completes right-handed basis
        R = self.radius
        phi = np.linspace(0, 2*np.pi, 36)

        if self.dim==3:



            zmin, zmax = ax.get_xlim()
            # two end caps along axis (for outline only)
            # zmin, zmax = -1.0, 1.0   # purely visual extent

            # angular parameter



            for z in np.linspace(zmin, zmax, 11):
                ring = (
                    R * np.cos(phi)[:, None] * e0[None, :]
                + R * np.sin(phi)[:, None] * e1[None, :]
                + z * axis[None, :]
                ) + self.origin[None, :]

                ax.plot(
                    ring[:, 0],
                    ring[:, 1],
                    ring[:, 2],
                    color="k",
                    linewidth=3.0,
                    alpha=1.0
                )

            # draw axis line
            axis_line = np.array([zmin * axis, zmax * axis])
            ax.plot(
                axis_line[:, 0] + self.origin[0],
                axis_line[:, 1] + self.origin[1],
                axis_line[:, 2] + self.origin[2],
                color="k",
                linestyle="--",
                linewidth=3.0,
                alpha=1.0
            )

            ax.set_box_aspect([1, 1, 1])

        elif self.dim==2:
            ring = (
                R * np.cos(phi)[:, None] * e0[None, :]
            + R * np.sin(phi)[:, None] * e1[None, :]
            ) + self.origin[None, :]

            ax.plot(
                ring[:, 0],
                ring[:, 1],
                color="k",
                linewidth=3.0,
                alpha=1.0
            )

        return fig, ax

    def plotAxis(self, fig, ax):

        axis = self.axis / np.linalg.norm(self.axis)
        e0 = self.radial / np.linalg.norm(self.radial)
        e1 = np.cross(axis, e0)   # completes right-handed basis

        R = self.radius

        zmin, zmax = ax.get_xlim()
        # two end caps along axis (for outline only)
        # zmin, zmax = -1.0, 1.0   # purely visual extent

        # angular parameter
        phi = np.linspace(0, 2*np.pi, 36)

        # draw axis line
        # axis_line = np.array([zmin * axis, zmax * axis])
        axis_line = np.array([-3 * self.radius * axis, 3* self.radius* axis])

        ax.plot(
            axis_line[:, 0] + self.origin[0],
            axis_line[:, 1] + self.origin[1],
            axis_line[:, 2] + self.origin[2],
            color="k",
            linestyle="--",
            linewidth=3.0,
            alpha=1.0
        )

        ax.set_box_aspect([1, 1, 1])

        return fig, ax