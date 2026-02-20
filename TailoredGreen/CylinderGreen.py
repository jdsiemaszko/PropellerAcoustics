from matplotlib import axis
from .TailoredGreen import TailoredGreen
import numpy as np
from scipy.special import hankel1, jv, kv, h1vp, hankel1e, jvp
from numpy.polynomial.legendre import leggauss
from Constants.const import PREF
from Constants.helpers import p_to_SPL, getCylindricalCoordinates
import matplotlib.pyplot as plt


def gradCylindricalToCartesian(gradient, r, phi, z, axis, origin, radial):
    """
    Convert gradient from cylindrical to Cartesian coordinates.

    gradient - gradient in cylindrical coordinates of size (3, N)
    r, phi, z - cylindrical coordinates of size (N,)
    axis - cylinder axis vector of size (3,)
    origin - point on the cylinder axis of size (3,)
    radial - radial direction vector of size (3,)

    returns:
    grad_cart - gradient in Cartesian coordinates of size (3, N)
    """
    # Normalize axis and radial vectors
    axis = axis / np.linalg.norm(axis)
    radial = radial / np.linalg.norm(radial)
    
    # Compute the normal vector
    normal = np.cross(axis, radial)

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
    )

    return grad_cart


def beta_safe(m, x):
    """
    Safely compute beta = (J_{m-1} - J_{m+1}) / (H_{m-1} - H_{m+1})
    
    Parameters
    ----------
    m : int, float, or array_like
        Order(s) of the Bessel functions
    x : float, complex, or array_like
        Argument(s) of the Bessel functions
        
    Returns
    -------
    beta : complex or ndarray
        The ratio (J_{m-1} - J_{m+1}) / (H_{m-1} - H_{m+1})
        Shape is broadcast shape of m and x
        
    Examples
    --------
    >>> beta_safe(2, 5.0)  # Single values
    >>> beta_safe([1, 2, 3], 5.0)  # Multiple orders, single x
    >>> beta_safe(2, [1.0, 5.0, 10.0])  # Single order, multiple x
    >>> beta_safe([1, 2], [5.0, 10.0])  # Paired values
    >>> beta_safe([[1, 2], [3, 4]], [5.0, 10.0])  # Broadcasting
    """
    # Convert inputs to arrays
    m = np.asarray(m)
    x = np.asarray(x)
    
    # Store original shapes for output
    m_scalar = m.ndim == 0
    x_scalar = x.ndim == 0
    
    # Ensure at least 1D for processing
    m = np.atleast_1d(m)
    x = np.atleast_1d(x)
    
    # Broadcast to common shape
    try:
        m_bc, x_bc = np.broadcast_arrays(m, x)
    except ValueError as e:
        raise ValueError(f"Cannot broadcast m with shape {m.shape} and x with shape {x.shape}") from e
    
    # Flatten for iteration
    m_flat = m_bc.ravel()
    x_flat = x_bc.ravel()
    
    # Initialize output
    beta = np.zeros(m_flat.shape, dtype=complex)
    
    # Process each (m, x) pair
    for i, (mi, xi) in enumerate(zip(m_flat, x_flat)):
        beta[i] = _beta_single(mi, xi)
    
    # Reshape to broadcast shape
    beta = beta.reshape(m_bc.shape)
    
    # Return scalar if both inputs were scalar
    if m_scalar and x_scalar:
        return beta.item()
    
    return beta

def _beta_single(m, x):
    """
    Compute beta for a single (m, x) pair.
    
    Parameters
    ----------
    m : float
        Order of the Bessel functions
    x : float or complex
        Argument of the Bessel functions
        
    Returns
    -------
    beta : complex
        The ratio
    """
    # Handle edge case: x ≈ 0
    if np.abs(x) < 1e-10:
        if m == 0:
            return 0.0 + 0.0j
        else:
            return 0.0 + 0.0j
    
    abs_x = np.abs(x)
    
    # Strategy depends on argument magnitude
    if abs_x < 50:
        # Direct computation for moderate arguments
        return _beta_direct(m, x)
    else:
        # Asymptotic approach for large arguments
        return _beta_asymptotic(m, x)

def _beta_direct(m, x):
    """
    Direct computation of beta using Bessel functions.
    """
    try:
        Jm_minus = jv(m - 1, x)
        Jm_plus = jv(m + 1, x)
        Hm_minus = hankel1(m - 1, x)
        Hm_plus = hankel1(m + 1, x)
        
        numerator = Jm_minus - Jm_plus
        denominator = Hm_minus - Hm_plus
        
        # Check for numerical issues in denominator
        if np.abs(denominator) < 1e-100:
            # Use derivative formulation instead
            # f_{m-1} - f_{m+1} = 2*f'_m
            try:
                numerator = 2 * jvp(m, x, 1)
                denominator = 2 * h1vp(m, x, 1)
            except:
                # If derivatives fail, fall back to asymptotic
                return _beta_asymptotic(m, x)
        
        # Check for overflow/underflow
        if not np.isfinite(numerator) or not np.isfinite(denominator):
            return _beta_asymptotic(m, x)
        
        result = numerator / denominator
        
        # Sanity check
        if not np.isfinite(result):
            return _beta_asymptotic(m, x)
        
        return result
        
    except (RuntimeWarning, FloatingPointError, OverflowError):
        return _beta_asymptotic(m, x)

def _beta_asymptotic(m, x):
    """
    Asymptotic formula for beta when |x| is large.
    """
    try:
        # Use derivative formulation for better numerical stability
        Jm_deriv = jvp(m, x, 1)
        Hm_deriv = h1vp(m, x, 1)
        
        if np.abs(Hm_deriv) > 1e-100 and np.isfinite(Jm_deriv) and np.isfinite(Hm_deriv):
            result = Jm_deriv / Hm_deriv
            if np.isfinite(result):
                return result
        
        # Fallback: compute ratio of main functions
        Jm = jv(m, x)
        Hm = hankel1(m, x)
        
        if np.abs(Hm) > 1e-100 and np.isfinite(Jm) and np.isfinite(Hm):
            result = Jm / Hm
            if np.isfinite(result):
                return result
        
        # Ultimate fallback for very large x
        # As x -> infinity, beta approaches exp(-2i*x) for real x
        if np.isreal(x) and x > 0:
            return np.exp(-2j * x)
        else:
            return 1.0 + 0.0j
            
    except:
        return 1.0 + 0.0j

# Vectorized version using scipy's vectorization
def beta_safe_vectorized(m, x):
    """
    Fully vectorized version using scipy's internal vectorization.
    Generally faster but may be less robust for edge cases.
    
    Parameters
    ----------
    m : int, float, or array_like
        Order(s) of the Bessel functions
    x : float, complex, or array_like
        Argument(s) of the Bessel functions
        
    Returns
    -------
    beta : complex or ndarray
        The ratio (J_{m-1} - J_{m+1}) / (H_{m-1} - H_{m+1})
    """
    m = np.asarray(m)
    x = np.asarray(x)
    
    # Store if inputs were scalar
    return_scalar = (m.ndim == 0 and x.ndim == 0)
    
    # Broadcast
    m, x = np.broadcast_arrays(np.atleast_1d(m), np.atleast_1d(x))
    
    # Compute all Bessel functions (vectorized)
    Jm_minus = jv(m - 1, x)
    Jm_plus = jv(m + 1, x)
    Hm_minus = hankel1(m - 1, x)
    Hm_plus = hankel1(m + 1, x)
    
    numerator = Jm_minus - Jm_plus
    denominator = Hm_minus - Hm_plus
    
    # Handle small denominators
    small_denom = np.abs(denominator) < 1e-100
    
    if np.any(small_denom):
        # Use derivatives where needed
        Jm_deriv = 2 * jvp(m, x, 1)
        Hm_deriv = 2 * h1vp(m, x, 1)
        
        numerator = np.where(small_denom, Jm_deriv, numerator)
        denominator = np.where(small_denom, Hm_deriv, denominator)
    
    # Regularize remaining small denominators
    epsilon = np.finfo(complex).eps
    denominator_safe = np.where(
        np.abs(denominator) < epsilon,
        epsilon * (1 + 1j),
        denominator
    )
    
    result = numerator / denominator_safe
    
    # Handle non-finite results
    if np.any(~np.isfinite(result)):
        # For non-finite values, try asymptotic approximation
        Jm = jv(m, x)
        Hm = hankel1(m, x)
        fallback = np.where(np.abs(Hm) > epsilon, Jm / Hm, 1.0 + 0.0j)
        result = np.where(np.isfinite(result), result, fallback)
    
    if return_scalar:
        return result.item()
    
    return result
class CylinderGreen(TailoredGreen):
    """
    Class for computing and plotting the Tailored Green's function
    for a cylinder scatterer.
    """
    def __init__(self, radius:float, axis:np.ndarray, origin:np.ndarray, radial:np.ndarray=None, dim=3,
                 numerics={
                    'nmax': 16,
                    'Nq_prop': 100,
                    'maxKmultiple': 5
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

        self.normal = np.cross(self.axis, self.radial) # normal vector of the cylinder, completing the right-handed coordinate system

    def conditionTerms(self, newterm, eps=1e12, printBool=False):
        # condition terms to avoid NaNs and Infs
        newterm = np.where(
            np.isnan(newterm) | np.isinf(newterm),
            0.0,
            newterm
        )
        newterm = np.where(
            np.abs(newterm) > eps,
            0.0,
            newterm
        )
        if printBool:
            print(f'ignoring a total of {np.sum(np.isnan(newterm)) + np.sum(np.isinf(newterm)) + np.sum(np.abs(newterm) > eps)} problematic terms out of {newterm.size}' )
        return newterm

    def getScatteringGreenMemory(self, x, y, k):

        mmax = self._numerics.get("nmax", int(k.max() * self.radius) + 10)
        Nq_prop = self._numerics.get("Nq_prop", 100)
        eps_k = self._numerics.get("eps_k", 1e-6)
        eps_radius = self._numerics.get("eps_radius", 1e-3)
        # maxKmultiple = self._numerics.get("maxKmultiple", 5)

        if self.dim != 3:
            raise NotImplementedError

        k = np.atleast_1d(k)
        Nk = k.size
        Nx = x.shape[1]
        Ny = y.shape[1]

        # obs_r, obs_theta, obs_phi = getSphericalCoordinatesWRTx(x)
        # src_r, src_theta, src_phi = getSphericalCoordinatesWRTx(y)

        obs_r, obs_phi, obs_x = getCylindricalCoordinates(x, self.axis, self.origin, self.radial, self.normal)
        src_r, src_phi, src_x = getCylindricalCoordinates(y, self.axis, self.origin, self.radial, self.normal)

        dz   = obs_x[:, None] - src_x[None, :]          # (Nx, Ny), Note: dz is along the cylinder axis!
        dphi = obs_phi[:, None] - src_phi[None, :] % (np.pi)    # (Nx, Ny)

        mask_radius = obs_r < self.radius   # (Nx)

        G = np.zeros((Nk, Nx, Ny), dtype=np.complex128)

        # ---- m-dependent quantities ----
        m = np.arange(0, mmax, 1)    # (m,)  
        epsm =  np.ones(mmax)
        epsm[1:] = 2                 

        # angular dependence (m, Nx, Ny)
        cos1 = np.cos(m[:, None, None] * dphi[None, :, :])

         # ---- Gauss–Legendre nodes on [0, pi/2] ----
        theta, w_theta = leggauss(Nq_prop)
        theta = 0.25 * np.pi * (theta + 1.0)
        w_theta *= 0.25 * np.pi

        sin_t = np.sin(theta)
        cos_t = np.cos(theta)

        # r_< and r_>
        rmin = np.minimum(obs_r[:, None], src_r[None, :])
        rmax = np.maximum(obs_r[:, None], src_r[None, :])


        for ik, k0 in enumerate(k):
            # corr = np.ones((Nq_prop, mmax, Nx, Ny), dtype=np.complex128) # expensive?
            kz = k0 * cos_t                # (q,)
            kk = k0 * sin_t                # (q,)
            w  = k0 * sin_t * w_theta      # Jacobian, (q,)

            kk_eps = eps_k * k0
            small = kk < kk_eps

            

            cos0 = np.cos(kz[:, None, None] * dz)

            kk_qm = kk[:, None]
            m_qm = m[None, :]

            # Jm0 = jv(m_qm , kk_qm * a)
            # Hm0 = hankel1(m_qm, kk_qm * a)
            # beta = Jm0 / Hm0

            # Jm0 = jv(m_qm - 1, kk_qm * self.radius) * (m_qm > 1)
            # Jm1 = jv(m_qm + 1, kk_qm * self.radius) 
            # Hm0 = hankel1(m_qm - 1, kk_qm * self.radius) * (m_qm > 1)
            # Hm1 = hankel1(m_qm + 1, kk_qm * self.radius)

            # beta = (Jm0 - Jm1) / (Hm0 - Hm1)             # (q, m)
            beta = beta_safe(m_qm, kk_qm * self.radius)

            # ---- Hankel terms ----
            H_obs = hankel1(
                m_qm[:, :, None],
                kk_qm[:, :, None] * obs_r[None, None, :]        
            ) * (obs_r[None, None, :] > self.radius * (1+eps_radius))                                                    # (q, m, Nx)

            H_src = hankel1(
                m_qm[:, :, None],
                kk_qm[:, :, None] * src_r[None, None, :]
            )      # (q, m, Ny)


            ##### small kk corrections
            # ignore actual computations
            # H_obs[small, :, :]  = 1.0
            # H_src[small, :, :] = 1.0
            # beta[small, :] = 1.0

            # # replace by a correction term
            # corr[small, :, :, :] = (
            #     -(1.0 / np.pi**2)
            #     * (rmin / rmax)[None, None, :, :]**m_qm[:, :, None, None]
            # )
            # corr[small, 0, :, :] = (
            #     -(2.0 / np.pi**2)
            #     * np.log(rmax[None, None, :, :] / self.radius)
            # )

            # remove all ugly terms

            a = beta[:, :, None, None] * H_obs[:, :, :, None] * H_src[:, :, None, :]
            # a = self.conditionTerms(a, eps=1e12, printBool=True)

            newterm = np.sum(
                w[:, None, None, None]
                # * beta[:, :, None, None]
                # * H_obs[:, :, :, None]
                # * H_src[:, :, None, :]
                * a
                * cos0[:, None, :, :]
                * cos1[None, :, :, :]
                * epsm[None, :, None, None]
                # * corr[:, :, :, :],
                ,
                axis=(0, 1)  # sum over kk (axis 0) and m (axis 1)
            )
            newterm[mask_radius, :] = 0.0

            G[ik] = newterm

        G *= (-1j / (4 * np.pi))

        return G

    def getScatteringGreen(self, x, y, k):

        mmax = self._numerics.get("nmax", int(k.max() * self.radius) + 10)
        Nq_prop = self._numerics.get("Nq_prop", 100)
        eps_radius = self._numerics.get("eps_radius", 1e-3)

        if self.dim != 3:
            raise NotImplementedError

        k = np.atleast_1d(k)
        Nk = k.size
        Nx = x.shape[1]
        Ny = y.shape[1]

        obs_r, obs_phi, obs_x = getCylindricalCoordinates(
            x, self.axis, self.origin, self.radial, self.normal
        )
        src_r, src_phi, src_x = getCylindricalCoordinates(
            y, self.axis, self.origin, self.radial, self.normal
        )

        G = np.zeros((Nk, Nx, Ny), dtype=np.complex128)

        m = np.arange(0, mmax)
        epsm = np.ones(mmax)
        epsm[1:] = 2

        theta, w_theta = leggauss(Nq_prop)
        theta = 0.25 * np.pi * (theta + 1.0)
        w_theta *= 0.25 * np.pi

        sin_t = np.sin(theta)
        cos_t = np.cos(theta)

        for ik, k0 in enumerate(k):

            print(f"[k-loop] {ik+1}/{Nk}  k={k0:.4e}")

            kz = k0 * cos_t
            kk = k0 * sin_t
            w  = k0 * sin_t * w_theta

            kk_qm = kk[:, None]
            m_qm  = m[None, :]

            # --- beta ---
            # Hm_minus = hankel1(m_qm - 1, kk_qm * self.radius)
            # Hm_plus  = hankel1(m_qm + 1, kk_qm * self.radius)
            # Jm_minus = jv(m_qm - 1, kk_qm * self.radius)
            # Jm_plus  = jv(m_qm + 1, kk_qm * self.radius)

            # beta = (Jm_minus - Jm_plus) / (Hm_minus - Hm_plus)    

            beta = beta_safe(m_qm, kk_qm * self.radius)


            # --- SOURCE HANKELS (q,m,Ny) ---
            H_src = hankel1(
                m_qm[:, :, None],
                kk_qm[:, :, None] * src_r[None, None, :]
            )

            for ix in range(Nx):

                print(f"    observer {ix+1}/{Nx}")

                if obs_r[ix] < self.radius:
                    continue

                dz   = obs_x[ix] - src_x
                dphi = obs_phi[ix] - src_phi

                cos1 = np.cos(m[:, None] * dphi[None, :])
                cos0 = np.cos(kz[:, None] * dz[None, :])

                H_obs = hankel1(
                    m_qm[:, :, None],
                    kk_qm[:, :, None] * obs_r[ix]
                ) * (obs_r[ix] > self.radius * (1+eps_radius)) # ignore the part inside the cylinder!

                A = beta[:, :, None] * H_obs * H_src

                newterm = np.einsum(
                    "q,qmn,qn,mn,m->n",
                    w,
                    A,
                    cos0,
                    cos1,
                    epsm,
                    optimize=True
                )

                G[ik, ix, :] += newterm

        G *= (-1j / (4 * np.pi))
        return G

    def getScatteringGreenGradientMemory(self, x, y, k,):

        #  mmax=16, Nq_prop=100, maxKmultiple=5
        
        mmax = self._numerics.get("nmax", int(k.max() * self.radius) + 10)
        Nq_prop = self._numerics.get("Nq_prop", 100)
        eps_k = self._numerics.get("eps_k", 1e-6)
        eps_radius = self._numerics.get("eps_radius", 1e-3)

        if self.dim != 3:
            raise NotImplementedError

        k = np.atleast_1d(k)
        Nk = k.size
        Nx = x.shape[1]
        Ny = y.shape[1]

        # obs_r, obs_theta, obs_phi = getSphericalCoordinatesWRTx(x)
        # src_r, src_theta, src_phi = getSphericalCoordinatesWRTx(y)

        obs_r, obs_phi, obs_x = getCylindricalCoordinates(x, self.axis, self.origin, self.radial, self.normal)
        src_r, src_phi, src_x = getCylindricalCoordinates(y, self.axis, self.origin, self.radial, self.normal)

        dz   = obs_x[:, None] - src_x[None, :]          # (Nx, Ny), Note: dz is along the cylinder axis!
        dphi = obs_phi[:, None] - src_phi[None, :] % (np.pi)    # (Nx, Ny)

        gradG = np.zeros((3, Nk, Nx, Ny), dtype=np.complex128)

        # ---- m-dependent quantities ----
        m = np.arange(0, mmax, 1)    # (m,)  
        epsm =  np.ones(mmax)
        epsm[1:] = 2                 

        # angular dependence (m, Nx, Ny)
        cos1 = np.cos(m[:, None, None] * dphi[None, :, :])
        
        # derivative w.r.t. phi
        dcos1_dphi = m[:, None, None] * np.sin(m[:, None, None] * dphi[None, :, :])

         # ---- Gauss–Legendre nodes on [0, pi/2] ----
        theta, w_theta = leggauss(Nq_prop)
        theta = 0.25 * np.pi * (theta + 1.0)
        w_theta *= 0.25 * np.pi

        sin_t = np.sin(theta)
        cos_t = np.cos(theta)

        for ik, k0 in enumerate(k):

            # corr = np.ones((Nq_prop, mmax, Nx, Ny), dtype=np.complex128) # expensive?
            kz = k0 * cos_t                # (q,)
            kk = k0 * sin_t                # (q,)
            w  = k0 * sin_t * w_theta      # Jacobian

            # base term
            cos0 = np.cos(kz[:, None, None] * dz)      # (q, Nx, Ny)

            # z derivative
            dcos0_dz = kz[:, None, None] * np.sin(kz[:, None, None] * dz)  # (q, Nx, Ny)


            # ---- beta(q, m) ----
            kk_qm = kk[:, None]
            m_qm = m[None, :]

            # Jm0 = jv(m_qm , kk_qm * a)
            # Hm0 = hankel1(m_qm, kk_qm * a)
            # beta = Jm0 / Hm0

            # Jm0 = jv(m_qm - 1, kk_qm * self.radius) * (m_qm > 1)
            # Jm1 = jv(m_qm + 1, kk_qm * self.radius) 
            # Hm0 = hankel1(m_qm - 1, kk_qm * self.radius) * (m_qm > 1)
            # Hm1 = hankel1(m_qm + 1, kk_qm * self.radius)

            # beta = (Jm0 - Jm1) / (Hm0 - Hm1)             # (q, m)

            beta = beta_safe(m_qm, kk_qm * self.radius)


            # ---- Hankel terms ----
            H_obs = hankel1(
                m_qm[:, :, None],
                kk_qm[:, :, None] * obs_r[None, None, :]
            )  * (obs_r[None, None, :] > self.radius * (1+eps_radius))    # (q, m, Nx)

            # base term
            H_src = hankel1(
                m_qm[:, :, None],
                kk_qm[:, :, None] * src_r[None, None, :]
            )        

            # dr
            dH_src_dr = -kk_qm[:, :, None] * h1vp(
                m_qm[:, :, None],
                kk_qm[:, :, None] * src_r[None, None, :],
                n=1 # first derivative!
            )


            a = beta[:, :, None, None] * H_obs[:, :, :, None] * dH_src_dr[:, :, None, :]
            # a = self.conditionTerms(a, eps=1e12, printBool=True)

            newterm_dr = np.sum(
                w[:, None, None, None]
                # * beta[:, :, None, None]
                # * H_obs[:, :, :, None]
                # # * H_src[:, :, None, :]
                # * dH_src_dr[:, :, :, None]
                * a
                * cos0[:, None, :, :]
                * cos1[None, :, :, :]
                * epsm[None, :, None, None],
                axis=(0, 1)  # sum over kk (axis 0) and m (axis 1)
            )

            del a
            b = beta[:, :, None, None] * H_obs[:, :, :, None] * H_src[:, :, None, :]
            # b = self.conditionTerms(b, eps=1e12, printBool=True)

            # 1/r dG/dphi
            newterm_dphi = np.sum(
                w[:, None, None, None]
                # * beta[:, :, None, None]
                # * H_obs[:, :, :, None]
                # * H_src[:, :, None, :]
                * b
                * cos0[:, None, :, :]
                # * cos1[None, :, :, :]
                * dcos1_dphi[None, :, :, :]
                / src_r[None, None, None, :] # 1/r factor !
                * epsm[None, :, None, None],
                axis=(0, 1)  # sum over kk (axis 0) and m (axis 1)
            )

            newterm_dz = np.sum(
                w[:, None, None, None]
                # * beta[:, :, None, None]
                # * H_obs[:, :, :, None]
                # * H_src[:, :, None, :]
                * b
                # * cos0[:, None, :, :]
                * dcos0_dz[:, None, :, :]
                * cos1[None, :, :, :]
                * epsm[None, :, None, None],
                axis=(0, 1)  # sum over kk (axis 0) and m (axis 1)
            )
            del b

            # dG/dr
            gradG[0, ik] += newterm_dr
            # (1/r) dG/dphi
            gradG[1, ik] += newterm_dphi 
            # dG/dz
            gradG[2, ik] += newterm_dz

        gradG *= (-1j / (4 * np.pi))

        # transform to global cartesian coordinates, gradG = [dG/dx, dG/dy, dG/dz]
        gradG_cart = gradCylindricalToCartesian(gradG, src_r, src_phi, src_x, self.axis, self.origin, self.radial)
        return gradG_cart

    def getScatteringGreenGradient(self, x, y, k):

        mmax = self._numerics.get("nmax", int(k.max() * self.radius) + 10)
        Nq_prop = self._numerics.get("Nq_prop", 100)
        eps_radius = self._numerics.get("eps_radius", 1e-3)

        if self.dim != 3:
            raise NotImplementedError

        k = np.atleast_1d(k)
        Nk = k.size
        Nx = x.shape[1]
        Ny = y.shape[1]

        obs_r, obs_phi, obs_x = getCylindricalCoordinates(
            x, self.axis, self.origin, self.radial, self.normal
        )
        src_r, src_phi, src_x = getCylindricalCoordinates(
            y, self.axis, self.origin, self.radial, self.normal
        )

        gradG = np.zeros((3, Nk, Nx, Ny), dtype=np.complex128)

        m = np.arange(0, mmax)
        epsm = np.ones(mmax)
        epsm[1:] = 2

        theta, w_theta = leggauss(Nq_prop)
        theta = 0.25 * np.pi * (theta + 1.0)
        w_theta *= 0.25 * np.pi

        sin_t = np.sin(theta)
        cos_t = np.cos(theta)

        for ik, k0 in enumerate(k):

            print(f"[k-loop] {ik+1}/{Nk}  k={k0:.4e}")

            kz = k0 * cos_t
            kk = k0 * sin_t
            w = k0 * sin_t * w_theta

            kk_qm = kk[:, None]
            m_qm = m[None, :]

            # --- beta using recurrence Hankels ---
            # Hm_minus = hankel1(m_qm - 1, kk_qm * self.radius) # shape (k, m)
            # Hm_plus  = hankel1(m_qm + 1, kk_qm * self.radius)

            # Jm_minus = jv(m_qm - 1, kk_qm * self.radius)
            # Jm_plus  = jv(m_qm + 1, kk_qm * self.radius)

            # beta = (Jm_minus - Jm_plus) / (Hm_minus - Hm_plus)
            beta = beta_safe(m_qm, kk_qm * self.radius)


            # --- SOURCE HANKELS (precompute once) ---
            H_src = hankel1(
                m_qm[:, :, None],
                kk_qm[:, :, None] * src_r[None, None, :]
            )

            # derivative via recurrence
            H_src_minus = hankel1(
                m_qm[:, :, None] - 1,
                kk_qm[:, :, None] * src_r[None, None, :]
            )
            H_src_plus = hankel1(
                m_qm[:, :, None] + 1,
                kk_qm[:, :, None] * src_r[None, None, :]
            )

            dH_src_dr = -kk_qm[:, :, None] * 0.5 * (
                H_src_minus - H_src_plus
            )

            for ix in range(Nx):

                print(f"    observer {ix+1}/{Nx}")

                dz = obs_x[ix] - src_x
                dphi = obs_phi[ix] - src_phi

                cos1 = np.cos(m[:, None] * dphi[None, :])
                dcos1_dphi = m[:, None] * np.sin(m[:, None] * dphi[None, :])

                cos0 = np.cos(kz[:, None] * dz[None, :])
                dcos0_dz = kz[:, None] * np.sin(kz[:, None] * dz[None, :])

                H_obs = hankel1(
                    m_qm[:, :, None],
                    kk_qm[:, :, None] * obs_r[ix]
                ) * (obs_r[ix] > self.radius * (1 + eps_radius))

                A = beta[:, :, None] * H_obs * dH_src_dr
                B = beta[:, :, None] * H_obs * H_src

                newterm_dr = np.einsum(
                    "q,qmn,qn,mn,m->n",
                    w,
                    A,
                    cos0,
                    cos1,
                    epsm,
                    optimize=True
                )

                newterm_dphi = np.einsum(
                    "q,qmn,qn,mn,m->n",
                    w,
                    B,
                    cos0,
                    dcos1_dphi / src_r[None, :],
                    epsm,
                    optimize=True
                )

                newterm_dz = np.einsum(
                    "q,qmn,qn,mn,m->n",
                    w,
                    B,
                    dcos0_dz,
                    cos1,
                    epsm,
                    optimize=True
                )

                gradG[0, ik, ix, :] += newterm_dr
                gradG[1, ik, ix, :] += newterm_dphi
                gradG[2, ik, ix, :] += newterm_dz

        gradG *= (-1j / (4 * np.pi))

        gradG_cart = gradCylindricalToCartesian(
            gradG,
            src_r,
            src_phi,
            src_x,
            self.axis,
            self.origin,
            self.radial,
        )

        return gradG_cart


    # def plotFarFieldGradient(self, k, y, R=None, Nphi=36, Ntheta=18):
    #     super().plotFarFieldGradient(k, y, R=R, Nphi=Nphi, Ntheta=Ntheta)

    def plotDirectivity(
        self, k, y, R=None, Nphi=18, Ntheta=36,
        ref=PREF,
        extra_script=lambda self, fig, ax: None,
            blending=0.1,
    valmin = None, valmax=None):
        super().plotDirectivity(
            k, y, R=R, Nphi=Nphi, Ntheta=Ntheta,
            ref=ref,
            extra_script=self.plotAxis,
            blending=0.1,
            valmin = valmin, valmax=valmax
        )

    def plotSelf(self, fig, ax):
            """
            Plot a cylindrical outline for reference.
            """

            axis = self.axis / np.linalg.norm(self.axis)
            e0 = self.radial / np.linalg.norm(self.radial)
            e1 = np.cross(axis, e0)   # completes right-handed basis

            R = self.radius

            zmin, zmax = ax.get_xlim()
            # two end caps along axis (for outline only)
            # zmin, zmax = -1.0, 1.0   # purely visual extent

            # angular parameter
            phi = np.linspace(0, 2*np.pi, 36)



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


if __name__ == "__main__":
    axis = np.array([1.0, 0.0, 0.0])
    origin = np.array([0.0, 0.0, 0.0])
    # radial = np.array([0.0, 1.0, 0.0])

    # r, phi, z = getCylindricalCoordinates(points, axis, origin, radial,normal)
    # print(points[:, 0], points[:, 1], points[:, 2])
    # print(r, phi, z)
    # x = np.array([[2.0, 1.0, 1.0]]).T
    # cg.plotFarFieldGradient(k=np.array([10.0]), y=np.array([[0], [1.01], [0]]), R=0.501, Ntheta=2, Nphi = 36, Nr= 5, Rmax=1.0)
    # cg.plotFarFieldGradient(k=np.array([10.0]), y=np.array([[0], [1.0], [0]]), R=1.0, Ntheta=18, Nphi = 36, Nr= 5, Rmax=1.5)
    
    # for nmax  in [4, 8, 16]:
    #     print(f"nmax = {nmax}")
    #     cg = CylinderGreen(radius=0.5, axis=axis, origin=origin, dim=3, 
    #                        numerics={
    #                     'nmax': nmax,
    #                     'Nq_prop': 100,
    #                     'maxKmultiple': 5
    #                 })
    #     cg.plotScatteringYZ(y=np.array([[0.0], [1.0], [0]]), k=np.array([10.0]), rmin=0.5, rmax=10.0)


    # for kmax  in [5, 10, 25]:
    #     print(f"kmax = {kmax}")
    #     cg = CylinderGreen(radius=0.5, axis=axis, origin=origin, dim=3, 
    #                        numerics={
    #                     'nmax': 8,
    #                     'Nq_prop': 100,
    #                     'maxKmultiple': kmax
    #                 })
    #     # cg.plotScatteringYZ(y=np.array([[0.0], [1.0], [0]]), k=np.array([10.0]), rmin=0.5, rmax=10.0)
    #     cg.plotDirectivity(
    #         k=np.array([10.0]), y=np.array([[0], [1.0], [0]]), R=200.0, Nphi=18, Ntheta=36
    #     )
    cg = CylinderGreen(radius=0.5, axis=axis, origin=origin, dim=3, 
                           numerics={
                        'nmax': 55,
                        'Nq_prop': 64,
                        'eps_k' : 1e-6,
                    })
    # cg.plotScatteringYZ(y=np.array([[0.0], [0.0], [1.0]]), k=np.array([10.0]), rmin=0.5, rmax=200.0)
    cg.plotDirectivity(k=np.array([10.0]), y=np.array([[0], [0.0], [1.0]]), R=5.0, Nphi=18, Ntheta=36)