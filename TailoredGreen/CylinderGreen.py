from matplotlib import axis
from .TailoredGreen import TailoredGreen
import numpy as np
from scipy.special import hankel1, jv, iv, kv, h1vp, hankel1e, jvp
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
class CylinderGreen(TailoredGreen):
    """
    Class for computing and plotting the Tailored Green's function
    for a cylinder scatterer.
    """
    def __init__(self, radius:float, axis:np.ndarray, origin:np.ndarray, radial:np.ndarray=None, dim=3,
                 numerics={
                    'nmax': 32,
                    'Nq_prop': 128,
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

    def getScatteringGreenPropagating(self, x, y, k):

        mmax = self._numerics.get("nmax", int(k.max() * self.radius) + 10)
        Nq_prop = self._numerics.get("Nq_prop", 100)
        eps_radius = self._numerics.get("eps_radius", 1e-12)

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

    def getScatteringGreenEvanescent(self, x, y, k):
        """
        Evanescent (k_z > k) contribution to the cylinder scattering Green's function.

        For k_z > k the in-plane wavenumber is imaginary:
            kk = sqrt(k^2 - k_z^2)  →  purely imaginary
        We write  kk = i*kappa,  kappa = sqrt(k_z^2 - k^2) > 0.

        Bessel functions of imaginary argument:
            J_m(i*kappa*r)  =  i^m  I_m(kappa*r)
            H_m^(1)(i*kappa*r)  =  (2/pi) * (-i)^(m+1)  K_m(kappa*r)

        For the scattering problem (observer AND source outside the cylinder)
        only K_m appears in the physical solution, because K_m(kappa*r) → 0
        as r → ∞ while I_m diverges.

        The scattering coefficient becomes (stability: use modified Bessel ratio):
            beta_evan(m, kappa*a) = -I_m'(kappa*a) / K_m'(kappa*a)
                                = (I_{m-1} + I_{m+1}) / (K_{m-1} + K_{m+1})
        (derivative recurrences: I_m' = (I_{m-1}+I_{m+1})/2,
                                K_m' = -(K_{m-1}+K_{m+1})/2)

        Integration variable: k_z = k / sin(theta)  with theta in (0, pi/2)
        maps [k, ∞) onto a finite interval amenable to Gauss-Legendre quadrature.
        """

        mmax     = self._numerics.get("nmax",      int(k.max() * self.radius) + 10)
        Nq_evan  = self._numerics.get("Nq_evan",   100)
        eps_r    = self._numerics.get("eps_radius", 1e-12)

        if self.dim != 3:
            raise NotImplementedError

        k   = np.atleast_1d(k)
        Nk  = k.size
        Nx  = x.shape[1]
        Ny  = y.shape[1]

        obs_r, obs_phi, obs_x = getCylindricalCoordinates(
            x, self.axis, self.origin, self.radial, self.normal
        )
        src_r, src_phi, src_x = getCylindricalCoordinates(
            y, self.axis, self.origin, self.radial, self.normal
        )

        G = np.zeros((Nk, Nx, Ny), dtype=np.complex128)

        m    = np.arange(0, mmax)
        epsm = np.ones(mmax);  epsm[1:] = 2

        # Map k_z in [k, ∞) via  k_z = k / sin(t),  t in (0, pi/2).
        # dt  →  dk_z :  dk_z = -k cos(t)/sin^2(t) dt
        # Jacobian factor:  |dk_z/dt| = k cos(t)/sin^2(t)
        # After the change of variables the integral over t runs (0, pi/2).
        # We map Gauss-Legendre nodes from [-1,1] to (0, pi/2).
        t_gl, w_gl = leggauss(Nq_evan)
        t    = 0.25 * np.pi * (t_gl + 1.0)       # t in (0, pi/2)
        w    = w_gl * 0.25 * np.pi               # raw GL weights

        sin_t  = np.sin(t)
        cos_t  = np.cos(t)

        for ik, k0 in enumerate(k):

            print(f"[evan k-loop] {ik+1}/{Nk}  k={k0:.4e}")

            # k_z = k0/sin(t),   kappa = k_z * cos(t)/... wait, let's be explicit:
            #   k_z   = k0 / sin_t                          shape (Nq,)
            #   kappa = sqrt(k_z^2 - k0^2) = k0*cos_t/sin_t  shape (Nq,)
            kz    = k0 / sin_t                              # (Nq,)
            kappa = k0 * cos_t / sin_t                      # (Nq,)  always >= 0

            # Jacobian:  dk_z = k0 * cos_t / sin_t^2 * dt
            jac   = k0 * cos_t / sin_t**2                  # (Nq,)
            ww    = w * jac                                 # effective quadrature weights

            # We also need exp(-kz * |dz|) from the z-integral; that factor is
            # already encoded in the cos0 term below via the imaginary kz:
            #   cos(kz * dz)  with kz real  →  oscillatory (propagating)
            # For the evanescent part kz is real and > k, so cos(kz*dz) is still
            # real and oscillatory in z — the evanescence is purely radial (kappa).
            # The kernel is therefore:
            #   K_m(kappa * r_obs) * K_m(kappa * r_src) * cos(kz * dz) * cos(m*dphi)
            # with the beta coefficient ensuring the scattering BC at r=a.

            kappa_qm = kappa[:, None]           # (Nq, 1)  for broadcasting with m
            m_qm     = m[None, :]               # (1, mmax)

            # --- scattering coefficient for evanescent modes ---
            #   beta = [I_{m-1}(u) + I_{m+1}(u)] / [K_{m-1}(u) + K_{m+1}(u)]
            # where u = kappa * radius.
            u = kappa_qm * self.radius          # (Nq, mmax)

            Im1 = iv(m_qm - 1, u)              # I_{m-1}
            Ip1 = iv(m_qm + 1, u)              # I_{m+1}
            Km1 = kv(m_qm - 1, u)              # K_{m-1}
            Kp1 = kv(m_qm + 1, u)              # K_{m+1}

            denom = Km1 + Kp1
            # Guard against near-zero denominator (very large u → K explodes,
            # but ratio I/K → 0 exponentially, so set beta=0 there).
            safe  = np.abs(denom) > 1e-300
            beta  = np.where(safe, (Im1 + Ip1) / np.where(safe, denom, 1.0), 0.0)
            # (Nq, mmax)

            # --- source K_m values (Nq, mmax, Ny) ---
            K_src = kv(
                m_qm[:, :, None],
                kappa_qm[:, :, None] * src_r[None, None, :]
            )

            for ix in range(Nx):

                print(f"    observer {ix+1}/{Nx}")

                if obs_r[ix] < self.radius:
                    continue                    # observer inside cylinder — skip

                dz   = obs_x[ix]  - src_x      # (Ny,)
                dphi = obs_phi[ix] - src_phi    # (Ny,)

                # z-oscillation (kz is real even for evanescent modes)
                cos0 = np.cos(kz[:, None]  * dz[None,   :])   # (Nq, Ny)
                cos1 = np.cos(m[:,  None]  * dphi[None, :])    # (mmax, Ny)

                # observer K_m:  (Nq, mmax, 1)
                r_obs = obs_r[ix]
                if r_obs <= self.radius * (1 + eps_r):
                    continue                    # inside or on the cylinder boundary

                K_obs = kv(
                    m_qm[:, :, None],
                    kappa_qm[:, :, None] * r_obs
                )                               # (Nq, mmax, 1)

                # The full kernel:
                #   beta(q,m) * K_obs(q,m) * K_src(q,m,n) * cos0(q,n) * cos1(m,n) * epsm(m)
                A = beta[:, :, None] * K_obs * K_src   # (Nq, mmax, Ny)

                newterm = np.einsum(
                    "q,qmn,qn,mn,m->n",
                    ww,          # (Nq,)
                    A,           # (Nq, mmax, Ny)
                    cos0,        # (Nq, Ny)
                    cos1,        # (mmax, Ny)
                    epsm,        # (mmax,)
                    optimize=True
                )                               # (Ny,)

                G[ik, ix, :] += newterm

        # Prefactor: same (-i/4pi) as propagating part, but the evanescent
        # Bessel identity introduces an extra factor of (2/pi) relative to the
        # H_m^(1) → K_m substitution (H_m^(1)(iz) = (2/pi)(-i)^{m+1} K_m(z)).
        # For the *scattering* Green's function the overall prefactor is the same
        # because beta already absorbs the modal conversion factor; only the
        # integration measure changes (handled by jac above).
        G *= (-1j / (4 * np.pi))
        return G
    
    def getScatteringGreen(self, x, y, k):
        # TODO: implement the evernescent near-field terms!
        return self.getScatteringGreenPropagating(x, y, k) + self.getScatteringGreenEvanescent(x, y, k)
    
    def getScatteringGreenGradientHighMemory(self, x, y, k,):

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

    def getScatteringGreenGradientPropagating(self, x, y, k):

        mmax = self._numerics.get("nmax", int(k.max() * self.radius) + 10)
        Nq_prop = self._numerics.get("Nq_prop", 100)
        eps_radius = self._numerics.get("eps_radius", 1e-12)

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

                if np.any(np.isnan(gradG)):
                    print('WARNING: NaN values detected in the computation')

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
 
    def getScatteringGreenGradientEvanescent(self, x, y, k):
        """
        Evanescent (k_z > k) contribution to the gradient of the cylinder
        scattering Green's function, with respect to the SOURCE coordinates y.

        Variable substitution:  k_z = k/sin(t),  t in (0, pi/2)
        kappa = sqrt(k_z^2 - k^2) = k*cos(t)/sin(t)
        Jacobian: dk_z = k*cos(t)/sin^2(t) dt

        Radial Bessel functions (outside cylinder, r > a):
        K_m(kappa*r)   — decays as r→∞, physically admissible
        K_m'(kappa*r) = -kappa/2 * (K_{m-1} + K_{m+1})  [recurrence]

        Angular derivative:
        d/dphi [cos(m*dphi)] = m * sin(m*dphi)
        then divide by src_r to convert to the phi unit-vector component

        Axial derivative:
        d/dz_src [cos(kz*(z_obs - z_src))] = kz * sin(kz*dz)
        (note sign: derivative w.r.t. z_src flips the sign of dz,
        but cos is even so the sin picks up the minus from the chain rule —
        however the propagating code uses +kz*sin, matching d/dz_src of
        cos(kz*(z_obs-z_src)) = +kz*sin(kz*dz), consistent convention)
        """

        mmax    = self._numerics.get("nmax",       int(k.max() * self.radius) + 10)
        Nq_evan = self._numerics.get("Nq_evan",    100)
        eps_r   = self._numerics.get("eps_radius", 1e-12)

        if self.dim != 3:
            raise NotImplementedError

        k  = np.atleast_1d(k)
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

        m    = np.arange(0, mmax)
        epsm = np.ones(mmax);  epsm[1:] = 2

        # Map k_z in [k,∞) via k_z = k/sin(t), t in (0, pi/2)
        t_gl, w_gl = leggauss(Nq_evan)
        t  = 0.25 * np.pi * (t_gl + 1.0)
        w  = w_gl * 0.25 * np.pi

        sin_t = np.sin(t)
        cos_t = np.cos(t)

        for ik, k0 in enumerate(k):

            print(f"[evan grad k-loop] {ik+1}/{Nk}  k={k0:.4e}")

            kz    = k0 / sin_t                      # (Nq,)  k_z > k
            kappa = k0 * cos_t / sin_t              # (Nq,)  kappa > 0
            jac   = k0 * cos_t / sin_t**2           # (Nq,)  |dk_z/dt|
            ww    = w * jac                         # effective quadrature weights

            kappa_qm = kappa[:, None]               # (Nq, 1)
            m_qm     = m[None, :]                   # (1, mmax)

            # --- scattering coefficient beta (same as in scalar evanescent) ---
            u   = kappa_qm * self.radius            # (Nq, mmax)
            Im1 = iv(m_qm - 1, u)
            Ip1 = iv(m_qm + 1, u)
            Km1 = kv(m_qm - 1, u)
            Kp1 = kv(m_qm + 1, u)
            denom = Km1 + Kp1
            safe  = np.abs(denom) > 1e-50
            beta  = np.where(safe, (Im1 + Ip1) / np.where(safe, denom, 1.0), 0.0)
            # (Nq, mmax)

            # --- source K_m and its radial derivative (Nq, mmax, Ny) ---
            kappa_src = kappa_qm[:, :, None] * src_r[None, None, :]   # (Nq, mmax, Ny)

            K_src      = kv(m_qm[:, :, None], kappa_src)
            K_src_m1   = kv(m_qm[:, :, None] - 1, kappa_src)
            K_src_p1   = kv(m_qm[:, :, None] + 1, kappa_src)

            # K_m'(u) = -1/2 * (K_{m-1}(u) + K_{m+1}(u))
            # Chain rule: d/dr [K_m(kappa*r)] = kappa * K_m'(kappa*r)
            #           = -kappa/2 * (K_{m-1}(kappa*r) + K_{m+1}(kappa*r))
            dK_src_dr = -kappa_qm[:, :, None] * 0.5 * (K_src_m1 + K_src_p1)
            # (Nq, mmax, Ny)

            for ix in range(Nx):

                print(f"    observer {ix+1}/{Nx}")

                if obs_r[ix] <= self.radius * (1 + eps_r):
                    continue                        # observer inside/on cylinder

                dz   = obs_x[ix]  - src_x          # (Ny,)
                dphi = obs_phi[ix] - src_phi        # (Ny,)

                cos1        = np.cos(m[:, None]  * dphi[None, :])           # (mmax, Ny)
                dcos1_dphi  = m[:, None] * np.sin(m[:, None] * dphi[None,:])# (mmax, Ny)

                cos0        = np.cos(kz[:, None]  * dz[None, :])            # (Nq, Ny)
                dcos0_dz    = kz[:, None] * np.sin(kz[:, None] * dz[None,:])# (Nq, Ny)

                # observer K_m (Nq, mmax, 1) — scalar, no Ny axis needed
                kappa_obs = kappa_qm[:, :, None] * obs_r[ix]
                K_obs     = kv(m_qm[:, :, None], kappa_obs)                 # (Nq, mmax, 1)

                # --- three gradient components ---
                A = beta[:, :, None] * K_obs * dK_src_dr   # radial:   d/dr_src
                B = beta[:, :, None] * K_obs * K_src        # angular and axial

                # d/dr_src
                newterm_dr = np.einsum(
                    "q,qmn,qn,mn,m->n",
                    ww, A, cos0, cos1, epsm,
                    optimize=True
                )

                # d/dphi_src  (1/r factor converts arc-length to angle)
                newterm_dphi = np.einsum(
                    "q,qmn,qn,mn,m->n",
                    ww, B, cos0,
                    dcos1_dphi / src_r[None, :],
                    epsm,
                    optimize=True
                )

                # d/dz_src
                newterm_dz = np.einsum(
                    "q,qmn,qn,mn,m->n",
                    ww, B, dcos0_dz, cos1, epsm,
                    optimize=True
                )

                gradG[0, ik, ix, :] += newterm_dr
                gradG[1, ik, ix, :] += newterm_dphi
                gradG[2, ik, ix, :] += newterm_dz

                if np.any(np.isnan(gradG)):
                    print('WARNING: NaN values detected in the computation')

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
    
    def getScatteringGreenGradient(self, x, y, k):
        # TODO: implement the evernescent near-field terms!
        return self.getScatteringGreenGradientPropagating(x, y, k) + self.getScatteringGreenGradientEvanescent(x, y, k) 
    

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

        return fig, ax