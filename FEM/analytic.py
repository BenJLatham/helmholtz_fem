import numpy as np
from scipy.special import spherical_jn, spherical_yn, spherical_in, legendre
from scipy.integrate import quad
import cxroots as cx


__all__ = [
    "spherical_hankel1",
    "spherical_hankel1_prime",
    "determinant",
    "scattering_coeffs",
    "generate_weight",
    "vgenerate_weight",
    "analytic_total_field",
    "analytic_norm",
]

def spherical_hankel1(n, z):
    """
    Spherical Hankel function of the first kind: h_n^{(1)}(z).
    """
    return spherical_jn(n, z) + 1j * spherical_yn(n, z)


def spherical_hankel1_prime(n, z):
    """
    Derivative of the spherical Hankel function of the first kind.
    """
    return spherical_jn(n, z, derivative=True) + 1j * spherical_yn(n, z, derivative=True)


def determinant(n, k, epsm):
    """
    Boundary determinant for mode n at wavenumber k and permittivity epsm.
    """
    if epsm > 0:
        km = np.sqrt(epsm) * k
        return (
            (-1/np.sqrt(epsm)) * (spherical_hankel1(n, k) * spherical_jn(n, km, derivative=True))
            + spherical_hankel1_prime(n, k) * spherical_jn(n, km)
        )
    else:
        kmm = np.sqrt(-epsm) * k
        return (
            (1/np.sqrt(-epsm)) * (spherical_hankel1(n, k) * spherical_in(n, kmm, derivative=True))
            + spherical_hankel1_prime(n, k) * spherical_in(n, kmm)
        )


def scattering_coeffs(n, epsm, k):
    """
    Compute the scattering (A) and internal (B) coefficients for spherical mode n.
    Returns (A, B).
    """
    cst = (1j**n) * np.sqrt((2*n + 1) * (4 * np.pi))
    I = lambda z: spherical_in(n, z)
    J = lambda z: spherical_jn(n, z)
    Y = lambda z: spherical_yn(n, z)
    H = lambda z: J(z) + 1j * Y(z)
    Di = lambda z: spherical_in(n, z, derivative=True)
    Dj = lambda z: spherical_jn(n, z, derivative=True)
    Dy = lambda z: spherical_yn(n, z, derivative=True)
    Dh = lambda z: Dj(z) + 1j * Dy(z)

    if epsm > 0:
        km = k * np.sqrt(epsm)
        A = (
            cst * ((1/np.sqrt(epsm)) * J(k) * Dj(km) - Dj(k) * J(km))
            / (Dh(k) * J(km) - (1/np.sqrt(epsm)) * H(k) * Dj(km))
        )
        B = (A * H(k) + cst * J(k)) / J(km)
    else:
        kmm = k * np.sqrt(-epsm)
        A = (
            cst * ((-1/np.sqrt(-epsm)) * J(k) * Di(kmm) - Dj(k) * I(kmm))
            / (Dh(k) * I(kmm) + (1/np.sqrt(-epsm)) * H(k) * Di(kmm))
        )
        B = (A * H(k) + cst * J(k)) / I(kmm)
    return A, B


def generate_weight(n, r, k, epsm):
    """
    Radial weight function for the analytic solution of mode n.
    - r >= 1: outside region (scattered + incident)
    - r < 1: inside region
    """
    A, B = scattering_coeffs(n, epsm, k)
    cst = (1j**n) * np.sqrt((2*n + 1) * (4 * np.pi))
    if r >= 1.0:
        return A * spherical_hankel1(n, k * r) + cst * spherical_jn(n, k * r)
    else:
        if epsm >= 0:
            return B * spherical_jn(n, k * np.sqrt(epsm) * r)
        else:
            return B * spherical_in(n, k * np.sqrt(-epsm) * r)

# Vectorized version over r
vgenerate_weight = np.vectorize(
    generate_weight, excluded=[0, 2, 3], otypes=[complex]
)


def analytic_total_field(coords, tol_int, k, epsm):
    """
    Compute the total field u(r,theta) at points `coords` in R^3, truncated at mode `tol_int`.

    Parameters
    ----------
    coords : array_like of shape (N, 3)
        Cartesian coordinates of evaluation points.
    tol_int : int
        Maximum spherical mode index (inclusive).
    k : float
        Free-space wavenumber.
    epsm : float
        Relative permittivity inside the sphere.

    Returns
    -------
    u : ndarray of shape (N,)
        Complex total field at each point.
    """
    pts = np.asarray(coords)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")
    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    r = np.sqrt(x**2 + y**2 + z**2)
    # avoid division by zero at r=0
    cos_theta = np.where(r > 0, z / r, 1.0)
    u = np.zeros_like(r, dtype=complex)

    for n in range(tol_int + 1):
        # normalized Legendre P_n(cos theta)
        Pn = legendre(n)(cos_theta)
        Y_n0 = np.sqrt((2*n + 1) / (4 * np.pi)) * Pn
        # radial part for all points
        A, B = scattering_coeffs(n, epsm, k)
        cst = (1j**n) * np.sqrt((2*n + 1) * (4 * np.pi))
        f_out = A * spherical_hankel1(n, k * r) + cst * spherical_jn(n, k * r)
        if epsm >= 0:
            f_in = B * spherical_jn(n, k * np.sqrt(epsm) * r)
        else:
            f_in = B * spherical_in(n, k * np.sqrt(-epsm) * r)
        radial = np.where(r >= 1.0, f_out, f_in)
        u += radial * Y_n0

    return u


def analytic_norm(R, tol_int, epsm, k):
    """
    L2 norm of the analytic solution on [0,R], combining inside and outside.

    Integrates |u|^2 r^2 dr over radial domain and sums over modes.
    """
    total = 0.0
    for n in range(tol_int + 1):
        # build radial weight functions
        A, B = scattering_coeffs(n, epsm, k)
        cst = (1j**n) * np.sqrt((2*n + 1) * (4 * np.pi))
        # outside integrand
        w1 = lambda r: A * spherical_hankel1(n, k * r) + cst * spherical_jn(n, k * r)
        # inside integrand
        if epsm >= 0:
            w2 = lambda r: B * spherical_jn(n, k * np.sqrt(epsm) * r)
        else:
            w2 = lambda r: B * spherical_in(n, k * np.sqrt(-epsm) * r)
        # radial weighting
        integrand1 = lambda r: np.abs(w1(r))**2 * r**2
        integrand2 = lambda r: np.abs(w2(r))**2 * r**2
        total += quad(integrand2, 0.0, 1.0)[0]
        total += quad(integrand1, 1.0, R)[0]
    return np.sqrt(total)


def ddet(n, k, epsm):
    """
    ∂_k determinant(n,k,epsm) from your NGsolve code
    """
    # reuse your ddjn, ddyn, ddin, ddhankel definitions:
    ddjn = lambda n,k : spherical_jn(n-1,k,derivative=True) \
                       - ( (n+1)*( (1/k)*spherical_jn(n,k,derivative=True)
                                   + (-1/k**2)*spherical_jn(n,k) ) )
    ddyn = lambda n,k : spherical_yn(n-1,k,derivative=True) \
                       - ( (n+1)*( (1/k)*spherical_yn(n,k,derivative=True)
                                   + (-1/k**2)*spherical_yn(n,k) ) )
    ddin = lambda n,k : spherical_in(n-1,k,derivative=True) \
                       - ( (n+1)*( (1/k)*spherical_in(n,k,derivative=True)
                                   + (-1/k**2)*spherical_in(n,k) ) )
    ddhankel = lambda n,k: ddjn(n,k) + 1j*ddyn(n,k)

    if epsm > 0:
        km = np.sqrt(epsm)*k
        return (
          (-1/np.sqrt(epsm))*(spherical_hankel1(n,k)*spherical_jn(n,km,derivative=True))
            + spherical_hankel1_prime(n,k)*spherical_jn(n,km)
          )  +  (
          ddhankel(n,k)*spherical_jn(n,km)
            + spherical_hankel1_prime(n,k)*np.sqrt(epsm)*spherical_jn(n,km,derivative=True)
          )
    else:
        kmm = np.sqrt(-epsm)*k
        return (
          (1/np.sqrt(-epsm))*(spherical_hankel1(n,k)*spherical_in(n,kmm,derivative=True))
            + spherical_hankel1_prime(n,k)*spherical_in(n,kmm)
          )  +  (
          ddhankel(n,k)*spherical_in(n,kmm)
            + spherical_hankel1_prime(n,k)*np.sqrt(-epsm)*spherical_in(n,kmm,derivative=True)
          )

def find_resonance(n, epsm, r_min=0.1, r_max=10.0):
    """
    Find the near‐real plasmonic resonance λ for mode n
    via cxroots.Annulus, using your det & ddet.
    Returns (k_real, k_complex).
    """
    detn  = lambda lam: determinant(n, lam, epsm)
    ddetn = lambda lam: ddet(n, lam, epsm)

    contour = cx.Annulus(0, (r_min, r_max))
    roots, _ = contour.roots(detn, ddetn)
    pos = [z for z in roots if z.real > 0]
    kp  = min(pos, key=lambda z: abs(z.imag))
    return kp.real, kp