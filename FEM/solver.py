from ngsolve import (
    H1, BilinearForm, LinearForm, GridFunction,
    CoefficientFunction, grad, dx, ds
)
import numpy as np
from .mesh import generate_sphere_mesh, NGMesh
import analytic
from analytic import find_resonance, scattering_coeffs, generate_weight
from .vtk_utils import gridfunction_to_point_data, export_fields_to_vtk

class StandardFEM:
    '''
    Standard FEM solver for Helmholtz scattering by a dielectric sphere
    in a spherical computational domain with first-order absorbing boundary.
    '''
    def __init__(self, k: float, epsm: float, R: float, order: int = 1):
        self.k = k
        self.epsm = epsm
        self.R = R
        self.order = order
        # placeholders for mesh and solution objects
        self.mesh3d = None
        self.ngmesh = None
        self.fes = None
        self.a = None
        self.f = None
        self.gf = None

    def setup(self,
              a: float = 0.15,
              r_inner: float = 1.0,
              r_outer: float = None,
              h_band: float = 0.25,
              h_outer: float = 0.5):
        '''
        Build the mesh, finite element space, and assemble system.
        '''
        if r_outer is None:
            r_outer = self.R
        mesh3d, netgen_mesh = generate_sphere_mesh(
            a, r_inner, r_outer, h_band, h_outer)
        self.mesh3d = mesh3d
        self.ngmesh = NGMesh(netgen_mesh)
        self.fes = H1(self.ngmesh, order=self.order, complex=True)
        u, v = self.fes.TnT()
        self.a = BilinearForm(self.fes)

        # 1) grad‐grad with piecewise 1/eps
        self.a += (1/self.epsm) * grad(u) * grad(v) * dx("metal")
        self.a +=              1 * grad(u) * grad(v) * dx("vacuum")

        # 2) mass term (−k² u v) over _both_ regions
        self.a += -(self.k**2) * u * v * ( dx("metal") + dx("vacuum") )

        # 3) simple absorbing BC on the outer boundary
        self.a += 1j*self.k * u * v * ds("outer")

        self.a.Assemble()
        
        self.f = LinearForm(self.fes)
        inc = CoefficientFunction('exp(1j*k*z)')
        self.f += (inc.Deriv('n') + 1j*self.k*inc) * v * ds("outer")
        self.f.Assemble()

    def solve(self,
              export_vtk: bool = False,
              vtk_filename: str = "solution.vtk"
             ) -> GridFunction:
        """
        Solve the system and return the GridFunction.
        If export_vtk is True, write out a VTK file of the total field.
        """
        self.gf = GridFunction(self.fes)
        self.gf.Set(self.a.mat.Inverse(self.fes.FreeDofs()) * self.f.vec)

        if export_vtk:
            # sample at mesh points
            u_vals = gridfunction_to_point_data(self.mesh3d, self.gf)
            fields = {'FEM_Total_Field': u_vals}
            export_fields_to_vtk(self.mesh3d, self.mesh3d.cells,
                                 fields, vtk_filename)

        return self.gf

    def compute_error(self, tol_int: int) -> float:
        '''
        Compute relative L2 error versus analytic solution.
        '''
        coords = self.mesh3d.points
        u_ana = analytic.analytic_total_field(coords, tol_int, self.k, self.epsm)
        u_fe = np.array([self.gf(xi, yi, zi) for xi, yi, zi in coords])
        return np.linalg.norm(u_fe - u_ana) / np.linalg.norm(u_ana)

class PlasmonEnrichedFEM(StandardFEM):
    '''
    Enriched FEM solver with plasmonic resonance basis functions.
    '''
    def __init__(self, k: float, epsm: float, R: float, order: int = 1,
                 ℓ: int = 1, resonance_k: complex = None):
        super().__init__(k, epsm, R, order)
        self.ℓ = ℓ
        self.resonance_k = resonance_k

    def setup(self, *args, **kwargs):
        # 1) Base system assembly
        super().setup(*args, **kwargs)

        # 2) Locate plasmonic pole if not provided
        if self.resonance_k is None:
            _, kp = find_resonance(self.ℓ, self.epsm)
            self.resonance_k = kp
        else:
            kp = self.resonance_k

        # 3) Compute scattering coeffs at kp.real
        Aℓ, Bℓ = scattering_coeffs(self.ℓ, self.epsm, kp.real)

        # 4) Define enrichment mode function up and its conjugate
        def up_fun(x, y, z):
            r = np.sqrt(x*x + y*y + z*z)
            w = generate_weight(self.ℓ, r, kp.real, self.epsm)
            θ = np.arccos(z/r)
            Yℓ0 = np.sqrt((2*self.ℓ+1)/(4*np.pi)) * np.polynomial.legendre.Legendre.basis(self.ℓ)(np.cos(θ))
            return w * Yℓ0

        self.up      = CoefficientFunction(up_fun)
        self.conj_up = CoefficientFunction(lambda x,y,z: np.conj(up_fun(x,y,z)))

        # 5) Assemble coupling forms b and c
        u, v = self.fes.TnT()
        self.b = LinearForm(self.fes)
        self.b += ( (self.k**2 - kp.real**2) * self.conj_up * v ) * dx
        self.b.Assemble()
        self.c = LinearForm(self.fes)
        self.c += ( (self.k**2 - kp.real**2) * self.up * v ) * dx
        self.c.Assemble()

        # 6) Scalar Schur block d and RHS g
        self.d = self.b.vec @ self.conj_up.vec
        self.g = 0

    def solve(self,
              export_vtk: bool = False,
              vtk_filename: str = "enriched_solution.vtk"
             ) -> GridFunction:
        """
        Solve the 2×2 Schur system and return the enriched GridFunction.
        If export_vtk is True, write out a VTK file of the enriched total field.
        """
        # 1) solve A u0 = f
        u0 = GridFunction(self.fes)
        u0.vec.data = self.a.mat.Inverse(self.fes.FreeDofs()) @ self.f.vec

        # 2) build Schur complement
        Ainv_b = self.a.mat.Inverse(self.fes.FreeDofs()) @ self.b.vec
        cT_u0 = self.c.vec @ u0.vec
        S = self.d - (self.c.vec @ Ainv_b)

        # 3) solve for alpha
        α = (self.g - cT_u0) / S

        # 4) corrected field
        u = GridFunction(self.fes)
        u.vec.data = u0.vec - α * Ainv_b

        # store
        self.gf    = u
        self.alpha = α

        if export_vtk:
            u_vals = gridfunction_to_point_data(self.mesh3d, self.gf)
            fields = {'Enriched_Total_Field': u_vals}
            export_fields_to_vtk(self.mesh3d, self.mesh3d.cells,
                                 fields, vtk_filename)

        return u