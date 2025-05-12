# Helmholtz-FEM

[![CI](https://github.com/BenJLatham/helmholtz_fem/actions/workflows/ci.yml/badge.svg)](https://github.com/BenJLatham/helmholtz_fem/actions) [![PyPI version](https://badge.fury.io/py/helmholtz_fem.svg)](https://pypi.org/project/helmholtz_fem) [![Coverage Status](https://coveralls.io/repos/github/BenJLatham/helmholtz_fem/badge.svg?branch=main)](https://coveralls.io/github/BenJLatham/helmholtz_fem?branch=main)

Standard and plasmonic-enriched finite-element solvers for Helmholtz scattering by a dielectric sphere, built on Gmsh + Netgen + NGSolve.

---

## Features

- **Mesh generation** via Gmsh transfinite sphere (with interior/exterior labeling)  
- **Standard FEM** solver 
- **Plasmonic Enriched FEM** solver: single resonance Schur-complement enrichment  
- **Pure-Python analytic routines** (Bessel/Hankel expansions, total-field eval.)  
- **VTK export** of complex fields (real, imag, abs) for Paraview/VisIt  
- Lightweight API, easy to script and extend

---

## Installation

```bash
# Clone repository
git clone https://github.com/BenJLatham/helmholtz_fem.git
cd helmholtz_fem

# Install in editable mode (requires cmake for Netgen, Gmsh SDK installed,
# plus Python packages: numpy, scipy, meshio, netgen, ngsolve, cxroots)
pip install -e .[test]
````

**Requirements**:

* Python ≥ 3.7
* [numpy](https://numpy.org)
* [scipy](https://scipy.org)
* [meshio](https://github.com/nschloe/meshio)
* [netgen](https://github.com/NGSolve/netgen)
* [ngsolve](https://ngsolve.org)
* [gmsh-sdk](https://github.com/NSGeeks/gmsh-sdk)
* [cxroots](https://github.com/nschloe/cxroots)

---

## Quick Start

```python
from helmholtz_fem import StandardFEM, PlasmonEnrichedFEM

# 1) Standard scattering solve
fem = StandardFEM(k=5.0, epsm=-2.0, R=5.0, order=2)
fem.setup(a=0.15, h_band=0.2, h_outer=0.5)
u_std = fem.solve(export_vtk=True, vtk_filename="std_fem.vtk")
err_std = fem.compute_error(tol_int=30)
print(f"Standard FEM L²-error ≈ {err_std:.2e}")

# 2) Plasmon-enriched solve (mode ℓ=1)
enr = PlasmonEnrichedFEM(k=5.0, epsm=-2.0, R=5.0, order=2, ℓ=1)
enr.setup(a=0.15, h_band=0.2, h_outer=0.5)
u_enr = enr.solve(export_vtk=True, vtk_filename="enriched_fem.vtk")
err_enr = enr.compute_error(tol_int=30)
print(f"Enriched FEM L²-error ≈ {err_enr:.2e}, α ≈ {enr.alpha:.2e}")
```

---

---

## API Reference

### `StandardFEM(k, epsm, R, order=1)`

* **`setup(a, r_inner, r_outer=None, h_band, h_outer)`**
* **`solve(export_vtk=False, vtk_filename="solution.vtk") → GridFunction`**
* **`compute_error(tol_int) → float`**

### `PlasmonEnrichedFEM(k, epsm, R, order=1, ℓ=1, resonance_k=None)`

Inherits all methods of `StandardFEM`, adds:

* **`setup(...)`** locates the plasmon pole via `analytic.find_resonance`
* **`solve(export_vtk=False, vtk_filename="enriched_solution.vtk") → GridFunction`**
  solves the 2×2 Schur system, stores `self.alpha`

### `helmholtz_fem.analytic`

* **`analytic_total_field(points, tol, k, epsm)`**
* **`find_resonance(n, epsm, r_min=0.1, r_max=10.0)`**
* **`scattering_coeffs(n, epsm, k_real)`**
* **`generate_weight(n, r, k, epsm)`**

---


## License

This project is licensed under the [MIT License](LICENSE).

---

## Citation

If you use **Helmholtz-FEM** in published work, please cite:

> **Benjamin Latham**, *Helmholtz-FEM: Standard and plasmonic-enriched FEM solvers for Helmholtz scattering*, Version 0.1.0, 2025.
> DOI: *to be assigned*

```
```
