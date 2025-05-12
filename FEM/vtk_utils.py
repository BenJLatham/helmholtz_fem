import numpy as np
import meshio
from ngsolve import GridFunction

"""
VTK export utilities for FEM solution fields.
"""

def gridfunction_to_point_data(mesh3d: meshio.Mesh, gf: GridFunction) -> np.ndarray:
    """
    Evaluate a GridFunction at the mesh points and return complex values.

    Parameters
    ----------
    mesh3d : meshio.Mesh
        The meshio mesh containing point coordinates.
    gf : GridFunction
        The NGsolve GridFunction to evaluate.

    Returns
    -------
    values : ndarray of shape (N,)
        Complex values of gf at each mesh point.
    """
    pts = mesh3d.points
    # Evaluate grid function at each point
    return np.array([gf(x, y, z) for x, y, z in pts], dtype=complex)


def export_fields_to_vtk(
    mesh3d: meshio.Mesh,
    cells,
    fields: dict,
    filename: str
) -> None:
    """
    Export multiple solution fields to a single VTK file.

    This function splits complex-valued fields into real, imaginary, and magnitude components.

    Parameters
    ----------
    mesh3d : meshio.Mesh
        The Gmsh-generated mesh (with .points attribute).
    cells : list
        The cell blocks defining element connectivity (e.g., mesh3d.cells).
    fields : dict
        Mapping from field name to numpy array of shape (N,) of float or complex.
    filename : str
        Output filename for the VTK file (e.g., 'solution.vtk').
    """
    point_data = {}
    for name, data in fields.items():
        arr = np.asarray(data)
        if np.iscomplexobj(arr):
            # split complex into three real arrays
            point_data[f"{name}_real"] = arr.real
            point_data[f"{name}_imag"] = arr.imag
            point_data[f"{name}_abs"]  = np.abs(arr)
        else:
            point_data[name] = arr

    meshio.write_points_cells(
        filename,
        points=mesh3d.points,
        cells=cells,
        point_data=point_data
    )
