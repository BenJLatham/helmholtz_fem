"""
T-Conf Sphere Transfinite Mesh Configuration

This module provides low-level helpers to apply transfinite meshing
to curves, surfaces, and volumes in a symmetric spherical mesh.
"""

from typing import Sequence, Union
import gmsh
from T_Conf.utils import ensure_gmsh_available


def set_transfinite(
    radial_curves: Sequence[int],
    circular_curves: Sequence[int],
    surfaces: Sequence[Union[int, tuple]],
    n_div_radial: int,
    n_div_circular: int,
    bumpcoef: float = 1.0,
) -> None:
    """
    Apply transfinite meshing to the given curves and surfaces.

    Parameters
    ----------
    radial_curves : Sequence[int]
        IDs of curves running radially outward from the sphere center.
    circular_curves : Sequence[int]
        IDs of curves forming concentric circles around the sphere.
    surfaces : Sequence[Union[int, tuple]]
        Surface entries to meshing. Each entry may be:
          - int: surface tag.
          - (tag, arrangement: str).
          - (tag, arrangement: str, cornerTags: Sequence[int]).
    n_div_radial : int
        Number of mesh divisions along each radial curve.
    n_div_circular : int
        Number of mesh divisions along each circular curve.
    bumpcoef : float, optional
        Coefficient to bias element sizing (default is 1.0 for uniform spacing).
    """
    ensure_gmsh_available()

    # Curves
    for cid in radial_curves:
        gmsh.model.geo.mesh.setTransfiniteCurve(cid, n_div_radial, coef=bumpcoef)
    for cid in circular_curves:
        gmsh.model.geo.mesh.setTransfiniteCurve(cid, n_div_circular, coef=bumpcoef)

    # Surfaces
    for entry in surfaces:
        if isinstance(entry, tuple):
            if len(entry) == 2:
                tag, arrangement = entry
                gmsh.model.geo.mesh.setTransfiniteSurface(tag, arrangement)
            elif len(entry) == 3:
                tag, arrangement, cornerTags = entry
                gmsh.model.geo.mesh.setTransfiniteSurface(tag, arrangement, list(cornerTags))
            else:
                raise ValueError(f"Invalid surf entry: {entry}")
        else:
            gmsh.model.geo.mesh.setTransfiniteSurface(int(entry))


def set_transfinite_volume(volumes: Sequence[int], corner_points: Sequence[int] = None) -> None:
    """
    Apply transfinite meshing to the given volumes.

    Parameters
    ----------
    volumes : Sequence[int]
        IDs of volumes to mesh using a structured transfinite grid.
    corner_points : Sequence[int], optional
        IDs of corner points to control the volume mesh orientation.

    Notes
    -----
    - If `corner_points` is provided, it will enforce a specific node ordering in the volume.
    - Call after set_transfinite on curves and surfaces, and before gmsh.model.geo.synchronize().
    """
    ensure_gmsh_available()

    for vid in volumes:
        if corner_points:
            gmsh.model.geo.mesh.setTransfiniteVolume(vid, list(corner_points))
        else:
            gmsh.model.geo.mesh.setTransfiniteVolume(vid)
