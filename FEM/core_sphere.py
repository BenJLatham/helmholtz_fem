"""
T-Conf Sphere Core Module

This module provides the `Transfinite_Sphere` function to build a symmetric,
T-conforming spherical mesh in Gmsh, splitting the domain into an inner band
and bulk regions, then creating physical groups for materials and boundaries.
"""

import gmsh
import math
import numpy as np

from FEM.utils import ensure_gmsh_available
from FEM.transfinite import set_transfinite
from FEM.mesh_utils import create_wedge_volumes


def Transfinite_Sphere(
    a: float,
    rInner: float,
    rOuter: float,
    h_band: float = 0.1,
    h_outer: float = 0.1,
    Bumpcoef: float = 1,
    InnerMaterialName: str = "InnerMaterial",
    OuterMaterialName: str = "OuterMaterial",
    InterfaceName: str = "Interface",
    OuterBoundaryName: str = "Outer",
) -> None:
    """
    Build a sphere from radius 0 to rOuter, with a transfinite band
    of thickness 2*a centered at rInner, meshed symmetrically, and
    the remaining inner and outer bulk regions meshed unstructured.
    The domain is split into 8 octants via revolution.

    Parameters
    ----------
    a : float
        Half-thickness of the transfinite band around rInner.
    rInner : float
        Radius of the center of the transfinite band.
    rOuter : float
        Outer radius of the sphere.
    h_band : float, optional
        Mesh size parameter for band curves (default 0.1).
    h_outer : float, optional
        Mesh size parameter for outer curves (default 0.1).
    Bumpcoef : float, optional
        Bias coefficient for transfinite progression (default 1).
    InnerMaterialName : str, optional
        Physical group name for inner volumes (default 'InnerMaterial').
    OuterMaterialName : str, optional
        Physical group name for outer volumes (default 'OuterMaterial').
    InterfaceName : str, optional
        Physical group name for the interface surfaces (default 'Interface').
    OuterBoundaryName : str, optional
        Physical group name for the outer spherical boundary (default 'Outer').

    Notes
    -----
    After calling this function, you must call:

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.write("your_mesh.msh")
        gmsh.fltk.run()  # if you want the GUI
        gmsh.finalize()

    to generate and save the mesh.
    """
    # Ensure Gmsh is available
    ensure_gmsh_available()

    # Only initialize Gmsh if not already
    if not gmsh.isInitialized():
        gmsh.initialize()
    # Only add model if none exists
    try:
        current = gmsh.model.getCurrent()
    except Exception:
        # No model exists yet
        gmsh.model.add("TransfiniteSphere")
    else:
        if not current:
            gmsh.model.add("TransfiniteSphere")

    # Parameters for revolveSurface
    revolveAxis = (1, 0, 0)
    revolveCenter = (-rOuter - 1, 0, 0)
    angleSweep = math.pi / 2

    if rInner - a < 0:
        raise ValueError("Parameter error: rInner - a must be nonnegative. (a is too large)")
    if rInner + a > rOuter:
        raise ValueError("Parameter error: rOuter < rInner + a. (a is too large)")

    nDiv_radial = max(4, int(np.ceil(2 * a / h_band)))
    nDiv_curved = max(4, int(np.ceil((0.5 * math.pi) / h_band)))

    allInnerVols = []
    allOuterVols = []

    def meshSizeAt(x, y, z):
        r = math.sqrt(x * x + y * y + z * z)
        # if we’re in the band: fine
        if rInner - a <= r <= rInner + a:
            return h_band
        # otherwise coarser
        return h_outer

    # Define common points
    p0 = gmsh.model.geo.addPoint(0, 0, 0, meshSizeAt(0, 0, 0))
    p1 = gmsh.model.geo.addPoint(rInner - a, 0, 0, meshSizeAt(rInner - a, 0, 0))
    p2 = gmsh.model.geo.addPoint(rInner + a, 0, 0, meshSizeAt(rInner + a, 0, 0))
    p3 = gmsh.model.geo.addPoint(0, 0, rInner - a, meshSizeAt(0, 0, rInner - a))
    p4 = gmsh.model.geo.addPoint(0, 0, rInner + a, meshSizeAt(0, 0, rInner + a))
    p5 = gmsh.model.geo.addPoint(rInner, 0, 0, meshSizeAt(rInner, 0, 0))
    p6 = gmsh.model.geo.addPoint(0, 0, rInner, meshSizeAt(0, 0, rInner))
    p7 = gmsh.model.geo.addPoint(rOuter, 0, 0, meshSizeAt(rOuter, 0, 0))
    p8 = gmsh.model.geo.addPoint(0, 0, rOuter, meshSizeAt(0, 0, rOuter))

    # Additional points for lower (negative z) region.
    p3_down = gmsh.model.geo.addPoint(0, 0, -(rInner - a), meshSizeAt(0, 0, -(rInner - a)))
    p4_down = gmsh.model.geo.addPoint(0, 0, -(rInner + a), meshSizeAt(0, 0, -(rInner + a)))
    p6_down = gmsh.model.geo.addPoint(0, 0, -rInner, meshSizeAt(0, 0, -rInner))
    p8_down = gmsh.model.geo.addPoint(0, 0, -rOuter, meshSizeAt(0, 0, -rOuter))

    # Points for negative x region.
    p1_neg = gmsh.model.geo.addPoint(-(rInner - a), 0, 0, meshSizeAt(-(rInner - a), 0, 0))
    p2_neg = gmsh.model.geo.addPoint(-(rInner + a), 0, 0, meshSizeAt(-(rInner + a), 0, 0))
    p5_neg = gmsh.model.geo.addPoint(-rInner, 0, 0, meshSizeAt(-rInner, 0, 0))
    p7_neg = gmsh.model.geo.addPoint(-rOuter, 0, 0, meshSizeAt(-rOuter, 0, 0))

    # ------------------ Wedge 1 (first octant) ------------------
    # Outer band surface.
    arcOuterUp1 = gmsh.model.geo.addCircleArc(p2, p0, p4)
    arcInterface1 = gmsh.model.geo.addCircleArc(p5, p0, p6)
    lOutUp1a = gmsh.model.geo.addLine(p5, p2)
    lOutUp1b = gmsh.model.geo.addLine(p4, p6)
    outerLoopUp1 = gmsh.model.geo.addCurveLoop([-lOutUp1b, -arcOuterUp1, -lOutUp1a, arcInterface1])
    outerSurfUp1 = gmsh.model.geo.addPlaneSurface([outerLoopUp1])

    if a == rInner:
        lInUp1a = gmsh.model.geo.addLine(p0, p5)
        lInUp1b = gmsh.model.geo.addLine(p6, p0)
        innerLoopUp1 = gmsh.model.geo.addCurveLoop([lInUp1a, arcInterface1, lInUp1b])
        innerSurfUp1 = gmsh.model.geo.addPlaneSurface([innerLoopUp1])
        innerBulkSurf1 = None
        # Separate radial and curved edges for transfinite meshing.
        set_transfinite(
            [lInUp1a, lInUp1b, lOutUp1a, lOutUp1b],  # radial edges
            [arcOuterUp1, arcInterface1],  # curved edges
            [(innerSurfUp1, "Right"), (outerSurfUp1, "Left")],
            nDiv_radial,
            nDiv_curved,
            Bumpcoef,
        )
    else:
        arcInnerUp1 = gmsh.model.geo.addCircleArc(p1, p0, p3)
        lInUp1a = gmsh.model.geo.addLine(p1, p5)
        lInUp1b = gmsh.model.geo.addLine(p6, p3)
        innerLoopUp1 = gmsh.model.geo.addCurveLoop(
            [-lInUp1a, -arcInterface1, -lInUp1b, arcInnerUp1]
        )
        innerSurfUp1 = gmsh.model.geo.addPlaneSurface([innerLoopUp1])
        set_transfinite(
            [lInUp1a, lInUp1b, lOutUp1a, lOutUp1b],
            [arcOuterUp1, arcInnerUp1, arcInterface1],
            [(innerSurfUp1, "Right"), (outerSurfUp1, "Left")],
            nDiv_radial,
            nDiv_curved,
            Bumpcoef,
        )
        lInBulk1a = gmsh.model.geo.addLine(p0, p1)
        lInBulk1b = gmsh.model.geo.addLine(p3, p0)
        innerBulkLoop1 = gmsh.model.geo.addCurveLoop([lInBulk1a, arcInnerUp1, lInBulk1b])
        innerBulkSurf1 = gmsh.model.geo.addPlaneSurface([innerBulkLoop1])

    # Outer bulk surface for wedge 1.
    if a + rInner == rOuter:
        outerBulkSurf1 = None
    else:
        arcOuterBulk1 = gmsh.model.geo.addCircleArc(p7, p0, p8)
        lOutBulk1a = gmsh.model.geo.addLine(p2, p7)
        lOutBulk1b = gmsh.model.geo.addLine(p8, p4)
        outerBulkLoop1 = gmsh.model.geo.addCurveLoop(
            [-lOutBulk1b, -arcOuterBulk1, -lOutBulk1a, arcOuterUp1]
        )
        outerBulkSurf1 = gmsh.model.geo.addPlaneSurface([outerBulkLoop1])

    gmsh.model.geo.synchronize()
    inner_vols1, outer_vols1 = create_wedge_volumes(
        innerSurfUp1,
        outerSurfUp1,
        innerBulkSurf1,
        outerBulkSurf1,
        revolveAxis,
        revolveCenter,
        angleSweep,
        nDiv_radial,
        nDiv_curved,
        Bumpcoef,
    )
    allInnerVols += [vol for vol in inner_vols1 if vol is not None]
    allOuterVols += [vol for vol in outer_vols1 if vol is not None]

    # ------------------ Wedge 2 (lower, negative z quadrant) ------------------
    arcOuterDown2 = gmsh.model.geo.addCircleArc(p2, p0, p4_down)
    arcInterfaceDown2 = gmsh.model.geo.addCircleArc(p5, p0, p6_down)
    lOutDown2a = gmsh.model.geo.addLine(p5, p2)
    lOutDown2b = gmsh.model.geo.addLine(p4_down, p6_down)
    outerLoopDown2 = gmsh.model.geo.addCurveLoop(
        [lOutDown2a, arcOuterDown2, lOutDown2b, -arcInterfaceDown2]
    )
    outerSurfDown2 = gmsh.model.geo.addPlaneSurface([outerLoopDown2])

    if a == rInner:
        lInDown2a = gmsh.model.geo.addLine(p0, p5)
        lInDown2b = gmsh.model.geo.addLine(p6_down, p0)
        innerLoopDown2 = gmsh.model.geo.addCurveLoop([lInDown2a, arcInterfaceDown2, lInDown2b])
        innerSurfDown2 = gmsh.model.geo.addPlaneSurface([innerLoopDown2])
        innerBulkSurfDown2 = None
        set_transfinite(
            [lInDown2a, lInDown2b, lOutDown2a, lOutDown2b],
            [arcOuterDown2, arcInterfaceDown2],
            [(innerSurfDown2, "Right"), (outerSurfDown2, "Left")],
            nDiv_radial,
            nDiv_curved,
            Bumpcoef,
        )
    else:
        arcInnerDown2 = gmsh.model.geo.addCircleArc(p1, p0, p3_down)
        lInDown2a = gmsh.model.geo.addLine(p1, p5)
        lInDown2b = gmsh.model.geo.addLine(p6_down, p3_down)
        innerLoopDown2 = gmsh.model.geo.addCurveLoop(
            [-lInDown2a, -arcInterfaceDown2, -lInDown2b, arcInnerDown2]
        )
        innerSurfDown2 = gmsh.model.geo.addPlaneSurface([innerLoopDown2])
        set_transfinite(
            [lInDown2a, lInDown2b, lOutDown2a, lOutDown2b],
            [arcOuterDown2, arcInnerDown2, arcInterfaceDown2],
            [(innerSurfDown2, "Right"), (outerSurfDown2, "Left")],
            nDiv_radial,
            nDiv_curved,
            Bumpcoef,
        )
        lInBulkDown2a = gmsh.model.geo.addLine(p0, p1)
        lInBulkDown2b = gmsh.model.geo.addLine(p3_down, p0)
        innerBulkLoopDown2 = gmsh.model.geo.addCurveLoop(
            [lInBulkDown2a, arcInnerDown2, lInBulkDown2b]
        )
        innerBulkSurfDown2 = gmsh.model.geo.addPlaneSurface([innerBulkLoopDown2])

    # Outer bulk surface for wedge 2.
    if a + rInner == rOuter:
        outerBulkSurfDown2 = None
    else:
        arcOuterBulkDown2 = gmsh.model.geo.addCircleArc(p7, p0, p8_down)
        lOutBulkDown2a = gmsh.model.geo.addLine(p2, p7)
        lOutBulkDown2b = gmsh.model.geo.addLine(p8_down, p4_down)
        outerBulkLoopDown2 = gmsh.model.geo.addCurveLoop(
            [-lOutBulkDown2b, -arcOuterBulkDown2, -lOutBulkDown2a, arcOuterDown2]
        )
        outerBulkSurfDown2 = gmsh.model.geo.addPlaneSurface([outerBulkLoopDown2])

    gmsh.model.geo.synchronize()
    inner_vols2, outer_vols2 = create_wedge_volumes(
        innerSurfDown2,
        outerSurfDown2,
        innerBulkSurfDown2,
        outerBulkSurfDown2,
        revolveAxis,
        revolveCenter,
        angleSweep,
        nDiv_radial,
        nDiv_curved,
        Bumpcoef,
    )
    allInnerVols += [vol for vol in inner_vols2 if vol is not None]
    allOuterVols += [vol for vol in outer_vols2 if vol is not None]

    # ------------------ Wedge 3 (negative x, positive z quadrant) ------------------
    arcOuterUp3 = gmsh.model.geo.addCircleArc(p2_neg, p0, p4)
    arcInterface3 = gmsh.model.geo.addCircleArc(p5_neg, p0, p6)
    lOutUp3a = gmsh.model.geo.addLine(p5_neg, p2_neg)
    lOutUp3b = gmsh.model.geo.addLine(p4, p6)
    outerLoopUp3 = gmsh.model.geo.addCurveLoop([lOutUp3a, arcOuterUp3, lOutUp3b, -arcInterface3])
    outerSurfUp3 = gmsh.model.geo.addPlaneSurface([outerLoopUp3])

    if a == rInner:
        lInUp3a = gmsh.model.geo.addLine(p0, p5_neg)
        lInUp3b = gmsh.model.geo.addLine(p6, p0)
        innerLoopUp3 = gmsh.model.geo.addCurveLoop([lInUp3a, arcInterface3, lInUp3b])
        innerSurfUp3 = gmsh.model.geo.addPlaneSurface([innerLoopUp3])
        innerBulkSurf3 = None
        set_transfinite(
            [lInUp3a, lInUp3b, lOutUp3a, lOutUp3b],
            [arcOuterUp3, arcInterface3],
            [(innerSurfUp3, "Right"), (outerSurfUp3, "Left")],
            nDiv_radial,
            nDiv_curved,
            Bumpcoef,
        )
    else:
        arcInnerUp3 = gmsh.model.geo.addCircleArc(p1_neg, p0, p3)
        lInUp3a = gmsh.model.geo.addLine(p1_neg, p5_neg)
        lInUp3b = gmsh.model.geo.addLine(p6, p3)
        innerLoopUp3 = gmsh.model.geo.addCurveLoop(
            [-lInUp3a, -arcInterface3, -lInUp3b, arcInnerUp3]
        )
        innerSurfUp3 = gmsh.model.geo.addPlaneSurface([innerLoopUp3])
        set_transfinite(
            [lInUp3a, lInUp3b, lOutUp3a, lOutUp3b],
            [arcOuterUp3, arcInnerUp3, arcInterface3],
            [(innerSurfUp3, "Right"), (outerSurfUp3, "Left")],
            nDiv_radial,
            nDiv_curved,
            Bumpcoef,
        )
        lInBulk3a = gmsh.model.geo.addLine(p0, p1_neg)
        lInBulk3b = gmsh.model.geo.addLine(p3, p0)
        innerBulkLoop3 = gmsh.model.geo.addCurveLoop([lInBulk3a, arcInnerUp3, lInBulk3b])
        innerBulkSurf3 = gmsh.model.geo.addPlaneSurface([innerBulkLoop3])

    # Outer bulk surface for wedge 3.
    if a + rInner == rOuter:
        outerBulkSurf3 = None
    else:
        arcOuterBulk3 = gmsh.model.geo.addCircleArc(p7_neg, p0, p8)
        lOutBulk3a = gmsh.model.geo.addLine(p2_neg, p7_neg)
        lOutBulk3b = gmsh.model.geo.addLine(p8, p4)
        outerBulkLoop3 = gmsh.model.geo.addCurveLoop(
            [-lOutBulk3b, -arcOuterBulk3, -lOutBulk3a, arcOuterUp3]
        )
        outerBulkSurf3 = gmsh.model.geo.addPlaneSurface([outerBulkLoop3])

    gmsh.model.geo.synchronize()
    inner_vols3, outer_vols3 = create_wedge_volumes(
        innerSurfUp3,
        outerSurfUp3,
        innerBulkSurf3,
        outerBulkSurf3,
        revolveAxis,
        revolveCenter,
        angleSweep,
        nDiv_radial,
        nDiv_curved,
        Bumpcoef,
    )
    allInnerVols += [vol for vol in inner_vols3 if vol is not None]
    allOuterVols += [vol for vol in outer_vols3 if vol is not None]

    # ------------------ Wedge 4 (negative x, negative z quadrant) ------------------
    arcOuterDown4 = gmsh.model.geo.addCircleArc(p2_neg, p0, p4_down)
    arcInterfaceDown4 = gmsh.model.geo.addCircleArc(p5_neg, p0, p6_down)
    lOutDown4a = gmsh.model.geo.addLine(p5_neg, p2_neg)
    lOutDown4b = gmsh.model.geo.addLine(p4_down, p6_down)
    outerLoopDown4 = gmsh.model.geo.addCurveLoop(
        [lOutDown4a, arcOuterDown4, lOutDown4b, -arcInterfaceDown4]
    )
    outerSurfDown4 = gmsh.model.geo.addPlaneSurface([outerLoopDown4])

    if a == rInner:
        lInDown4a = gmsh.model.geo.addLine(p0, p5_neg)
        lInDown4b = gmsh.model.geo.addLine(p6_down, p0)
        innerLoopDown4 = gmsh.model.geo.addCurveLoop([lInDown4a, arcInterfaceDown4, lInDown4b])
        innerSurfDown4 = gmsh.model.geo.addPlaneSurface([innerLoopDown4])
        innerBulkSurfDown4 = None
        set_transfinite(
            [lInDown4a, lInDown4b, lOutDown4a, lOutDown4b],
            [arcOuterDown4, arcInterfaceDown4],
            [(innerSurfDown4, "Right"), (outerSurfDown4, "Left")],
            nDiv_radial,
            nDiv_curved,
            Bumpcoef,
        )
    else:
        arcInnerDown4 = gmsh.model.geo.addCircleArc(p1_neg, p0, p3_down)
        lInDown4a = gmsh.model.geo.addLine(p1_neg, p5_neg)
        lInDown4b = gmsh.model.geo.addLine(p6_down, p3_down)
        innerLoopDown4 = gmsh.model.geo.addCurveLoop(
            [-lInDown4a, -arcInterfaceDown4, -lInDown4b, arcInnerDown4]
        )
        innerSurfDown4 = gmsh.model.geo.addPlaneSurface([innerLoopDown4])
        set_transfinite(
            [lInDown4a, lInDown4b, lOutDown4a, lOutDown4b],
            [arcOuterDown4, arcInnerDown4, arcInterfaceDown4],
            [(innerSurfDown4, "Right"), (outerSurfDown4, "Left")],
            nDiv_radial,
            nDiv_curved,
            Bumpcoef,
        )
        lInBulkDown4a = gmsh.model.geo.addLine(p0, p1_neg)
        lInBulkDown4b = gmsh.model.geo.addLine(p3_down, p0)
        innerBulkLoopDown4 = gmsh.model.geo.addCurveLoop(
            [lInBulkDown4a, arcInnerDown4, lInBulkDown4b]
        )
        innerBulkSurfDown4 = gmsh.model.geo.addPlaneSurface([innerBulkLoopDown4])

    # Outer bulk surface for wedge 4.
    if a + rInner == rOuter:
        outerBulkSurfDown4 = None
    else:
        arcOuterBulkDown4 = gmsh.model.geo.addCircleArc(p7_neg, p0, p8_down)
        lOutBulkDown4a = gmsh.model.geo.addLine(p2_neg, p7_neg)
        lOutBulkDown4b = gmsh.model.geo.addLine(p8_down, p4_down)
        outerBulkLoopDown4 = gmsh.model.geo.addCurveLoop(
            [-lOutBulkDown4b, -arcOuterBulkDown4, -lOutBulkDown4a, arcOuterDown4]
        )
        outerBulkSurfDown4 = gmsh.model.geo.addPlaneSurface([outerBulkLoopDown4])

    gmsh.model.geo.synchronize()
    inner_vols4, outer_vols4 = create_wedge_volumes(
        innerSurfDown4,
        outerSurfDown4,
        innerBulkSurfDown4,
        outerBulkSurfDown4,
        revolveAxis,
        revolveCenter,
        angleSweep,
        nDiv_radial,
        nDiv_curved,
        Bumpcoef,
    )
    allInnerVols += [vol for vol in inner_vols4 if vol is not None]
    allOuterVols += [vol for vol in outer_vols4 if vol is not None]

    # ------------------ Physical Groups and Interfaces ------------------
    innerGroupTag = gmsh.model.addPhysicalGroup(3, allInnerVols)
    gmsh.model.setPhysicalName(3, innerGroupTag, InnerMaterialName)

    outerGroupTag = gmsh.model.addPhysicalGroup(3, allOuterVols)
    gmsh.model.setPhysicalName(3, outerGroupTag, OuterMaterialName)

    # Determine interface surfaces between inner and outer volumes.
    ifaceTags = []
    for vIn in allInnerVols:
        bndIn = gmsh.model.getBoundary([(3, vIn)], oriented=False, recursive=False)
        inSurfs = {s for (d, s) in bndIn if d == 2}
        for vOut in allOuterVols:
            bndOut = gmsh.model.getBoundary([(3, vOut)], oriented=False, recursive=False)
            outSurfs = {s for (d, s) in bndOut if d == 2}
            common = inSurfs.intersection(outSurfs)
            if common:
                for iface in common:
                    if iface not in ifaceTags:
                        ifaceTags.append(iface)

    outerBoundarySurfs = set()
    for vOut in allOuterVols:
        bndOut = gmsh.model.getBoundary([(3, vOut)], oriented=False, recursive=False)
        candidateSurfs = {s for (d, s) in bndOut if d == 2}
        candidateSurfs -= set(ifaceTags)
        for s in candidateSurfs:
            upVols, downVols = gmsh.model.getAdjacencies(2, s)
            if len(upVols) == 1:
                outerBoundarySurfs.add(s)

    outerBoundaryTag = gmsh.model.addPhysicalGroup(2, list(outerBoundarySurfs))
    gmsh.model.setPhysicalName(2, outerBoundaryTag, OuterBoundaryName)

    gmsh.model.geo.synchronize()
    interfaceTag = gmsh.model.addPhysicalGroup(2, ifaceTags)
    gmsh.model.setPhysicalName(2, interfaceTag, InterfaceName)

    gmsh.model.geo.synchronize()
