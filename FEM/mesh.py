import gmsh, meshio
from netgen.read_gmsh import ReadGmsh   # or `import netgen.meshing as nm`
import netgen.meshing as nm
from Transfinite_Sphere import Transfinite_Sphere

__all__ = [
    "generate_sphere_mesh",  
]

def generate_sphere_mesh(
    a, r_inner, r_outer, h_band, h_outer,
    bumpcoef=1.0,
    inner_name="metal",
    outer_name="vacuum",
    interface_name="interface",
    mesh_filename=None,
    vol_filename=None
):
    gmsh.initialize()
    gmsh.model.add("Sphere")
    Transfinite_Sphere(
        a=a, rInner=r_inner, rOuter=r_outer,
        h_band=h_band, h_outer=h_outer, Bumpcoef=bumpcoef,
        InnerMaterialName=inner_name,
        OuterMaterialName=outer_name,
        InterfaceName=interface_name,
        OuterBoundaryName=outer_name
    )
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion",2.2)

    if mesh_filename is None:
        mesh_filename = f"sphere_h_{h_band}_R_{r_outer}.msh"
    if vol_filename is None:
        vol_filename = mesh_filename.replace(".msh", ".vol")

    gmsh.write(mesh_filename)
    gmsh.finalize()

    # this is the “real” mesh for your FEM
    mesh3d = meshio.read(mesh_filename)
    meshio.write(vol_filename, mesh3d, file_format="netgen")
    ng_mesh = nm.Mesh(dim=3)
    ng_mesh.Load(vol_filename)

    return mesh3d, ng_mesh
