from dolfin import *
from pyadjoint import *
import Reduced_Objective.Stokes as ro_stokes
import Tools.save_load_obj as tool
import Control_to_Trafo.dof_to_trafo as ctt
import Tools.settings_mesh as tsm
import Constraints.volume as Cv
import Constraints.barycenter_ as Cb
import Constraints.determinant as Cd
import Ipopt.ipopt_solver_ as ipopt_so
import numpy as np

mesh = Mesh()
with XDMFFile("./Output/Mesh_Generation/mesh_triangles.xdmf") as infile:
    infile.read(mesh)
    mvc = MeshValueCollection("size_t", mesh, 1)
mfile = File("./Output/Tests/ForwardEquation/mesh.pvd")
mfile << mesh
with XDMFFile("./Output/Mesh_Generation/facet_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
    boundaries = cpp.mesh.MeshFunctionSizet(mesh,mvc)

params = np.load('./Mesh_Generation/params.npy', allow_pickle='TRUE').item()

dmesh = MeshView.create(boundaries, params["design"])
exit(0)

# do just on single process and then share with other proceess
#boundary mesh and submesh
bmesh = BoundaryMesh(mesh, "exterior")
# get entity map of facets, dof of facet of boundary mesh to dof of facet of mesh
dofs = bmesh.entity_map(1)
#create MeshFunctionSizet on boundary
bmvc = MeshValueCollection("size_t", bmesh, 1)
bboundaries = cpp.mesh.MeshFunctionSizet(bmesh, bmvc)
#write boundaries[dof of facet in mesh] into bboundaries[dof of facet in bmesh]
bnum = bmesh.num_vertices()
bsize = bboundaries.size()
for i in range(bnum):
    bboundaries.set_value(i, boundaries[dofs[i]])
dmesh = SubMesh(bmesh, bboundaries, params["design"])
print('design boundary mesh created.........................................')