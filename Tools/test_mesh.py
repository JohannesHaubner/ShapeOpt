from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

mesh = Mesh()
with XDMFFile("./Output/Mesh_Generation/mesh_triangles.xdmf") as infile:
    infile.read(mesh)

mesh_global = Mesh(MPI.comm_self)
with XDMFFile(MPI.comm_self, "./Output/Mesh_Generation/mesh_triangles.xdmf") as infile:
    infile.read(mesh_global)

SpaceV = FunctionSpace(mesh, "CG", 1)

SpaceVg = FunctionSpace(mesh_global, "CG", 1)

F_global = Function(SpaceVg)
F_global = interpolate(Expression("x[0]", degree = 1), SpaceVg)

file = XDMFFile(MPI.comm_self, './Output/Tests/SettingsMesh/Fglobal.xdmf')
file.write(F_global)




def meshglobal_to_mesh_spaceV_spaceVg():
    # construct array with [indicees, xvalues, yvalues] (for global and gathered local mesh, on each process)
    f1 = Expression('x[0]', degree=1)
    f1_l = interpolate(f1, SpaceV)
    f1_g = interpolate(f1, SpaceVg).vector().get_local()
    f2 = Expression('x[1]', degree=1)
    f2_l = interpolate(f2, SpaceV)
    f2_g = interpolate(f2, SpaceVg).vector().get_local()
    f1_lg = f1_l.vector().gather(range(f1_l.vector().size()))
    f2_lg = f2_l.vector().gather(range(f2_l.vector().size()))

    ndofs = np.size(f1_g)

    data_global = np.column_stack((f1_g, f2_g, range(ndofs)))
    data_glocal = np.column_stack((f1_lg, f2_lg, range(ndofs)))

    # sort such that first row is increasing, if first row is the same second row is increasing

    data_global = data_global[data_global[:,1].argsort(kind='mergesort')]
    data_global = data_global[data_global[:,0].argsort(kind='mergesort')]
    data_glocal = data_glocal[data_glocal[:, 1].argsort(kind='mergesort')]
    data_glocal = data_glocal[data_glocal[:, 0].argsort(kind='mergesort')]

    # glocal_to_global-map
    gloc_glob = np.column_stack((data_global[:,2],data_glocal[:,2]))
    gloc_glob_sort1 = gloc_glob[gloc_glob[:,0].argsort(kind='mergesort')]
    global_to_glocal_map = gloc_glob_sort1[:,1]
    gloc_glob_sort2 = gloc_glob[gloc_glob[:, 1].argsort(kind='mergesort')]
    glocal_to_global_map = gloc_glob_sort2[:, 0]

    return global_to_glocal_map.astype(int), glocal_to_global_map.astype(int)

global_to_glocal_map, glocal_to_global_map = meshglobal_to_mesh_spaceV_spaceVg()

print(global_to_glocal_map[100])
print(glocal_to_global_map[2579])

def global_to_local(Fg):
    gathered_local = Fg.vector().get_local()
    print(gathered_local)
    f = Function(SpaceV)
    dof = SpaceV.dofmap()

    imin, imax = dof.ownership_range()
    f.vector().set_local(gathered_local[glocal_to_global_map[imin:imax]])
    f.vector().apply("")
    return f

def local_to_global(f):
    Fg = Function(SpaceVg)
    ndof = np.size(Fg.vector().get_local())
    gathered_local = f.vector().gather(range(ndof))
    Fg.vector().set_local(gathered_local[global_to_glocal_map])
    Fg.vector().apply("")
    return Fg



f = global_to_local(F_global)

file = XDMFFile('./Output/Tests/SettingsMesh/Flocal.xdmf')
file.write(f)

Fg = local_to_global(f)
file = XDMFFile(MPI.comm_self, './Output/Tests/SettingsMesh/Fglocal.xdmf')
file.write(Fg)
