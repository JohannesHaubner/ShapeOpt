from dolfin import *
import dolfin as df
from dolfin_adjoint import *
import numpy as np
from pyadjoint.overloaded_type import create_overloaded_object
from shapeopt.Tools.subdomains import SubMeshCollection, transfer_to_subfunc
#import mpi4py as MPI
#from dolfin_adjoint import *
import matplotlib.pyplot as plt

import mpi4py as MPI4py
comm = MPI4py.MPI.COMM_WORLD
id = comm.Get_rank()


class Initialize_Mesh_and_FunctionSpaces():
    def __init__(self, path_mesh, load_mesh=False):
      if load_mesh:
        stri = path_mesh + "/mesh_triangles_new.xdmf"
        stri2 = path_mesh + "/facet_mesh_new.xdmf"
        stri3 = path_mesh + "/domains_new.xdmf"
      else:
        stri = path_mesh + "/mesh_triangles.xdmf"
        stri2 = path_mesh + "/facet_mesh.xdmf"

      # load mesh
      mesh = Mesh(MPI.comm_world)
      with XDMFFile(stri) as infile:
        infile.read(mesh)

      if load_mesh:
          mvc = MeshValueCollection("size_t", mesh, 1)
          with XDMFFile(stri3) as infile:
              infile.read(mvc)
      else:
          mvc = MeshValueCollection("size_t", mesh, 2)
          with XDMFFile(stri) as infile:
            infile.read(mvc, "name_to_read")
      domains = cpp.mesh.MeshFunctionSizet(mesh, mvc)
      dfile = File("./Output/Tests/ForwardEquation/domains.pvd")
      dfile << domains

      # read boundary parts
      mfile = File("./Output/Tests/ForwardEquation/mesh.pvd")
      mfile << mesh
      if load_mesh:
          mvc = MeshValueCollection("size_t", mesh, 1)
          with XDMFFile(stri2) as infile:
              infile.read(mvc)
      else:
          mvc = MeshValueCollection("size_t", mesh, 1)
          with XDMFFile(stri2) as infile:
            infile.read(mvc, "name_to_read")
      boundaries = MeshFunction('size_t', mesh, mvc)
      #xdmf2 = XDMFFile("./Output/Mesh_Generation/facet_mesh_new.xdmf")
      #xdmf2.write(boundaries)
      #mvc = MeshValueCollection("size_t", mesh, 1)
      #with XDMFFile("./Output/Mesh_Generation/facet_mesh_new.xdmf") as infile:
      #    infile.read(mvc)
      #boundaries = MeshFunction('size_t', mesh, mvc)
      #exit(0)

      self.boundaries = boundaries
      self.domains = domains

      # create pyadjoint mesh
      mesh = create_overloaded_object(mesh)
      mvc = MeshValueCollection("size_t", mesh, 1)
      new_boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)
      new_boundaries.set_values(boundaries.array())
      boundaries = new_boundaries



      mesh_global = Mesh(MPI.comm_self)
      with XDMFFile(MPI.comm_self, stri) as infile:
        infile.read(mesh_global)

      if load_mesh and domains:
          mvc_glob = MeshValueCollection("size_t", mesh_global, 1)
          with XDMFFile(MPI.comm_self, stri3) as infile:
              infile.read(mvc_glob)
      else:
          mvc_glob = MeshValueCollection("size_t", mesh_global, 2)
          with XDMFFile(MPI.comm_self, stri) as infile:
            infile.read(mvc_glob, "name_to_read")
      domains_global = cpp.mesh.MeshFunctionSizet(mesh_global, mvc_glob)
      dfile = File("./Output/Tests/ForwardEquation/domains_global.pvd")
      dfile << domains_global

      # full boundary information on each process
      mvc2 = MeshValueCollection("size_t", mesh_global, 1)
      if load_mesh:
          with XDMFFile(MPI.comm_self, stri2) as infile:
              infile.read(mvc2)
      else:
          with XDMFFile(MPI.comm_self, stri2) as infile:
            infile.read(mvc2, "name_to_read")
      boundaries_global = cpp.mesh.MeshFunctionSizet(mesh_global, mvc2)

      # save to pvd file for testing
      bdfile = File("./Output/Tests/ForwardEquation/boundary.pvd")
      bdfile << boundaries

      # load mesh parameters
      params = np.load(path_mesh + '/params.npy', allow_pickle='TRUE').item()

      # define design boundary mesh on each process

      # first go to fluid mesh in order to allow the interface to be a design boundary
      subdomain_boundaries = params["subdomain_boundaries"]
      subdomain_labels = params["subdom_labels"]
      boundary_labels = params["bdry_labels"]

      #  call SubMeshCollection
      meshes_local = SubMeshCollection(domains, boundaries, subdomain_labels, boundary_labels, subdomain_boundaries)

      markers_fluid_local = meshes_local.subdomains["fluid"].boundaries

      fluid_mesh_local = markers_fluid_local.mesh().mesh

      #  call SubMeshCollection
      meshes = SubMeshCollection(domains_global, boundaries_global, subdomain_labels, boundary_labels, subdomain_boundaries)

      markers_fluid = meshes.subdomains["fluid"].boundaries

      fluid_mesh = markers_fluid.mesh().mesh
      fluid_markers = markers_fluid.mesh().boundaries

      bmesh = BoundaryMesh(fluid_mesh, "exterior")
      dofs = bmesh.entity_map(1)

      # create MeshFunctionSizet on boundary
      bmvc = MeshValueCollection("size_t", bmesh, 1)
      bboundaries = cpp.mesh.MeshFunctionSizet(bmesh, bmvc)

      # write boundaries[dof of facet in mesh] into bboundaries[dof of facet in bmesh]
      bnum = bmesh.num_vertices()
      bsize = bboundaries.size()

      for i in range(bnum):
          bboundaries.set_value(i, fluid_markers[dofs[i]])

      # boundary mesh and submesh
      #bmesh = BoundaryMesh(mesh_global, "exterior")

      # get entity map of facets, dof of facet of boundary mesh to dof of facet of mesh
      #dofs = bmesh.entity_map(1)

      # create MeshFunctionSizet on boundary
      #bmvc = MeshValueCollection("size_t", bmesh, 1)
      #bboundaries = cpp.mesh.MeshFunctionSizet(bmesh, bmvc)

      # write boundaries[dof of facet in mesh] into bboundaries[dof of facet in bmesh]
      #bnum = bmesh.num_vertices()
      #bsize = bboundaries.size()

      #for i in range(bnum):
      #  bboundaries.set_value(i, boundaries_global[dofs[i]])

      # write to pvd-file for testing
      bdfile = File(MPI.comm_self, "./Output/Tests/ForwardEquation/bboundary.pvd")
      bdfile << bboundaries

      # create design-boundary mesh
      dmesh = MeshView.create(bboundaries, params["design"])
      print('design boundary mesh created.........................................')


      # normal vector on mesh
      n = FacetNormal(mesh)
      self.n = n
      
      # define function spaces
      self.V = FunctionSpace(mesh, "CG", 1)
      #Vbl = FunctionSpace(bmesh_local, "CG", 1)
      #Vbln = VectorFunctionSpace(bmesh_local, "CG", 1)
      Vg = FunctionSpace(mesh_global, "CG",1)
      Vfg = FunctionSpace(fluid_mesh, "CG", 1)
      Vgn = VectorFunctionSpace(mesh_global, "CG", 1)
      Vfgn = VectorFunctionSpace(fluid_mesh, "CG", 1)
      Vb = FunctionSpace(bmesh, "CG", 1)
      self.Vd = FunctionSpace(dmesh, "CG", 1)
      self.Vn = VectorFunctionSpace(mesh, "CG", 1)
      Vbn = VectorFunctionSpace(bmesh, "CG", 1)
      self.Vdn = VectorFunctionSpace(dmesh, "CG", 1)

      dof_map_fluid_full = self.__dof_maps_fluid_full(Vfg, Vg)

      #a = interpolate(Expression("x[0]*x[1]", degree =2), Vfg)
      #b = Function(Vg)
      #b.vector()[dof_map_fluid_full] = a.vector()[:]
      #b.vector().apply("")
      ##from IPython import embed; embed()
      #xdmffile = XDMFFile(MPI.comm_self, "./Output/Tests/SettingsMesh/TEST.xdmf")
      #xdmffile.write_checkpoint(b, 'p', XDMFFile.Encoding.HDF5)
      
      self.mesh = mesh
      self.dmesh = dmesh
      self.params = params

      self.ds = ds

      # dof-maps between V and Vg
      global_to_glocal_map, glocal_to_global_map = self.__meshglobal_to_mesh__(mesh_global)

      # dof-maps between V and Vb
      Vb_to_V_map = self.__Vb_to_V(Vfg, Vb, global_to_glocal_map, dof_map_fluid_full)

      # dof-maps between V and Vd
      self.Vd_to_V_map = self.__Vd_to_V(Vb, Vb_to_V_map)

      #self.__test_Vb_to_V(Vfg, Vb, global_to_glocal_map, dof_map_fluid_full)
      #self.__test_Vdn_to_Vn()
      #self.__test_Vd_to_V()
      #self.__test_V_to_Vd()
      #self.__test_Vd_to_V_to_Vd()
      #self.__test_Vn_to_Vdn()
      #exit(0)

      # normal on design boundary
      v = TestFunction(self.Vn)
      n_mesh = assemble_mixed(inner(n, v)*ds(fluid_mesh_local))
      n_norm = assemble_mixed(inner(Constant(("1.0","1.0")), v) * ds(fluid_mesh_local)) # to norm n_mesh
      normal = Function(self.Vn)
      normal.vector().set_local(n_mesh.get_local())
      normal.vector().apply("")
      nnorm = Function(self.Vn)
      nnorm.vector().set_local(n_norm.get_local())
      nnorm.vector().apply("")
      dnormalf = self.Vn_to_Vdn(normal)
      n_normf = self.Vn_to_Vdn(nnorm)
      n_normf.vector().apply("")
      normed_normal = [dnormalf.vector().get_local()[i]/n_normf.vector().get_local()[i] for i in range(np.size(dnormalf.vector().get_local()))]
      self.dnormalf = Function(self.Vdn)
      self.dnormalf.vector().set_local(normed_normal)
      self.dnormalf.vector().apply("")

    def vec_to_Vn(self, x):
        """
        x contains gathered dofs of Vn-function and writes them into a Vn function
        """
        v = Function(self.Vn)
        dof = self.Vn.dofmap()
        imin, imax = dof.ownership_range()
        v.vector().set_local(x[imin:imax])
        v.vector().apply("")
        return v

    def vec_to_Vd(self, x):
        """ takes a vector with all dofs and writes them on each process to a function v """
        SpaceVd = self.Vd
        fx = Function(SpaceVd)
        fx.vector().set_local(x)
        fx.vector().apply("")
        return fx

    def Vd_to_vec(self, v):
        SpaceVd = self.Vd
        dof = SpaceVd.dofmap()
        imin, imax = dof.ownership_range()
        xvalues = v.vector().get_local()
        x = xvalues
        return x

    def Vd_to_V(self, x):
        """
        maps Vd-function x to function in V
        """
        v = Function(self.V)
        dofsv = np.zeros(v.vector().size())
        dofsv[self.Vd_to_V_map] = x.vector().get_local()
        # gathered local dofs to processes
        dof = self.V.dofmap()
        imin, imax = dof.ownership_range()
        v.vector().set_local(dofsv[imin:imax])
        v.vector().apply("")
        return v

    def V_to_Vd(self, x):
        """
        maps V-function x to a function in Vd
        """
        vd = Function(self.Vd)
        # gather dofs
        ndof = x.vector().size()
        gathered_local = x.vector().gather(range(ndof))

        # write correct dofs into x
        vd.vector().set_local(gathered_local[self.Vd_to_V_map])
        vd.vector().apply("")
        return vd


    def Vn_to_Vdn(self, x):
        """
        maps Vn-function x to a function in Vdn
        """
        x_is = x.split(deepcopy=True)
        vbs = []
        for v_i in x_is:
            vbs.append(self.V_to_Vd(v_i))
        split_to_vec = FunctionAssigner(self.Vdn, [vi.function_space() for vi in vbs])
        vdn = Function(self.Vdn)
        split_to_vec.assign(vdn, vbs)
        return vdn

    def Vdn_to_Vn(self, x):
        """
        maps Vdn-function to a function in Vn
        """
        vb_is = x.split(deepcopy=True)
        vs = []
        for vb_i in vb_is:
            vs.append(self.Vd_to_V(vb_i))
        split_to_vec = FunctionAssigner(self.Vn, [vi.function_space() for vi in vs])
        vn = Function(self.Vn)
        split_to_vec.assign(vn, vs)
        return vn

    def __V_to_Vg(self, v, Vg, glocal_to_global_map):
        ndof = v.vector().size()
        gathered_local = v.vector().gather(range(ndof))
        defog = Function(Vg)
        defog.vector().set_local(gathered_local[glocal_to_global_map])
        return defog

    def __Vn_to_Vgn(self, v, Vg, Vgn, glocal_to_global_map):
        vb_is = v.split(deepcopy=True)
        vs = []
        for vb_i in vb_is:
            vs.append(self.__V_to_Vg(vb_i, Vg, glocal_to_global_map))
        split_to_vec = FunctionAssigner(Vgn, [vi.function_space() for vi in vs])
        vn = Function(Vgn)
        split_to_vec.assign(vn, vs)
        return vn


    def __meshglobal_to_mesh__(self, mesh_global):
        " returns dof maps between V and Vg "
        SpaceV = self.V
        SpaceVg = FunctionSpace(mesh_global, "CG",1)

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

        data_global = data_global[data_global[:, 1].argsort(kind='mergesort')]
        data_global = data_global[data_global[:, 0].argsort(kind='mergesort')]
        data_glocal = data_glocal[data_glocal[:, 1].argsort(kind='mergesort')]
        data_glocal = data_glocal[data_glocal[:, 0].argsort(kind='mergesort')]

        # glocal_to_global-map
        gloc_glob = np.column_stack((data_global[:, 2], data_glocal[:, 2]))
        gloc_glob_sort1 = gloc_glob[gloc_glob[:, 0].argsort(kind='mergesort')]
        global_to_glocal_map = gloc_glob_sort1[:, 1]
        gloc_glob_sort2 = gloc_glob[gloc_glob[:, 1].argsort(kind='mergesort')]
        glocal_to_global_map = gloc_glob_sort2[:, 0]

        return global_to_glocal_map.astype(int), glocal_to_global_map.astype(int)

    def __global_to_local(self, Fg, glocal_to_global_map):
        " maps Function from Vg to V "
        SpaceV = self.V

        gathered_local = Fg.vector().get_local()
        #print(gathered_local))

        f = Function(SpaceV)
        dof = SpaceV.dofmap()

        imin, imax = dof.ownership_range()
        f.vector().set_local(gathered_local[glocal_to_global_map[imin:imax]])
        f.vector().apply("")
        return f

    def __local_to_global(self, f, global_to_glocal_map):
        " maps Function from V to Vg"
        SpaceVg = self.Vg

        Fg = Function(SpaceVg)
        ndof = np.size(Fg.vector().get_local())
        gathered_local = f.vector().gather(range(ndof))
        Fg.vector().set_local(gathered_local[global_to_glocal_map])
        Fg.vector().apply("")
        return Fg

    def SyncSum(self,vec):
     """ Returns sum of vec over all mpi processes.
     
     Each vec vector must have the same dimension for each MPI process """
     
     comm = MPI.comm_world
     NormalsAllProcs = np.zeros(comm.Get_size()*len(vec), dtype=vec.dtype)
     comm.Allgather(vec, NormalsAllProcs)

     out = np.zeros(len(vec))
     for j in range(comm.Get_size()):
         out += NormalsAllProcs[len(vec)*j:len(vec)*(j+1)]
     return out
      
    def __V_to_Vb_map(self, Vg, Vb, glocal_to_global_map, fluid_full_map):

      bmesh = Vb.mesh()
      mesh_global = Vg.mesh()

      v = Function(self.V)
      ndofv = v.vector().size()
      rnd = np.array(range(ndofv))

      # map from mesh_global to bmesh
      ggm = glocal_to_global_map[rnd] # local to global
      ggm = ggm[fluid_full_map] # global to fluid

      # We use a dof->Vertex mapping to create a global array with all DOF values ordered by mesh vertices
      DofToVert = dof_to_vertex_map(v.function_space())

      VGlobal = np.zeros(v.vector().size())
      vec = v.vector().get_local()
      for i in range(len(vec)):
          Vert = MeshEntity(mesh_global,0,DofToVert[i])
          globalIndex = Vert.global_index()
          VGlobal[i] = ggm[globalIndex]

      # Use the inverse mapping to see the DOF values of a boundary function
      surface_space = Vb
      surface_function = Function(surface_space)
      mapa = bmesh.entity_map(0)
      DofToVert = dof_to_vertex_map(surface_space)
      
      LocValues = surface_function.vector().get_local()
      for i in range(len(LocValues)):
          VolVert = MeshEntity(mesh,0,mapa[int(DofToVert[i])])
          GlobalIndex = VolVert.global_index()
          LocValues[i] = VGlobal[GlobalIndex]

      return LocValues
  
    def __Vb_to_V(self, Vfg, Vb, global_to_glocal_map, fluid_full_map):
      """ Take a CG1 function f defined on bmesh and return a volume vector with same 
      values on the boundary but zero in volume
      """
      SpaceV = self.V
      SpaceVg = Vfg
      SpaceB = Vb

      f = Function(SpaceB)

      # assign f to function in Fg
      bmesh = Vb.mesh()
      mapb = bmesh.entity_map(0)
      d2v = dof_to_vertex_map(SpaceB)
      v2d = vertex_to_dof_map(SpaceVg)


      Vb_to_Vfg_map = np.zeros(np.size(f.vector().get_local()))
      for i in range(np.size(f.vector().get_local())):
          GVertID = Vertex(bmesh, d2v[i]).index()  # Local Vertex ID for given dof on boundary mesh
          PVertID = mapb[GVertID]  # Local Vertex ID of parent mesh
          PDof = v2d[PVertID]
          Vb_to_Vfg_map[i] = int(PDof)
      
      Vb_to_V_map_new = [global_to_glocal_map[fluid_full_map[int(c)]] for c in Vb_to_Vfg_map]

      return Vb_to_V_map_new


    def __test_Vb_to_V(self, Vfg, Vb, global_to_glocal_map, fluid_full_map):
        print(Vb)
        f = Function(Vb)
        n = np.size(f.vector().get_local())
        f.vector().set_local(np.ones(n))
        f.vector().apply("")
        Vb_to_V_map = self.__Vb_to_V(Vfg, Vb, global_to_glocal_map, fluid_full_map)

        p = Function(self.V)
        values = np.zeros(p.vector().size())
        values[Vb_to_V_map] = f.vector().get_local()

        dof = self.V.dofmap()
        imin, imax = dof.ownership_range()
        p.vector().set_local(values[imin:imax])
        p.vector().apply("")
        as_backend_type(p.vector()).vec().ghostUpdate()

        bdfile = File(MPI.comm_self, "./Output/Tests/SettingsMesh/Vb_to_V.pvd")
        bdfile << p
        pass
    
    def __dof_maps_fluid_full(self, V, V_full):
        # fluid mesh to global mesh
        v_full = Function(V_full)
        ndof_full = len(v_full.vector()[:])
        v_full.vector()[:] = range(ndof_full)

        v = transfer_to_subfunc(v_full, V)

        array = v.vector()[:].astype(int)
        return array

    def __Vd_to_V(self, Vb, Vb_to_V_map):
        """ Transfers a function from a MeshView submesh to its parent mesh """
        # Extract meshes
        mesh = Vb.mesh()
        submesh = self.Vd.mesh()

        # function
        f = Function(self.Vd)

        # Build cell mapping between sub and parent mesh
        cell_map = submesh.topology().mapping()[mesh.id()].cell_map()

        # Get cell dofmaps
        dofmap = self.Vd.dofmap()
        dofmap_full = Vb.dofmap()

        # Transfer dofs
        GValues = np.zeros(np.size(f.vector().get_local()))
        for c in cells(submesh):
            GValues[dofmap.cell_dofs(c.index())] = dofmap_full.cell_dofs(cell_map[c.index()])
        #GValuesnew = np.zeros(np.size(f.vector().get_local()))
        #for i in range(len(GValues)):
        #    GValuesnew[i] = int(Vb_to_V_map[int(GValues[i])])
        GValuesnew = [Vb_to_V_map[int(c)] for c in GValues]
        return GValuesnew

    def __test_Vd_to_V(self):
        f = interpolate(Expression("x[1]", degree = 2), self.Vd)
        #f = Function(self.Vd)
        #f.vector().set_local(np.asarray(range(np.size(f.vector().get_local()))))

        p = self.Vd_to_V(f)

        xdmffile = XDMFFile(MPI.comm_world, "./Output/Tests/SettingsMesh/Vd_to_V.xdmf")
        xdmffile.write_checkpoint(p, 'p', XDMFFile.Encoding.HDF5)
        pass

    def __test_V_to_Vd(self):
        f = interpolate(Expression("x[0]", degree = 2), self.V)

        p = self.V_to_Vd(f)

        xdmffile = XDMFFile(MPI.comm_world, "./Output/Tests/SettingsMesh/V_to_Vd.xdmf")
        xdmffile.write_checkpoint(p, 'p', XDMFFile.Encoding.HDF5)
        pass

    def __test_Vd_to_V_to_Vd(self):
        f = interpolate(Expression("x[0]", degree = 2), self.Vd)

        p = self.Vd_to_V(f)

        f = self.V_to_Vd(p)
        p = self.Vd_to_V(f)

        bdfile = XDMFFile(MPI.comm_world, "./Output/Tests/SettingsMesh/Vd_to_V_to_Vd.xdmf")
        bdfile.write_checkpoint(p, 'p', XDMFFile.Encoding.HDF5)
        pass

    def __test_Vn_to_Vdn(self):
        f = project(Expression(("x[0]", "pow(x[1]-x[0],2)"), element= self.Vn.ufl_element()),self.Vn)

        p = self.Vn_to_Vdn(f)

        xdmffile = XDMFFile(MPI.comm_world, "./Output/Tests/SettingsMesh/Vn_to_Vdn.xdmf")
        xdmffile.write_checkpoint(p, 'p', XDMFFile.Encoding.HDF5)
        pass

    def __test_Vdn_to_Vn(self):
        f = df.project(Expression(("x[0]", "pow(x[1]-x[0],2)"), element= self.Vdn.ufl_element()),self.Vdn)

        p = self.Vdn_to_Vn(f)

        xdmffile = XDMFFile("./Output/Tests/SettingsMesh/Vdn_to_Vn.xdmf")
        xdmffile.write_checkpoint(p, 'p', XDMFFile.Encoding.HDF5)
        pass


      
    def get_mesh(self):
      #print('load mesh............................................................')
      return self.mesh

  
    def get_design_boundary_mesh(self):
      #print('load design boundary mesh............................................')
      return self.dmesh
  
    def get_params(self):
      #print('load mesh parameters.................................................')
      return self.params
  
    def get_boundaries(self):
      #print('load boundary data...................................................')
      return self.boundaries

    def get_domains(self):
      #print('load domain data.....................................................')
      return self.domains
  
    def get_V(self):
      return self.V
  
    def get_Vd(self):
      return self.Vd
  
    def get_Vn(self):
      return self.Vn

    def get_Vdn(self):
      return self.Vdn

    def get_ds(self):
      return self.ds
  
    def get_dnormalf(self):
      return self.dnormalf
  
    def get_n(self):
      return self.n

