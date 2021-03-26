#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 08:39:51 2020

@author: Johannes Haubner
"""
from dolfin import *
import numpy as np
from pyadjoint.overloaded_type import create_overloaded_object
#import mpi4py as MPI
#from dolfin_adjoint import *
import matplotlib.pyplot as plt

class Initialize_Mesh_and_FunctionSpaces():
    def __init__(self, mesh = None, boundaries = None, params = None):
      #load mesh
      #stop_annotating()
      if mesh == None:
          mesh = Mesh()
          with XDMFFile("./Output/Mesh_Generation/mesh_triangles.xdmf") as infile:
            infile.read(mesh)
            mvc = MeshValueCollection("size_t", mesh, 1)
          mesh = create_overloaded_object(mesh)
          mfile = File("./Output/Tests/ForwardEquation/mesh.pvd")
          mfile << mesh
          with XDMFFile("./Output/Mesh_Generation/facet_mesh.xdmf") as infile:
            infile.read(mvc, "name_to_read")
            boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)

          mesh_global = Mesh(MPI.comm_self)
          with XDMFFile(MPI.comm_self, "./Output/Mesh_Generation/mesh_triangles.xdmf") as infile:
            infile.read(mesh_global)

          mvc2 = MeshValueCollection("size_t", mesh_global, 1)
          with XDMFFile(MPI.comm_self, "./Output/Mesh_Generation/facet_mesh.xdmf") as infile:
            infile.read(mvc2, "name_to_read")
          boundaries_global = cpp.mesh.MeshFunctionSizet(mesh_global, mvc2)

          bdfile = File("./Output/Tests/ForwardEquation/boundary.pvd")
          bdfile << boundaries
          params = np.load('./Mesh_Generation/params.npy', allow_pickle='TRUE').item()

      else:
          raise('Not implemented yet')
          new_mesh = Mesh(mesh)
          mvc = MeshValueCollection("size_t", mesh, 1)
          new_boundaries =cpp.mesh.MeshFunctionSizet(new_mesh,mvc)
          new_boundaries.set_values(boundaries.array())
          mesh = new_mesh
          boundaries = new_boundaries
          params = params

      print('mesh created.........................................................')
      #define design boundary mesh

      #boundary mesh and submesh
      bmesh = BoundaryMesh(mesh_global, "exterior")
      bmesh_local = BoundaryMesh(mesh, "exterior")

      # get entity map of facets, dof of facet of boundary mesh to dof of facet of mesh
      dofs = bmesh.entity_map(1)

      #create MeshFunctionSizet on boundary
      bmvc = MeshValueCollection("size_t", bmesh, 1)
      bboundaries = cpp.mesh.MeshFunctionSizet(bmesh, bmvc)

      #write boundaries[dof of facet in mesh] into bboundaries[dof of facet in bmesh]
      bnum = bmesh.num_vertices()
      bsize = bboundaries.size()

      for i in range(bnum):
        bboundaries.set_value(i, boundaries_global[dofs[i]])

      bdfile = File(MPI.comm_self, "./Output/Tests/ForwardEquation/bboundary.pvd")
      bdfile << bboundaries

      dmesh = MeshView.create(bboundaries, params["design"])
      print('design boundary mesh created.........................................')

      # normal vector on mesh
      self.n = FacetNormal(mesh)
      
      # define function spaces
      self.V = FunctionSpace(mesh, "CG", 1)
      self.Vbl = FunctionSpace(bmesh_local, "CG", 1)
      self.Vbln = VectorFunctionSpace(bmesh_local, "CG", 1)
      self.Vg = FunctionSpace(mesh_global, "CG",1)
      self.Vb = FunctionSpace(bmesh, "CG", 1)
      self.Vd = FunctionSpace(dmesh, "CG", 1)
      self.Vn = VectorFunctionSpace(mesh, "CG", 1)
      self.Vbn = VectorFunctionSpace(bmesh, "CG", 1)
      self.Vdn = VectorFunctionSpace(dmesh, "CG", 1)
      dnormal = CellNormal(dmesh)
      self.dnormalf = project(dnormal, self.Vdn)
      
      self.mesh = mesh
      self.mesh_global = mesh_global
      self.bmesh = bmesh
      self.bmesh_local = bmesh_local
      self.dmesh = dmesh
      self.params = params
      self.boundaries = boundaries

      self.global_to_glocal_map, self.glocal_to_global_map = self.__meshglobal_to_mesh_spaceV_spaceVg__()

    def __meshglobal_to_mesh_spaceV_spaceVg__(self):
        " returns dof maps between V and Vg "
        SpaceV = self.V
        SpaceVg = self.Vg

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

    def global_to_local(self, Fg):
        " maps Function from Vg to V "
        SpaceV = self.V

        gathered_local = Fg.vector().get_local()
        #print(gathered_local)
        f = Function(SpaceV)
        dof = SpaceV.dofmap()

        imin, imax = dof.ownership_range()
        f.vector().set_local(gathered_local[self.glocal_to_global_map[imin:imax]])
        f.vector().apply("")
        return f

    def local_to_global(self, f):
        " maps Function from V to Vg"
        SpaceVg = self.Vg

        Fg = Function(SpaceVg)
        ndof = np.size(Fg.vector().get_local())
        gathered_local = f.vector().gather(range(ndof))
        Fg.vector().set_local(gathered_local[self.global_to_glocal_map])
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
 
    def vec_to_Vd(self,x):
      """ takes a vector with all dofs and writes them in parallel to a function v """
      SpaceVd = self.Vd
      fx = Function(SpaceVd)
      dof = SpaceVd.dofmap()
      
      imin, imax = dof.ownership_range()
      fx.vector().set_local(x[imin:imax])
      fx.vector().apply("")
      return fx
  
    def Vd_to_vec(self,v):
      SpaceVd = self.Vd
      dof = SpaceVd.dofmap()
      imin, imax = dof.ownership_range()
      xvalues = np.zeros(v.vector().size())
      for i in range(v.vector().local_size()):
          xvalues[imin:imax] = v.vector().get_local()
      x = self.SyncSum(xvalues)
      return x
      
    def V_to_Vb(self, v):
      """ Returns a boundary interpolation of the CG1-function v
      see fenicsproject.discourse.group/t/how-tomap-dofs-of-vector-functions...
      """
      # map from mesh_global to mesh
      v = self.local_to_global(v)

      mesh = self.mesh_global
      # We use a dof->Vertex mapping to create a global array with all DOF values ordered by mesh vertices
      DofToVert = dof_to_vertex_map(v.function_space())
      #print(DofToVert)
      VGlobal = np.zeros(v.vector().size())
      
      vec = v.vector().get_local()
      for i in range(len(vec)):
          Vert = MeshEntity(mesh,0,DofToVert[i])
          globalIndex = Vert.global_index()
          VGlobal[globalIndex] = vec[i]
      VGlobal = self.SyncSum(VGlobal)
      #print(VGlobal)
      # Use the inverse mapping to see the DOF values of a boundary function
      surface_space = FunctionSpace(self.bmesh, "CG", 1)
      surface_function = Function(surface_space)
      mapa = self.bmesh.entity_map(0)
      DofToVert = dof_to_vertex_map(surface_space)
      
      LocValues = surface_function.vector().get_local()
      for i in range(len(LocValues)):
          VolVert = MeshEntity(mesh,0,mapa[int(DofToVert[i])])
          GlobalIndex = VolVert.global_index()
          LocValues[i] = VGlobal[GlobalIndex]
          
      surface_function.vector().set_local(LocValues)
      surface_function.vector().apply('')
      return surface_function
  
    def Vb_to_V(self, f):
      """ Take a CG1 function f defined on bmesh and return a volume vector with same 
      values on the boundary but zero in volume
      """
      SpaceV = self.V
      SpaceVg = self.Vg
      SpaceB = self.Vb

      # assign f to function in Fg
      mapb = self.bmesh.entity_map(0)
      d2v = dof_to_vertex_map(SpaceB)
      v2d = vertex_to_dof_map(SpaceVg)

      dof = SpaceV.dofmap()
      Fg = Function(SpaceVg)
      GValues = np.zeros(Fg.vector().size())

      for i in range(f.vector().local_size()):
          GVertID = Vertex(self.bmesh, d2v[i]).index()  # Local Vertex ID for given dof on boundary mesh
          PVertID = mapb[GVertID]  # Local Vertex ID of parent mesh
          PDof = v2d[PVertID]  # Dof on parent mesh
          value = f.vector()[i]  # Value on local processor
          GValues[PDof] = value
      #GValues = self.SyncSum(GValues)
      Fg.vector().set_local(GValues)
      Fg.vector().apply("")
      # works
      #outfile_1 = XDMFFile(MPI.comm_self, "Output/Tests/SettingsMesh/Vdn_to_Vn.xdmf")
      #outfile_1.write(Fg)
      #outfile_1.close()

      F = self.global_to_local(Fg)

      return F
  
    def Vn_to_Vbn(self, vn):
      """ Take function vn in Vn and interpolate at boundary
      """
      v_is = vn.split(deepcopy = True)
      vbs = []
      for v_i in v_is:
          vbs.append(self.V_to_Vb(v_i))
      split_to_vec = FunctionAssigner(self.Vbn, [vi.function_space() for vi in vbs])
      vbn = Function(self.Vbn)
      split_to_vec.assign(vbn, vbs)
      return vbn
  
    def Vbn_to_Vn(self,vbn):
      vb_is = vbn.split(deepcopy = True)
      vs = []
      for vb_i in vb_is:
          vs.append(self.Vb_to_V(vb_i))
      split_to_vec = FunctionAssigner(self.Vn, [vi.function_space() for vi in vs])
      vn = Function(self.Vn)
      split_to_vec.assign(vn, vs)
      return vn

    def transfer_subfunction_to_parent(self, f, V_full):
        """ Transfers a function from a MeshView submesh to its parent mesh """
        # Extract meshes
        mesh = V_full.mesh()
        submesh = f.function_space().mesh()

        # Build cell mapping between sub and parent mesh
        cell_map = submesh.topology().mapping()[mesh.id()].cell_map()

        # Get cell dofmaps
        dofmap = f.function_space().dofmap()
        dofmap_full = V_full.dofmap()

        # Transfer dofs
        f_full = Function(V_full)
        GValues = np.zeros(np.size(f_full.vector().get_local()))
        for c in cells(submesh):
            GValues[dofmap_full.cell_dofs(cell_map[c.index()])] = f.vector()[dofmap.cell_dofs(c.index())]
        f_full.vector().set_local(GValues)
        f_full.vector().apply("")
        return f_full

    def transfer_parentfunction_to_sub(self, f, V_sub):
        """ Restrict a function f on parent mesh to a function on the subspace V_sub"""
        # Extract meshes
        mesh = f.function_space().mesh()
        submesh = V_sub.mesh()

        # Build cell mapping between sub and parent mesh
        cell_map = submesh.topology().mapping()[mesh.id()].cell_map()

        # Get cell dofmaps
        dofmap = V_sub.dofmap()
        dofmap_full = f.function_space().dofmap()

        # transfer dofs
        f_sub = Function(V_sub)
        GValues = np.zeros(np.size(f_sub.vector().get_local()))

        for c in cells(submesh):
            GValues[dofmap.cell_dofs(c.index())] = f.vector()[dofmap_full.cell_dofs(cell_map[c.index()])]
        f_sub.vector().set_local(GValues)
        f_sub.vector().apply("")
        return f_sub

  
    def Vb_to_Vd(self,vb):
      dfunction = self.transfer_parentfunction_to_sub(vb, self.Vd)
      return dfunction
  
    def Vd_to_Vb(self,vd):
      F = self.transfer_subfunction_to_parent(vd, self.Vb)
      return F
  
    def Vbn_to_Vdn(self,vbn):
      v_is = vbn.split(deepcopy = True)
      vbs = []
      for v_i in v_is:
          vbs.append(self.Vb_to_Vd(v_i))
      split_to_vec = FunctionAssigner(self.Vdn, [vi.function_space() for vi in vbs])
      vdn = Function(self.Vdn)
      split_to_vec.assign(vdn, vbs)
      return vdn
  
    def Vdn_to_Vbn(self,vdn):
      vbn = self.transfer_subfunction_to_parent(vdn, self.Vbn)
      return vbn 
      
    def Vdn_to_Vn(self,xd):
      xb = self.Vdn_to_Vbn(xd)
      x = self.Vbn_to_Vn(xb)
      return x
  
    def Vn_to_Vdn(self,x):
      # x is a function in Vn; take the trace on dmesh to obtain xd
      xb = self.Vn_to_Vbn(x)
      xd = self.Vbn_to_Vdn(xb)
      return xd
      
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
  
    def get_V(self):
      return self.V
  
    def get_Vb(self):
      return self.Vb
  
    def get_Vd(self):
      return self.Vd
  
    def get_Vn(self):
      return self.Vn
  
    def get_Vbn(self):
      return self.Vbn
  
    def get_Vdn(self):
      return self.Vdn
  
    def get_dnormalf(self):
      return self.dnormalf
  
    def get_n(self):
      return self.n
  
    def get_Mlumpedm05(self):
      return self.M_lumped_m05

    def V_to_Vbl(self, v):
        """ Returns a boundary interpolation of the CG1-function v
        see fenicsproject.discourse.group/t/how-tomap-dofs-of-vector-functions...
        """
        mesh = self.mesh
        # We use a dof->Vertex mapping to create a global array with all DOF values ordered by mesh vertices
        DofToVert = dof_to_vertex_map(v.function_space())
        # print(DofToVert)
        VGlobal = np.zeros(v.vector().size())

        vec = v.vector().get_local()
        for i in range(len(vec)):
            Vert = MeshEntity(mesh, 0, DofToVert[i])
            globalIndex = Vert.global_index()
            VGlobal[globalIndex] = vec[i]
        VGlobal = self.SyncSum(VGlobal)
        # print(VGlobal)
        # Use the inverse mapping to see the DOF values of a boundary function
        surface_space = self.Vbl
        surface_function = Function(surface_space)
        mapa = self.bmesh_local.entity_map(0)
        DofToVert = dof_to_vertex_map(surface_space)

        LocValues = surface_function.vector().get_local()
        for i in range(len(LocValues)):
            VolVert = MeshEntity(mesh, 0, mapa[int(DofToVert[i])])
            GlobalIndex = VolVert.global_index()
            LocValues[i] = VGlobal[GlobalIndex]

        surface_function.vector().set_local(LocValues)
        surface_function.vector().apply('')
        return surface_function

    def Vbl_to_V(self, f):
        """ Take a CG1 function f defined on bmesh and return a volume vector with same
        values on the boundary but zero in volume
        """
        SpaceV = self.V
        SpaceB = self.Vbl

        F = Function(SpaceV)
        LocValues = np.zeros(F.vector().local_size())
        GValues = np.zeros(F.vector().size())

        mapb = self.bmesh_local.entity_map(0)
        d2v = dof_to_vertex_map(SpaceB)
        v2d = vertex_to_dof_map(SpaceV)

        dof = SpaceV.dofmap()
        imin, imax = dof.ownership_range()

        for i in range(f.vector().local_size()):
            GVertID = Vertex(self.bmesh, d2v[i]).index()  # Local Vertex ID for given dof on boundary mesh
            PVertID = mapb[GVertID]  # Local Vertex ID of parent mesh
            PDof = v2d[PVertID]  # Dof on parent mesh
            value = f.vector()[i]  # Value on local processor
            GValues[dof.local_to_global_index(PDof)] = value
        GValues = self.SyncSum(GValues)

        F.vector().set_local(GValues[imin:imax])
        F.vector().apply("")
        return F

    def Vn_to_Vbln(self, vn):
        """ Take function vn in Vn and interpolate at boundary
        """
        v_is = vn.split(deepcopy=True)
        vbs = []
        for v_i in v_is:
            vbs.append(self.V_to_Vbl(v_i))
        split_to_vec = FunctionAssigner(self.Vbln, [vi.function_space() for vi in vbs])
        vbn = Function(self.Vbln)
        split_to_vec.assign(vbn, vbs)
        return vbn

    def Vbln_to_Vn(self, vbn):
        vb_is = vbn.split(deepcopy=True)
        vs = []
        for vb_i in vb_is:
            vs.append(self.Vbl_to_V(vb_i))
        split_to_vec = FunctionAssigner(self.Vn, [vi.function_space() for vi in vs])
        vn = Function(self.Vn)
        split_to_vec.assign(vn, vs)
        return vn

    def Vb_to_Vbl(self, v):
        return self.V_to_Vbl(self.Vb_to_V(v))

    def Vbl_to_Vb(self, v):
        return self.V_to_Vb(self.Vbl_to_V(v))

    def Vbn_to_Vbln(self, v):
        vb_is = v.split(deepcopy=True)
        vbs = []
        for v_i in vb_is:
            vbs.append(self.Vb_to_Vbl(v_i))
        split_to_vec = FunctionAssigner(self.Vbln, [vi.function_space() for vi in vbs])
        vbn = Function(self.Vbln)
        split_to_vec.assign(vbn, vbs)
        return vbn

    def Vbln_to_Vbn(self, v):
        vb_is = v.split(deepcopy=True)
        vbs = []
        for v_i in vb_is:
            vbs.append(self.Vbl_to_Vb(v_i))
        split_to_vec = FunctionAssigner(self.Vbn, [vi.function_space() for vi in vbs])
        vbn = Function(self.Vbn)
        split_to_vec.assign(vbn, vbs)
        return vbn
  
    def test_Vbn_to_Vn(self):
      vbn = project(Expression(("x[0]", "pow(x[1]-x[0],2)"), element= self.Vbn.ufl_element()),self.Vbn)
      vn = self.Vbn_to_Vn(vbn)
      outfile_1 = XDMFFile("Output/Tests/SettingsMesh/Vbn_to_Vn.xdmf")
      outfile_1.write(vn)
      outfile_1.close()
      pass

    def test_Vbln_to_Vn(self):
      vbn = project(Expression(("x[0]", "pow(x[1]-x[0],2)"), element= self.Vbln.ufl_element()),self.Vbln)
      vn = self.Vbln_to_Vn(vbn)
      outfile_1 = XDMFFile(MPI.comm_world, "Output/Tests/SettingsMesh/Vbln_to_Vn.xdmf")
      outfile_1.write(vn)
      outfile_1.close()
      pass
  
    def test_Vdn_to_Vn(self):
      vdn = project(Expression(("x[0]", "pow(x[1]-x[0],2)"), element= self.Vdn.ufl_element()),self.Vdn)
      vn = self.Vdn_to_Vn(vdn)
      outfile_1 = XDMFFile("Output/Tests/SettingsMesh/Vdn_to_Vn.xdmf")
      outfile_1.write(vn)
      outfile_1.close()
      pass 
  
    def test_Vn_to_Vbn(self):
      vn = project(Expression(("x[0]", "pow(x[1]-x[0],2)"), element= self.Vn.ufl_element()),self.Vn)
      vbn = self.Vn_to_Vbn(vn)
      vn = self.Vbn_to_Vn(vbn)
      outfile_1 = XDMFFile("Output/Tests/SettingsMesh/Vn_to_Vbn_to_Vn.xdmf")
      outfile_1.write(vn)
      outfile_1.close()
      pass 
  
    def test_Vn_to_Vdn(self):
      vn = project(Expression(("x[0]", "pow(x[1]-x[0],2)"), element= self.Vn.ufl_element()),self.Vn)
      vbn = self.Vn_to_Vdn(vn)
      vn = self.Vdn_to_Vn(vbn)
      outfile_1 = XDMFFile("Output/Tests/SettingsMesh/Vn_to_Vdn_to_Vn.xdmf")
      outfile_1.write(vn)
      outfile_1.close()
      pass

    def test_Vbln_to_Vbn(self):
      vn = project(Expression(("x[0]", "pow(x[1]-x[0],2)"), element= self.Vbln.ufl_element()),self.Vbln)
      vbn = self.Vbln_to_Vbn(vn)
      vn = self.Vbn_to_Vn(vbn)
      outfile_1 = XDMFFile(MPI.comm_self, "Output/Tests/SettingsMesh/Vbln_to_Vbn_to_Vn.xdmf")
      outfile_1.write(vn)
      outfile_1.close()
      print('here')
      pass