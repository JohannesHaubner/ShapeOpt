#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 08:39:51 2020

@author: Johannes Haubner
"""
from dolfin import *
import numpy as np
from dolfin_adjoint import *

class Initialize_Mesh_and_FunctionSpaces():
    def __init__(self):
      #load mesh
      stop_annotating
      mesh = Mesh()
      with XDMFFile("./Output/Mesh_Generation/mesh_triangles.xdmf") as infile:
        infile.read(mesh)
        mvc = MeshValueCollection("size_t", mesh, 1)
      mfile = File("./Output/Tests/ForwardEquation/mesh.pvd")
      mfile << mesh
      with XDMFFile("./Output/Mesh_Generation/facet_mesh.xdmf") as infile:
        infile.read(mvc, "name_to_read")
        boundaries = cpp.mesh.MeshFunctionSizet(mesh,mvc)

      bdfile = File("./Output/Tests/ForwardEquation/boundary.pvd")
      bdfile << boundaries

      params = np.load('./Mesh_Generation/params.npy', allow_pickle='TRUE').item()

      print('mesh created.........................................................')

      #define design boundary mesh

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
      
      # normal vector on mesh
      self.n = FacetNormal(mesh)
      
      # define function spaces
      self.V = FunctionSpace(mesh, "CG", 1)
      self.Vb = FunctionSpace(bmesh, "CG", 1)
      self.Vd = FunctionSpace(dmesh, "CG", 1)
      self.Vn = VectorFunctionSpace(mesh, "CG", 1)
      self.Vbn = VectorFunctionSpace(bmesh, "CG", 1)
      self.Vdn = VectorFunctionSpace(dmesh, "CG", 1)
      dnormal = CellNormal(dmesh)
      self.dnormalf = project(dnormal, self.Vdn)
      
      self.mesh = mesh
      self.bmesh = bmesh
      self.dmesh = dmesh
      self.params = params
      self.boundaries = boundaries
      
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
      mesh = self.mesh
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
      SpaceV = FunctionSpace(self.mesh, "CG", 1)
      SpaceB = FunctionSpace(self.bmesh, "CG", 1)
      
      F = Function(SpaceV)
      LocValues = np.zeros(F.vector().local_size())
      GValues = np.zeros(F.vector().size())
      
      mapb = self.bmesh.entity_map(0)
      d2v = dof_to_vertex_map(SpaceB)
      v2d = vertex_to_dof_map(SpaceV)
      
      dof = SpaceV.dofmap()
      imin, imax = dof.ownership_range()
      
      for i in range(f.vector().local_size()):
          GVertID = Vertex(self.bmesh, d2v[i]).index() # Local Vertex ID for given dof on boundary mesh
          PVertID = mapb[GVertID] # Local Vertex ID of parent mesh
          PDof = v2d[PVertID] # Dof on parent mesh
          value = f.vector()[i] # Value on local processor
          GValues[dof.local_to_global_index(PDof)] = value
      GValues = self.SyncSum(GValues)
      
      F.vector().set_local(GValues[imin:imax])
      F.vector().apply("")
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
  
    def Vb_to_Vd(self,vb):
      bmesh = self.bmesh
      # We use a dof->Vertex mapping to create a global array with all DOF values ordered by mesh vertices
      DofToVert = dof_to_vertex_map(vb.function_space())
      #print(DofToVert)
      VGlobal = np.zeros(vb.vector().size())
      
      vec = vb.vector().get_local()
      for i in range(len(vec)):
          Vert = MeshEntity(bmesh,0,DofToVert[i])
          globalIndex = Vert.global_index()
          VGlobal[globalIndex] = vec[i]
      VGlobal = self.SyncSum(VGlobal)
      #print(VGlobal)
      # Use the inverse mapping to see the DOF values of a boundary function
      dspace = FunctionSpace(self.dmesh, "CG", 1)
      dfunction = Function(dspace)
      mapa = self.dmesh.data().array('parent_vertex_indices',0)
      DofToVert = dof_to_vertex_map(dspace)
      
      LocValues = dfunction.vector().get_local()
      for i in range(len(LocValues)):
          VolVert = MeshEntity(bmesh,0,mapa[int(DofToVert[i])])
          GlobalIndex = VolVert.global_index()
          LocValues[i] = VGlobal[GlobalIndex]
          
      dfunction.vector().set_local(LocValues)
      dfunction.vector().apply('')
      return dfunction
  
    def Vd_to_Vb(self,vd):
      SpaceB = FunctionSpace(self.bmesh, "CG", 1)
      SpaceD = FunctionSpace(self.dmesh, "CG", 1)
      
      F = Function(SpaceB)
      LocValues = np.zeros(F.vector().local_size())
      GValues = np.zeros(F.vector().size())
      
      mapb = self.dmesh.data().array('parent_vertex_indices',0)
      d2v = dof_to_vertex_map(SpaceD)
      v2d = vertex_to_dof_map(SpaceB)
      
      dof = SpaceB.dofmap()
      imin, imax = dof.ownership_range()
      
      for i in range(vd.vector().local_size()):
          GVertID = Vertex(self.dmesh, d2v[i]).index() # Local Vertex ID for given dof on dmesh
          PVertID = mapb[GVertID] # Local Vertex ID of parent mesh
          PDof = v2d[PVertID] # Dof on parent mesh
          value = vd.vector()[i] # Value on local processor
          GValues[dof.local_to_global_index(PDof)] = value
      GValues = self.SyncSum(GValues)
      
      F.vector().set_local(GValues[imin:imax])
      F.vector().apply("")
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
      vd_is = vdn.split(deepcopy = True)
      vs = []
      for vd_i in vd_is:
          vs.append(self.Vd_to_Vb(vd_i))
      split_to_vec = FunctionAssigner(self.Vbn, [vi.function_space() for vi in vs])
      vbn = Function(self.Vbn)
      split_to_vec.assign(vbn, vs)
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
  
    def test_Vbn_to_Vn(self):
      vbn = project(Expression(("x[0]", "pow(x[1]-x[0],2)"), element= self.Vbn.ufl_element()),self.Vbn)
      vn = self.Vbn_to_Vn(vbn)
      outfile_1 = XDMFFile("Output/Tests/SettingsMesh/Vbn_to_Vn.xdmf")
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
