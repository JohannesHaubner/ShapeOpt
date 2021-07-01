#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:27:20 2020

@author: haubnerj
"""
from dolfin import *
from pyadjoint import *

import Reduced_Objective.FluidStructure as ro_stokes
import Tools.save_load_obj as tool
import Control_to_Trafo.dof_to_trafo as ctt

import Tools.settings_mesh as tsm
#import Tools.settings_mesh_storagebad as tsm
import Constraints.volume as Cv
import Constraints.barycenter_ as Cb
import Constraints.determinant as Cd
import Ipopt.ipopt_solver as ipopt_so #_pa as ipopt_so
import numpy as np

import Mesh_Postprocessing.post_process as mpp

import meshio


#from mpi4py import MPI
from pyadjoint.overloaded_type import create_overloaded_object

##
import Control_to_Trafo.Extension_Equation.Elastic_extension as extension
import Control_to_Trafo.Boundary_Operator.LaplaceBeltrami as boundary

stop_annotating()

import matplotlib.pyplot as plt

#print('---------------------------------------------------------------------')
#print('main.py started......................................................')
#print('---------------------------------------------------------------------')

#load mesh

init_mfs = tsm.Initialize_Mesh_and_FunctionSpaces()
mesh = init_mfs.get_mesh()
dmesh = init_mfs.get_design_boundary_mesh()
boundaries = init_mfs.get_boundaries()
domains = init_mfs.get_domains()
params = init_mfs.get_params()
dnormal = init_mfs.get_dnormalf()

###
#extension.Extension(mesh, boundaries, params).test()
#boundary.Boundary_Operator(dmesh, dnormal, 0.0).test()
#ctt.Extension(init_mfs).test_dof_to_deformation_precond()

#function space in which the control lives
Vd = init_mfs.get_Vd()
Vn = init_mfs.get_Vn()
V = init_mfs.get_V()
v = interpolate(Constant("1.0"),V)

## test deformation
#vn = interpolate(Constant(("1.0", "1.0")), Vn)
#init_mfs = tsm.Initialize_Mesh_and_FunctionSpaces(deformation=vn)
#exit(0)

#ctt.Extension(init_mfs).test_dof_to_deformation_precond()


geom_prop = np.load('./Mesh_Generation/geom_prop.npy', allow_pickle='TRUE').item()

param = {"reg": 1e-2, # regularization parameter
         "lb_off_p": 1.0, #Laplace Beltrami weighting
         "Vol_D": geom_prop["volume_hold_all_domain"], # volume parameter
         "Bary_D": geom_prop["barycenter_hold_all_domain"], # barycenter
         "Vol_O": geom_prop["volume_obstacle"],
         "Vol_DmO": geom_prop["volume_D_minus_obstacle"],
         "Bary_O": geom_prop["barycenter_obstacle"],
         "L": geom_prop["length_pipe"],
         "H": geom_prop["heigth_pipe"],
         "relax_eq": 0.0, #relax barycenter
         #"Bary_eps": 0.0, # slack for barycenter
         #"det_lb": 2e-1, # lower bound for determinant of transformation gradient
         "maxiter_IPOPT": 50
         }


#ro_stokes.test(init_mfs, param)
#exit(0)

#ctt.Extension(mesh, boundaries, dmesh, params).test_design_boundary_mesh()

#xf = Function(Vd)



#print(xf.vector().max())
#exit(0)

#Cv.Volume_Constraint(init_mfs, param["Vol_O"]).test()
#Cb.Barycenter_Constraint(init_mfs, param).test()
##Cd.Determinant_Constraint(init_mfs, param["det_lb"]).test()

x0 = interpolate(Constant('0.0'),Vd)
d0 = interpolate(Constant(('0.0','0.0')), Vn)

# update discretized params
param["Vol_DmO"] = assemble(v*dx)
param["Vol_O"] = param["Vol_D"] - param["Vol_DmO"]
bo = param["Bary_O"]
bc = Cb.Barycenter_Constraint(init_mfs, param).eval(x0)
param["Bary_O"] = np.add(bc, bo)

#print(param["Bary_O"])
#Jred = ro_stokes.reduced_objective(mesh, boundaries,params, param, red_func=True)
#problem = MinimizationProblem(Jred)

#ipopt_so.IPOPTSolver(problem, init_mfs, param).test_objective()
#ipopt_so.IPOPTSolver(problem, init_mfs, param).test_constraints()
#exit(0)
#

bdfile = File(MPI.comm_self, "./Output/mesh_optimize_test.pvd")

x0 = interpolate(Constant("0.0"),Vd).vector().get_local()

deform_mesh = True

param["lb_off_p"] = 1.0

for lb_off in [1e0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 1e-6]:# [1e0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 1e-6]:  #[1e0, 0.5, 0.25, 0.125, 0.1, 0.05, 0.025, 0.0125, 0.01, 0.005, 0.0025, 0.00125, 0.001, 0.0001, 0.00001, 1e-6]:

  deformation = ctt.Extension(init_mfs, param).dof_to_deformation_precond(init_mfs.vec_to_Vd(x0))
  defo = project(deformation, Vn)
  ALE.move(mesh, defo, annotate=False)
  if deform_mesh == True:
      defo_new  = mpp.harmonic(defo)
      ALE.move(mesh, defo_new, annotate=False)

      new_mesh = Mesh(mesh)

      mvc2 = MeshValueCollection("size_t", new_mesh, 2)
      new_domains = cpp.mesh.MeshFunctionSizet(new_mesh, mvc2)
      new_domains.set_values(domains.array())

      mvc = MeshValueCollection("size_t", new_mesh, 1)
      new_boundaries = cpp.mesh.MeshFunctionSizet(new_mesh, mvc)
      new_boundaries.set_values(boundaries.array())

      #plt.figure()
      #plot(new_domains)
      #plt.show()


      xdmf = XDMFFile("./Output/Mesh_Generation/mesh_triangles_new.xdmf")
      xdmf2 = XDMFFile("./Output/Mesh_Generation/facet_mesh_new.xdmf")
      xdmf3 = XDMFFile("./Output/Mesh_Generation/domains_new.xdmf")
      xdmf.write(new_mesh)
      xdmf2.write(new_boundaries)
      xdmf3.write(new_domains)

      init_mfs = tsm.Initialize_Mesh_and_FunctionSpaces(load_mesh=True)
  else:
      bdfile << defo
      init_mfs = tsm.Initialize_Mesh_and_FunctionSpaces()
  mesh = init_mfs.get_mesh()
  dmesh = init_mfs.get_design_boundary_mesh()
  boundaries = init_mfs.get_boundaries()
  domains = init_mfs.get_domains()
  params = init_mfs.get_params()
  dnormal = init_mfs.get_dnormalf()
  if deform_mesh == True:
    bdfile << domains

  Vd = init_mfs.get_Vd()
  Vn = init_mfs.get_Vn()
  V = init_mfs.get_V()
  v = interpolate(Constant("1.0"), V)

  if deform_mesh:
    x0 = interpolate(Constant("0.0"), Vd).vector().get_local()

  stop_annotating()
  set_working_tape(Tape())
  #param["reg"] = reg
  param["lb_off_p"] = lb_off
  Jred = ro_stokes.reduced_objective(mesh, domains, boundaries,params, param, red_func=True)
  problem = MinimizationProblem(Jred)

  IPOPT = ipopt_so.IPOPTSolver(problem, init_mfs, param)
  x, info = IPOPT.solve(x0)
  x0 = x

  #plt.figure()
  #plot(mesh)
  #plt.show()

deformation = ctt.Extension(init_mfs, param).dof_to_deformation_precond(init_mfs.vec_to_Vd(x0))
defo = project(deformation, Vn)
ALE.move(mesh, defo, annotate=False)
bdfile << defo