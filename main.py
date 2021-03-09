#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:27:20 2020

@author: haubnerj
"""
from dolfin import *
from pyadjoint import *
import Reduced_Objective.Stokes as ro_stokes
import Tools.save_load_obj as tool
import Control_to_Trafo.dof_to_trafo as ctt
import Tools.settings_mesh as tsm
import Constraints.volume as Cv
import Constraints.barycenter as Cb
import Constraints.determinant as Cd
import Ipopt.ipopt_solver_ as ipopt_so
import numpy as np

import matplotlib.pyplot as plt

print('---------------------------------------------------------------------')
print('main.py started......................................................')
print('---------------------------------------------------------------------')

#load mesh
init_mfs = tsm.Initialize_Mesh_and_FunctionSpaces()
mesh = init_mfs.get_mesh()
dmesh = init_mfs.get_design_boundary_mesh()
boundaries = init_mfs.get_boundaries()
params = init_mfs.get_params()

#function space in which the control lives
Vd = init_mfs.get_Vd()
V = init_mfs.get_V()
v = interpolate(Constant("1.0"),V)
#init_mfs.test_Vn_to_Vdn()


#ro_stokes.test(mesh, boundaries, params)

#ctt.Extension(mesh, boundaries, dmesh, params).test_design_boundary_mesh()
x = np.zeros(dmesh.num_vertices())
xf = Function(Vd)
#print(len(xf.vector()))
if len(xf.vector()) - len(x) == 0:
  pass
else:
  print('ERROR: the dimension of x is incorrect----------------------')
  exit(0)
xf.vector()[:] = x

#ctt.Extension(init_mfs).test_linear_elasticity()
geom_prop = np.load('./Mesh_Generation/geom_prop.npy', allow_pickle='TRUE').item()

param = {"reg": 1e-3, # regularization parameter
         "Vol_D": geom_prop["volume_hold_all_domain"], # volume parameter
         "Bary_D": geom_prop["barycenter_hold_all_domain"], # barycenter
         "Vol_O": geom_prop["volume_D_minus_obstacle"],
         "Bary_eps": 0.0, # slack for barycenter
         "det_lb": 2e-1, # lower bound for determinant of transformation gradient
         "maxiter_IPOPT": 25
         } 
print(xf.vector().max())
#ctt.Extension(init_mfs).test_dof_to_deformation_precond()

#Cv.Volume_Constraint(init_mfs, param["Vol_O"]).test()
#Cb.Barycenter_Constraint(init_mfs, param["Vol_D"], param["Bary_D"]).test(0)
#Cd.Determinant_Constraint(init_mfs, param["det_lb"]).test()

Jred = ro_stokes.reduced_objective(mesh, boundaries,params, red_func=True)
problem = MinimizationProblem(Jred)
x0 = interpolate(Constant('0.0'),Vd)

#ipopt_so.IPOPTSolver(problem, init_mfs, param).test_constraints()

for reg in [1e-5]:
  param["reg"] = reg
  IPOPT = ipopt_so.IPOPTSolver(problem, init_mfs, param)
  x = IPOPT.solve(x0)
  #x = 0.01*np.asarray(range(len(x0.vector().get_local())))
  x0.vector().set_local(x)
  print(x0)

deformation = ctt.Extension(init_mfs).dof_to_deformation_precond(x)

stop_annotating

mesh_new = mesh
ALE.move(mesh_new, deformation, annotate=False)

plt.figure()
plot(mesh_new)
plt.show()

