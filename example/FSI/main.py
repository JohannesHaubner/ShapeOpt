from dolfin import *
from pyadjoint import *
import numpy as np

from pathlib import Path
here = Path(__file__).parent.resolve()
import sys, os
sys.path.insert(0, str(here.parent.parent) + "/shapeopt")

import Tools.settings_mesh as tsm
import Mesh_Postprocessing.post_process as mpp
import Ipopt.ipopt_solver as ipopt_solver

from Constraints import constraints
from Control_to_Trafo import Extension
from Reduced_Objective import reduced_objectives

stop_annotating()

# specify path of directory that contains the files 'mesh_triangles.xdmf' and 'facet_mesh.xdmf'
path_mesh = str(here) + "/mesh"
# specify boundary and extension operator (use Extension.print_options())
boundary_option = 'laplace_beltrami_withbc'
extension_option = 'linear_elasticity'
# governing equations
application = 'fluid_structure'
# constraints
constraint_ids = ['volume', 'barycenter'] #needs to be a list

# set and load parameters
geom_prop = np.load(path_mesh + '/geom_prop.npy', allow_pickle='TRUE').item()
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
         "det_lb": 2e-1, # lower bound for determinant of transformation gradient
         "maxiter_IPOPT": 50,
         "T": 0.02, # simulation horizon for Fluid-Structure interaction simulation
         }

#load mesh
init_mfs = tsm.Initialize_Mesh_and_FunctionSpaces(path_mesh=path_mesh)
mesh = init_mfs.get_mesh()
dmesh = init_mfs.get_design_boundary_mesh()
boundaries = init_mfs.get_boundaries()
domains = init_mfs.get_domains()
params = init_mfs.get_params()
dnormal = init_mfs.get_dnormalf()

#function space in which the control lives
Vd = init_mfs.get_Vd()
Vn = init_mfs.get_Vn()
V = init_mfs.get_V()
v = interpolate(Constant("1.0"),V)

x0 = interpolate(Constant('0.0'), Vd)
d0 = interpolate(Constant(('0.0','0.0')), Vn)

# update discretized params
param["Vol_DmO"] = assemble(v*dx)
param["Vol_O"] = param["Vol_D"] - param["Vol_DmO"]
bo = param["Bary_O"]
bc = constraints['barycenter'](init_mfs, param, boundary_option, extension_option).eval(x0)
param["Bary_O"] = np.add(bc, bo)

# solve optimization problem
Jred = reduced_objectives[application].eval(mesh, domains, boundaries, params, param, red_func=True)
problem = MinimizationProblem(Jred)

ipopt_solver.IPOPTSolver(problem, init_mfs, param, application, constraint_ids,
                                           boundary_option, extension_option).test_objective()

if not os.path.exists(path_mesh + "/Output"):
    os.makedirs(path_mesh + "/Output")
bdfile = File(MPI.comm_self, path_mesh + "/Output/mesh_optimize_test.pvd")

x0 = interpolate(Constant("0.0"), Vd).vector().get_local()

deform_mesh = True

param["lb_off_p"] = 1.0

for lb_off in [1e0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 1e-6]:# [1e0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 1e-6]:  #[1e0, 0.5, 0.25, 0.125, 0.1, 0.05, 0.025, 0.0125, 0.01, 0.005, 0.0025, 0.00125, 0.001, 0.0001, 0.00001, 1e-6]:

  deformation = Extension(init_mfs, param, boundary_option='laplace_beltrami', extension_option='linear_elasticity').dof_to_deformation_precond(init_mfs.vec_to_Vd(x0))
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


      xdmf = XDMFFile(path_mesh + "/mesh_triangles_new.xdmf")
      xdmf2 = XDMFFile(path_mesh + "/facet_mesh_new.xdmf")
      xdmf3 = XDMFFile(path_mesh + "/domains_new.xdmf")
      xdmf.write(new_mesh)
      xdmf2.write(new_boundaries)
      xdmf3.write(new_domains)

      init_mfs = tsm.Initialize_Mesh_and_FunctionSpaces(path_mesh=path_mesh, load_mesh=True)
  else:
      bdfile << defo
      init_mfs = tsm.Initialize_Mesh_and_FunctionSpaces(path_mesh=path_mesh)
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
  Jred = reduced_objectives[application].eval(mesh, domains, boundaries, params, param, red_func=True)
  problem = MinimizationProblem(Jred)
  IPOPT = ipopt_solver.IPOPTSolver(problem, init_mfs, param, application, constraint_ids, boundary_option, extension_option)
  x, info = IPOPT.solve(x0)
  x0 = x

  #plt.figure()
  #plot(mesh)
  #plt.show()

xdmf = XDMFFile(path_mesh + "/mesh_triangles_final.xdmf")
xdmf2 = XDMFFile(path_mesh + "/facet_mesh_final.xdmf")
xdmf3 = XDMFFile(path_mesh + "/domains_final.xdmf")
xdmf.write(new_mesh)
xdmf2.write(new_boundaries)
xdmf3.write(new_domains)

deformation = Extension(init_mfs, param, boundary_option='laplace_beltrami', extension_option='linear_elasticity').dof_to_deformation_precond(init_mfs.vec_to_Vd(x0))
defo = project(deformation, Vn)
ALE.move(mesh, defo, annotate=False)
bdfile << defo