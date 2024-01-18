from dolfin import *
from pyadjoint import *
import numpy as np
import pytest

import shapeopt.Tools.settings_mesh as tsm

from shapeopt.Constraints import constraints
from shapeopt.Control_to_Trafo import Extension
from shapeopt.Reduced_Objective import reduced_objectives
from shapeopt.Control_to_Trafo.Boundary_Operator import boundary_operators
from shapeopt.Control_to_Trafo.Extension_Operator import extension_operators
import Ipopt.ipopt_solver as ipopt_solver

from pathlib import Path
here = Path(__file__).parent
path_mesh =  str(here.parent) + "/example/Stokes/mesh"

#load mesh
init_mfs = tsm.Initialize_Mesh_and_FunctionSpaces(path_mesh=path_mesh)
mesh = init_mfs.get_mesh()
dmesh = init_mfs.get_design_boundary_mesh()
boundaries = init_mfs.get_boundaries()
domains = init_mfs.get_domains()
params = init_mfs.get_params()
dnormal = init_mfs.get_dnormalf()

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
         "output_path": path_mesh + "/Output/", # folder where intermediate results are stored
         }

# specify boundary and extension operator (use Extension.print_options())
boundary_option = 'laplace_beltrami'
extension_option = 'linear_elasticity'
boundary_operator = boundary_operators[boundary_option](dmesh, dnormal, Constant(0.5))
extension_operator = extension_operators[extension_option](mesh, boundaries, params)
dof_to_trafo = Extension(init_mfs, boundary_operator, extension_operator)
# governing equations
application = 'stokes' #'fluid structure' needs to be tested: if no fluid domain assigned --> error since no fluid part of domain
# constraints
constraint_ids = ['volume'] #needs to be a list


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
bc = constraints['barycenter'](init_mfs, param, dof_to_trafo).eval(x0)
param["Bary_O"] = np.add(bc, bo)

# solve optimization problem
Jred = reduced_objectives[application].eval(mesh, domains, boundaries, params, param, red_func=True)
problem = MinimizationProblem(Jred)

def test_ipopt_objective():
    print('test ipopt objective')
    order, diff = ipopt_solver.IPOPTSolver(problem, init_mfs, param, application, constraint_ids,
                                           dof_to_trafo).test_objective()
    assert order > 1.8 or diff < 1e-12

def test_ipopt_constraints():
    print('test ipopt constraints')
    order, diff = ipopt_solver.IPOPTSolver(problem, init_mfs, param, application, constraint_ids,
                                           dof_to_trafo).test_constraints()
    assert order[0] > 1.8 or diff[0] < 1e-12
