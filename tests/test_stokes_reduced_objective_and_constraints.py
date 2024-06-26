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
         "Vol_solid": 1., # random value to make volume_solid constraint test work
         "solid": 5, # for volume_solid constraint test
         "relax_eq": 0.0, #relax barycenter
         #"Bary_eps": 0.0, # slack for barycenter
         "det_lb": 2e-1, # lower bound for determinant of transformation gradient
         "maxiter_IPOPT": 50
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
ids = []
for key, value in constraints.items():
    ids.append(key)

@pytest.mark.parametrize(
    "id", ids
)
def test_constraints(id):
    print('test constraint \t', id)
    order, diff = constraints[id](init_mfs, param, dof_to_trafo).test()
    assert order > 1.8 or diff < 1e-12


def test_stokes():
    print('test stokes')
    order, diff = reduced_objectives[application].test(init_mfs, param)
    assert order > 1.8 or diff < 1e-12