from dolfin import *
from pyadjoint import *
import numpy as np
import pytest

import shapeopt.Tools.settings_mesh as tsm
from shapeopt.Control_to_Trafo import Extension

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
         "maxiter_IPOPT": 50
         }

def test_dof_to_deformation():
    print('test dof to deformation')
    order, diff = Extension(init_mfs, param, boundary_option='laplace_beltrami', extension_option='linear_elasticity').test_dof_to_deformation_precond()
    assert order > 1.8 or diff < 1e-12