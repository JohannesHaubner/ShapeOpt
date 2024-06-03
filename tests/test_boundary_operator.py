from dolfin import *
from pyadjoint import *
import numpy as np
import pytest

import shapeopt.Tools.settings_mesh as tsm
from shapeopt.Control_to_Trafo.Boundary_Operator import boundary_operators

from pathlib import Path
here = Path(__file__).parent
path_mesh =  str(here.parent) + "/example/Stokes/mesh"

bo_ids = []
for key, value in boundary_operators.items():
    bo_ids.append(key)

#load mesh
init_mfs = tsm.Initialize_Mesh_and_FunctionSpaces(path_mesh=path_mesh)
mesh = init_mfs.get_mesh()
dmesh = init_mfs.get_design_boundary_mesh()
boundaries = init_mfs.get_boundaries()
domains = init_mfs.get_domains()
params = init_mfs.get_params()
dnormal = init_mfs.get_dnormalf()

@pytest.mark.parametrize(
    "bo_id", bo_ids
)
def test_boundary_operator(bo_id):
    print('test boundary operator \t', bo_id)
    order, diff = boundary_operators[bo_id](dmesh, dnormal, Constant(0.5)).test()
    assert order > 1.8 or diff < 1e-12

#test_boundary_operator(bo_ids[0])