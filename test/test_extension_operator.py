from dolfin import *
from pyadjoint import *
import numpy as np
import pytest

import shapeopt.Tools.settings_mesh as tsm
from shapeopt.Control_to_Trafo.Extension_Operator import extension_operators

from pathlib import Path
here = Path(__file__).parent
path_mesh =  str(here.parent) + "/example/Stokes/mesh"

eo_ids = []
for key, value in extension_operators.items():
    eo_ids.append(key)

#load mesh
init_mfs = tsm.Initialize_Mesh_and_FunctionSpaces(path_mesh=path_mesh)
mesh = init_mfs.get_mesh()
dmesh = init_mfs.get_design_boundary_mesh()
boundaries = init_mfs.get_boundaries()
domains = init_mfs.get_domains()
params = init_mfs.get_params()
dnormal = init_mfs.get_dnormalf()

@pytest.mark.parametrize(
    "eo_id", eo_ids
)
def test_extension_operator(eo_id):
    print('test extension operator \t', eo_id)
    order, diff = extension_operators[eo_id](mesh, boundaries, params).test()
    assert order > 1.8 or diff < 1e-12

#test_extension_operator(eo_ids[0])
