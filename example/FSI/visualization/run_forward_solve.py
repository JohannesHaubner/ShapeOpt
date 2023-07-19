from dolfin import *
from pyadjoint import *
import numpy as np
import gc

from pathlib import Path
here = Path(__file__).parent.resolve()
import sys, os

sys.path.insert(0, str(here.parent.parent.parent))
from example.FSI.main import param, geom_prop, path_mesh, application


sys.path.insert(0, str(here.parent.parent.parent) + '/src')

import shapeopt.Tools.settings_mesh as tsm
import shapeopt.Mesh_Postprocessing.post_process as mpp
import shapeopt.Ipopt.ipopt_solver as ipopt_solver

from shapeopt.Constraints import constraints
from shapeopt.Control_to_Trafo import Extension
from shapeopt.Reduced_Objective import reduced_objectives

stop_annotating()

# initial or optimized geometry
initial = True

if initial:
    load_mesh = False
    folder_name = str("/Init")
else:
    load_mesh = True
    folder_name = str("/Opt")

# set and load parameters
param["T"] = 0.10


#load mesh
init_mfs = tsm.Initialize_Mesh_and_FunctionSpaces(path_mesh=path_mesh, load_mesh=load_mesh)
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
v = interpolate(Constant(1.0),V)

x0 = interpolate(Constant(0.0), Vd)
d0 = interpolate(Constant((0.0, 0.0)), Vn)

mesh = init_mfs.get_mesh()
dmesh = init_mfs.get_design_boundary_mesh()
boundaries = init_mfs.get_boundaries()
domains = init_mfs.get_domains()
params = init_mfs.get_params()
dnormal = init_mfs.get_dnormalf()

Vd = init_mfs.get_Vd()
Vn = init_mfs.get_Vn()
V = init_mfs.get_V()
v = interpolate(Constant("1.0"), V)

x0 = interpolate(Constant("0.0"), Vd).vector().get_local()

stop_annotating()
set_working_tape(Tape())
#param["reg"] = reg
Jred = reduced_objectives[application].eval(mesh, domains, boundaries, params, param, red_func=True, visualize=True, vis_folder=folder_name)

print('simulation finished')