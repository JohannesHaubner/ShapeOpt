from dolfin import *
from pyadjoint import *
import numpy as np
import gc

from pathlib import Path
here = Path(__file__).parent.resolve()
import sys, os
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

# specify path of directory that contains the files 'mesh_triangles.xdmf' and 'facet_mesh.xdmf'
path_mesh = str(here.parent) + "/mesh"
# specify boundary and extension operator (use Extension.print_options())
boundary_option = 'laplace_beltrami_withbc'
extension_option = 'linear_elasticity'
# governing equations
application = 'fluid_structure'
# constraints
constraint_ids = ['volume', 'barycenter'] #needs to be a list

# set and load parameters
geom_prop = np.load(path_mesh + '/geom_prop.npy', allow_pickle='TRUE').item()
param = {"reg": 1e-1, # regularization parameter
         "lb_off_p": Constant(1.0), #Laplace Beltrami weighting
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
         "T": 0.10, # simulation horizon for Fluid-Structure interaction simulation
         "gammaP": 1e-3, # penalty parameter for determinant constraint violation
         "etaP": 0.2, # smoothing parameter for max term in determinant const. violation
         "output_path": path_mesh + "/Output/", # folder where intermediate results are stored
         }

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

print('J', Jred)