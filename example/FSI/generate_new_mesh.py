from dolfin import *
from pyadjoint import *
import numpy as np
import gc

from pathlib import Path
here = Path(__file__).parent.resolve()
import sys, os
sys.path.insert(0, str(here.parent.parent) + '/src')

import shapeopt.Tools.settings_mesh as tsm
import shapeopt.Mesh_Postprocessing.post_process as mpp
import shapeopt.Ipopt.ipopt_solver as ipopt_solver

from shapeopt.Constraints import constraints
from shapeopt.Control_to_Trafo import Extension
from shapeopt.Reduced_Objective import reduced_objectives
from shapeopt.Control_to_Trafo.Extension_Operator import extension_operators
from shapeopt.Control_to_Trafo.Boundary_Operator import boundary_operators
import shapeopt.Control_to_Trafo.dof_to_trafo as ctt

parameters["ghost_mode"] = "shared_facet"

stop_annotating()

# specify path of directory that contains the files 'mesh_triangles.xdmf' and 'facet_mesh.xdmf'
path_mesh = str(here) + "/mesh2"
# specify boundary and extension operator (use Extension.print_options())
boundary_option = 'laplace_beltrami_withbc2'
extension_option = 'linear_elasticity'
# governing equations
application = 'fluid_structure'
# constraints
constraint_ids = ['volume_solid'] #needs to be a list

if __name__ == "__main__":
    #load mesh
    init_mfs = tsm.Initialize_Mesh_and_FunctionSpaces(path_mesh=path_mesh)
    mesh = init_mfs.get_mesh()
    dmesh = init_mfs.get_design_boundary_mesh()
    boundaries = init_mfs.get_boundaries()
    domains = init_mfs.get_domains()
    params = init_mfs.get_params()
    dnormal = init_mfs.get_dnormalf()

    file = File("domains__.pvd")
    file << domains


    V = VectorFunctionSpace(mesh, "CG", 1)
    defo = Function(V)
    with XDMFFile(path_mesh + "/2final_mesh.xdmf") as infile:
        infile.read_checkpoint(defo, "defo")

    defo = project(100*defo, V)
    
    # move mesh and save moved mesh
    ALE.move(mesh, defo, annotate=False)

    mesh_new = Mesh(mesh)
        
    xdmf = XDMFFile(path_mesh + "/mesh_triangles_final2.xdmf")
    xdmf2 = XDMFFile(path_mesh + "/facet_mesh_final2.xdmf")
    xdmf3 = XDMFFile(path_mesh + "/domains_final2.xdmf")
    xdmf.write(mesh)
    xdmf2.write(boundaries)
    xdmf3.write(domains)

    
