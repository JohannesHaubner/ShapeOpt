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
path_mesh = str(here) + "/mesh3"
path_output = str(here) + "/interface_fsiII"
# specify boundary and extension operator (use Extension.print_options())
boundary_option = 'laplace_beltrami_withbc2'
extension_option = 'linear_elasticity'
# governing equations
application = 'fluid_structure'
# constraints
constraint_ids = ['volume_solid'] #needs to be a list

# set and load parameters
geom_prop = np.load(path_mesh + '/geom_prop.npy', allow_pickle='TRUE').item()
param = {"reg": 10, # regularization parameter
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
         "det_lb": 2e-1, #5e-1, # lower bound for determinant of transformation gradient
         "maxiter_IPOPT": 50,
         "T": 15.0, # simulation horizon for Fluid-Structure interaction simulation
         "deltat": 0.01, # time step size
         "gammaP": 1e-3, # penalty parameter for determinant constraint violation
         "output_path": path_output + "/Output2/", # folder where intermediate results are stored
         }


if __name__ == "__main__":
    #load mesh
    init_mfs = tsm.Initialize_Mesh_and_FunctionSpaces(path_mesh=path_mesh)
    mesh = init_mfs.get_mesh()
    dmesh = init_mfs.get_design_boundary_mesh()
    boundaries = init_mfs.get_boundaries()
    domains = init_mfs.get_domains()
    params = init_mfs.get_params()
    dnormal = init_mfs.get_dnormalf()

    def test_boundary_operator(bo_id):
        print('test boundary operator \t', bo_id)
        order, diff = boundary_operators[bo_id](dmesh, dnormal, Constant(0.0)).test()
        assert order > 1.8 or diff < 1e-12
    test_boundary_operator('laplace_beltrami')

    def test_extension_operator(eo_id):
        print('test extension operator \t', eo_id)
        order, diff = extension_operators[eo_id](mesh, boundaries, params, opt_inner_bdry=True).test()
        assert order > 1.8 or diff < 1e-12
    test_extension_operator("linear_elasticity")

    boundary_operator = boundary_operators[boundary_option](dmesh, dnormal, Constant(0.0))
    extension_operator = extension_operators[extension_option](mesh, boundaries, params, opt_inner_bdry=True)
    dof_to_trafo = ctt.Extension(init_mfs, boundary_operator, extension_operator)


    #function space in which the control lives
    Vd = init_mfs.get_Vd()
    Vn = init_mfs.get_Vn()
    V = init_mfs.get_V()
    v = interpolate(Constant(1.0),V)

    x0 = interpolate(Constant(0.0), Vd)
    d0 = interpolate(Constant((0.0, 0.0)), Vn)

    # update discretized params
    dx = Measure('dx', subdomain_data=domains)
    param["Vol_solid"] = assemble(v*dx(params["solid"]))
    param["Vol_fluid"] = assemble(v *dx(params["fluid"]))
    param["Vol_O"] = param["Vol_D"] - param["Vol_fluid"] - param["Vol_solid"]
    print(param["Vol_D"], param["Vol_fluid"], param["Vol_solid"])
    bo = param["Bary_O"]
    bc = constraints['barycenter'](init_mfs, param, dof_to_trafo).eval(x0)
    param["Bary_O"] = np.add(bc, bo)
    param["solid"] = params["solid"]

    # solve optimization problem
    #Jred = reduced_objectives[application].eval(mesh, domains, boundaries, params, param, red_func=True)
    #problem = MinimizationProblem(Jred)

    #ipopt_solver.IPOPTSolver(problem, init_mfs, param, application, constraint_ids,
    #                                           boundary_option, extension_option).test_objective()

    if not os.path.exists(path_output):
        os.makedirs(path_output)
    bdfile = File(MPI.comm_self, path_output + "/mesh_optimize_test.pvd")

    x0 = interpolate(Constant("0.0"), Vd).vector().get_local()

    remesh_flag = True

    counter = 1

    for lb_off in [1e-3]: 
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
        boundary_operator = boundary_operators[boundary_option](dmesh, dnormal, Constant(lb_off))
        extension_operator = extension_operators[extension_option](mesh, boundaries, params, opt_inner_bdry=True)
        dof_to_trafo = ctt.Extension(init_mfs, boundary_operator, extension_operator)
        Jred = reduced_objectives[application].eval(mesh, domains, boundaries, params, param, red_func=True)
        problem = MinimizationProblem(Jred)
        IPOPT = ipopt_solver.IPOPTSolver(problem, init_mfs, param, application, constraint_ids, dof_to_trafo)
        x, info = IPOPT.solve(x0)
        x0 = x

        print("FSI_main completed", flush=True)

    boundary_operator = boundary_operators[boundary_option](dmesh, dnormal, Constant(lb_off))
    extension_operator = extension_operators[extension_option](mesh, boundaries, params, opt_inner_bdry=True)
    dof_to_trafo = ctt.Extension(init_mfs, boundary_operator, extension_operator)
    deformation = dof_to_trafo.dof_to_deformation_precond(init_mfs.vec_to_Vd(x0))
    np.save("x0_result.npy", x0)
    #defo = project(deformation, Vn)

    # move mesh and save moved mesh
    ALE.move(mesh, defo, annotate=False)
    new_mesh = Mesh(mesh)

    mvc2 = MeshValueCollection("size_t", new_mesh, 2)
    new_domains = cpp.mesh.MeshFunctionSizet(new_mesh, mvc2)
    new_domains.set_values(domains.array())

    mvc = MeshValueCollection("size_t", new_mesh, 1)
    new_boundaries = cpp.mesh.MeshFunctionSizet(new_mesh, mvc)
    new_boundaries.set_values(boundaries.array())

    xdmf = XDMFFile(MPI.comm_self, path_mesh + "/mesh_triangles_final.xdmf")
    xdmf2 = XDMFFile(MPI.comm_self, path_mesh + "/facet_mesh_final.xdmf")
    xdmf3 = XDMFFile(MPI.comm_self, path_mesh + "/domains_final.xdmf")
    xdmf.write(new_mesh)
    xdmf2.write(new_boundaries)
    xdmf3.write(new_domains)


    #defo = project(deformation, Vn)
    #bdfile << defo
