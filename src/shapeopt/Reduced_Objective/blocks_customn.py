import dolfin
import numpy
import ufl_legacy as ufl
from pyadjoint import Block
from pyadjoint.enlisting import Enlist
from ufl_legacy.formatting.ufl2unicode import ufl2unicode

from fenics_adjoint.utils import function_from_vector, extract_subfunction, create_function, extract_mesh_from_form, linalg_solve
from fenics_adjoint.blocks.assembly import assemble_adjoint_value
from fenics_adjoint.blocks.dirichlet_bc import create_bc
from fenics_adjoint.blocks.solving import GenericSolveBlock

class SolveVarFormBlock(GenericSolveBlock):
    pop_kwargs_keys = GenericSolveBlock.pop_kwargs_keys

    def __init__(self, equation, func, bcs=[], *args, **kwargs):
        lhs = equation.lhs
        rhs = equation.rhs
        super().__init__(lhs, rhs, func, bcs, *args, **kwargs)

    def _init_solver_parameters(self, args, kwargs):
        super()._init_solver_parameters(args, kwargs)
        if len(self.forward_args) <= 0:
            self.forward_args = args

        if len(self.forward_kwargs) <= 0:
            self.forward_kwargs = kwargs

        if "solver_parameters" in self.forward_kwargs and "mat_type" in self.forward_kwargs["solver_parameters"]:
            self.assemble_kwargs["mat_type"] = self.forward_kwargs["solver_parameters"]["mat_type"]

        if len(self.adj_kwargs) <= 0:
            solver_parameters = kwargs.get("solver_parameters", {})
            if len(self.adj_args) <= 0:
                if "linear_solver" in solver_parameters:
                    adj_args = [solver_parameters["linear_solver"]]
                    if "preconditioner" in solver_parameters:
                        adj_args.append(solver_parameters["preconditioner"])
                    self.adj_args = tuple(adj_args)
                elif "newton_solver" in solver_parameters and "linear_solver" in solver_parameters["newton_solver"]:
                    adj_args = [solver_parameters["newton_solver"]["linear_solver"]]
                    if "preconditioner" in solver_parameters["newton_solver"]:
                        adj_args.append(solver_parameters["newton_solver"]["preconditioner"])
                    self.adj_args = tuple(adj_args)
            self.adj_kwargs = solver_parameters

    def _assemble_and_solve_adj_eq(self, dFdu_adj_form, dJdu, compute_bdy=True):
        dJdu_copy = dJdu.copy()
        bcs = self._homogenize_bcs()
        kwargs = self.assemble_kwargs.copy()
        kwargs["bcs"] = bcs
        dFdu = assemble_adjoint_value(dFdu_adj_form, **kwargs)

        # Apply boundary conditions on adj_dFdu and dJdu.
        for bc in bcs:
            bc.apply(dJdu)

        adj_sol = create_function(self.function_space)
        lu_solver_methods = dolfin.lu_solver_methods()
        solver_method = self.adj_args[0] if len(self.adj_args) >= 1 else "default"
        solver_method = "default" if solver_method == "lu" else solver_method

        if solver_method in lu_solver_methods:
            solver = dolfin.LUSolver(solver_method)
            solver_parameters = self.adj_kwargs.get("lu_solver", {})
        else:
            solver = dolfin.KrylovSolver(*self.adj_args)
            solver_parameters = self.adj_kwargs.get("krylov_solver", {})
        solver.parameters.update(solver_parameters)
        solver.solve(dFdu, adj_sol.vector(), dJdu)

        adj_sol_bdy = None
        if compute_bdy:
            adj_sol_bdy = function_from_vector(self.function_space,
                                               dJdu_copy - assemble_adjoint_value(
                                                   dolfin.action(dFdu_adj_form, adj_sol)))

        return adj_sol, adj_sol_bdy