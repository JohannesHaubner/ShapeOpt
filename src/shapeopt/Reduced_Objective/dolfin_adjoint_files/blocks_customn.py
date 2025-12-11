import dolfin
import numpy
from petsc4py import PETSc
try:
    import ufl_legacy as ufl
except:
    import ufl
from pyadjoint import Block
from pyadjoint.enlisting import Enlist
from ufl.formatting.ufl2unicode import ufl2unicode

from .utils import function_from_vector, extract_subfunction, create_function, extract_mesh_from_form, linalg_solve
from .assembly import assemble_adjoint_value
from .dirichlet_bc import create_bc
from .solving import GenericSolveBlock

from copy import copy, deepcopy

class SolveVarFormBlock(GenericSolveBlock):
    pop_kwargs_keys = GenericSolveBlock.pop_kwargs_keys

    def __init__(self, equation, func, bcs=[], snes=None, *args, **kwargs):
        lhs = equation.lhs
        rhs = equation.rhs
        super().__init__(lhs, rhs, func, bcs, *args, **kwargs)
        self.snes = snes

    def _init_solver_parameters(self, args, kwargs):
        #super()._init_solver_parameters(args, kwargs)
        pass

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
        solver = self.snes.getKSP()
        solver.setOperators(dolfin.as_backend_type(dFdu).mat(), dolfin.as_backend_type(dFdu).mat())
        print('solve with petsc')
        solver.solve(adj_sol.vector().vec(), dolfin.as_backend_type(dJdu).vec())
        print('solve done')

        adj_sol_bdy = None
        if compute_bdy:
            adj_sol_bdy = function_from_vector(self.function_space,
                                               dJdu_copy - assemble_adjoint_value(
                                                   dolfin.action(dFdu_adj_form, adj_sol)))

        return adj_sol, adj_sol_bdy