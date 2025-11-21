import dolfin
from pyadjoint.tape import annotate_tape, get_working_tape
from .blocks_customn import SolveVarFormBlock


class SNESSolver():
    def __init__(self, snes, problem):
        self.snes = snes
        self.problem = problem

    def solve(self, *args, **kwargs):
        ad_block_tag = kwargs.pop("ad_block_tag", None)
        annotate = annotate_tape(kwargs)

        if annotate:
            tape = get_working_tape()

            F = self.problem.L
            bcs = self.problem.bcs

            u = self.problem.u

            sb_kwargs = SolveVarFormBlock.pop_kwargs(kwargs)
            block = SolveVarFormBlock(F == 0, u, bcs,
                                      #solver_parameters={"newton_solver": self.parameters.copy()},
                                      ad_block_tag=ad_block_tag,
                                      **sb_kwargs)
            tape.add_block(block)

        #newargs = [self] + list(args)
        out = self.snes.solve(*args, **kwargs)

        if annotate:
            block.add_output(u.create_block_variable())

        return out