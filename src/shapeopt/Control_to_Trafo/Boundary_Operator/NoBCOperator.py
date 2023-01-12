from dolfin import *
import BoundaryOperator 
import numpy as np

class NoBCO(BoundaryOperator.BoundaryOperator):
    def __init__(self, dmesh, dnormal, lb_off):
        super().__init__(dmesh, dnormal, lb_off)

    def eval(self, x):
        # x: corresponds to control in self.Vd
        return x

    def chainrule(self, djy):
        # compute derivative of j(vector_laplace_beltrami(x)) under the knowledge of
        # djy = nabla j(y) (gradient)
        psi = TrialFunction(self.Vd)
        djx = assemble(djy * psi * dx(self.dmesh))
        return djx

if __name__ == "__main__":
    from dolfin import *
    from pyadjoint import *
    import numpy as np
    import pytest

    from pathlib import Path
    here = Path(__file__).parent
    import sys
    sys.path.insert(0, str(here.parent.parent.parent))

    import shapeopt.Tools.settings_mesh as tsm
    from shapeopt.Control_to_Trafo.Boundary_Operator import boundary_operators

    path_mesh =  str(here.parent.parent.parent.parent) + "/example/Stokes/mesh"

    #load mesh
    init_mfs = tsm.Initialize_Mesh_and_FunctionSpaces(path_mesh=path_mesh)
    mesh = init_mfs.get_mesh()
    dmesh = init_mfs.get_design_boundary_mesh()
    boundaries = init_mfs.get_boundaries()
    domains = init_mfs.get_domains()
    params = init_mfs.get_params()
    dnormal = init_mfs.get_dnormalf()

    print('test boundary operator \t')
    order, diff = NoBCO(dmesh, dnormal, 0.0).test()
