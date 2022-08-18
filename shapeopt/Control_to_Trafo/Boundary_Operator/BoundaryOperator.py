from dolfin import *
import numpy as np

from pathlib import Path
here = Path(__file__).parent
import sys
sys.path.insert(0, str(here.parent))
from shapeopt.Tools.first_order_check import perform_first_order_check

class BoundaryOperator(object):
    def __init__(self, dmesh, dnormal, lb_off):
        self.dmesh = dmesh
        self.Vd = FunctionSpace(dmesh, "CG", 1)
        self.Vdn = VectorFunctionSpace(dmesh, "CG", 1)
        self.dnormalf = dnormal
        self.lb_off = lb_off

    def eval(self, x):
        """
        evaluate operator that maps a scalar-valued function on design
        boundary to a vector-valued function on design boundary
        """
        return NotImplementedError

    def chainrule(self, djy):
        """
        compute derivative of j(eval(x)) under the knowledge of
        djy = nabla j(y) (gradient)
        """
        return NotImplementedError

    def test(self):
        # check laplace beltrami equation with first order derivative check
        print('Extension.test_vector_laplace_beltrami started..................')
        x0 = interpolate(Constant(0.15), self.Vd)
        ds = interpolate(Constant(1.0), self.Vd)
        # ds = interpolate(Expression('0.2*x[0]', degree=1), self.Vd)
        y0 = self.eval(x0)
        j0 = assemble(0.5 * inner(y0, y0) * dx(self.dmesh))
        # correction since assemble adds up values for all processes
        rank = MPI.comm_world.Get_size()
        j0 = j0 / rank
        djy = y0
        djx = self.chainrule(djy).get_local()
        epslist = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        ylist = [self.eval(x0 + eps * ds) for eps in epslist]
        jlist = [assemble(0.5 * inner(y, y) * dx) / rank for y in ylist]  # includes correction
        ds_ = ds.vector().get_local()
        order, diff = perform_first_order_check(jlist, j0, djx, ds_, epslist)
        return order, diff
