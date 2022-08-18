from dolfin import *
#from dolfin_adjoint import *
import numpy as np

from pathlib import Path
here = Path(__file__).parent
import sys
sys.path.insert(0, str(here.parent))
from shapeopt.Tools.first_order_check import perform_first_order_check

class Constraint(object):
    def __init__(self, Mesh_, param, boundary_option, extension_option):
        self.Mesh_ = Mesh_
        self.mesh = Mesh_.get_mesh()
        self.dim = self.Mesh_.mesh.geometric_dimension()
        self.Vd = Mesh_.get_Vd()
        self.Vn = Mesh_.get_Vn()
        self.V = Mesh_.get_V()
        self.param = param
        self.boundary_option = boundary_option
        self.extension_option = extension_option

    def eval(self, x):
        raise NotImplementedError

    def grad(self, x):
        raise NotImplementedError

    def test(self):
        # check eval and gradient computation with first order derivative check
        print('Constraint.test started................................')
        x0 = interpolate(Constant(0.01), self.Vd).vector().get_local()
        ds = interpolate(Constant(100.0), self.Vd).vector().get_local()
        # ds = interpolate(Expression('0.2*x[0]', degree=1), self.Vd)
        j0 = self.eval(self.Mesh_.vec_to_Vd(x0))
        djx = self.grad(self.Mesh_.vec_to_Vd(x0))
        epslist = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        ylist = [self.Mesh_.vec_to_Vd(x0 + eps * ds) for eps in epslist]
        jlist = [self.eval(y) for y in ylist]
        ds_ = ds  # .vector().get_local()
        order, diff = perform_first_order_check(jlist, j0, djx, ds_, epslist)
        return order, diff


