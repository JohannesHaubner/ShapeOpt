from dolfin import *
import numpy as np

from pathlib import Path
here = Path(__file__).parent
import sys
sys.path.insert(0, str(here.parent.parent))
from Tools.first_order_check import perform_first_order_check

class ExtensionOperator(object):
    def __init__(self, mesh, boundaries, params, opt_inner_bdry=False):
        # x denotes a scalar valued function on mesh
        self.mesh = mesh
        self.Vn = VectorFunctionSpace(mesh, "CG", 1)
        self.V = FunctionSpace(mesh, "CG",1)
        self.n = FacetNormal(mesh)
        self.params = params
        self.boundaries = boundaries
        self.mu = Constant("1.0") #self.param()
        self.opt_inner_bdry = opt_inner_bdry

        if opt_inner_bdry:
            self.ds = Measure('dS', domain=self.mesh, subdomain_data=boundaries)
        else:
            self.ds = Measure("ds", subdomain_data=boundaries)

    def eval(self, x):
        """
        evaluate operator that maps a vector-valued function on design
        boundary to a vector-valued function on whole mesh
        """
        return NotImplementedError

    def chainrule(self, djy, option, option2):
        """
        compute derivative of j(linear_elasticity(x)) under the knowledge of
        djy:
        option == 1: gradient  nabla j(x)
        option == 2: deriative j'(x)

        option2 == 1: djy is gradient
        option2 == 2: djy is derivative
        """
        return NotImplementedError

    def test(self):
          print('----------------------------------------------------------------')
          print('Extension.test started..........................................')

          x0 = interpolate(Constant(("1.0", "0.5")), self.Vn)
          ds = interpolate(Constant(("0.5", "0.2")), self.Vn)
          y0 = self.eval(x0)
          j0 = assemble(0.5 * inner(y0, y0) * dx)
          djy = y0
          djx = self.chainrule(djy, 2, 1)
          epslist = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
          ylist = []
          for eps in epslist:
              xeps = Function(self.Vn)
              xeps.assign(x0)
              xeps.vector().axpy(eps, ds.vector())
              ylist.append(self.eval(xeps))
          jlist = [assemble(0.5 * inner(y, y) * dx) for y in ylist]
          print(jlist)
          ndof = ds.vector().size()
          ds_ = ds.vector().gather(range(ndof))
          djx_ = djx.vector().gather(range(ndof))
          order, diff = perform_first_order_check(jlist, j0, djx_, ds_, epslist)
          return order, diff
