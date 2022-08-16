from dolfin import *
import numpy as np

class ExtensionOperator(object):
    def __init__(self, mesh, boundaries, params):
        # x denotes a scalar valued function on mesh
        self.mesh = mesh
        self.Vn = VectorFunctionSpace(mesh, "CG", 1)
        self.V = FunctionSpace(mesh, "CG",1)
        self.n = FacetNormal(mesh)
        self.ds = Measure("ds", subdomain_data=boundaries)
        self.params = params
        self.boundaries = boundaries
        self.mu = Constant("1.0") #self.param()

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
          order, diff = self.perform_first_order_check(jlist, j0, djx_, ds_, epslist)
          return order, diff

    def perform_first_order_check(self, jlist, j0, gradj0, ds, epslist):
        # j0: function value at x0
        # gradj0: gradient value at x0
        # epslist: list of decreasing eps-values
        # jlist: list of function values at x0+eps*ds for all eps in epslist
        diff0 = []
        diff1 = []
        order0 = []
        order1 = []
        i = 0
        for eps in epslist:
            je = jlist[i]
            di0 = je - j0
            di1 = je - j0 - eps * np.dot(gradj0, ds)
            diff0.append(abs(di0))
            diff1.append(abs(di1))
            if i == 0:
                order0.append(0.0)
                order1.append(0.0)
            if i > 0:
                order0.append(np.log(diff0[i - 1] / diff0[i]) / np.log(epslist[i - 1] / epslist[i]))
                order1.append(np.log(diff1[i - 1] / diff1[i]) / np.log(epslist[i - 1] / epslist[i]))
            i = i + 1
        for i in range(len(epslist)):
            print('eps\t', epslist[i], '\t\t check continuity\t', order0[i], '\t\t diff0 \t', diff0[i],
                  '\t\t check derivative \t', order1[i], '\t\t diff1 \t', diff1[i], '\n'),

        return order1[-1], diff1[-1]