from dolfin import *
import numpy as np

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
        order, diff = self.perform_first_order_check(jlist, j0, djx, ds_, epslist)
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