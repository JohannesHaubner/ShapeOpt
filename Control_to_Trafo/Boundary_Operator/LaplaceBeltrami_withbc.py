from dolfin import *
import numpy as np

class Boundary_Operator:
    def __init__(self, dmesh, dnormal, lb_off):
        self.dmesh = dmesh
        self.Vd = FunctionSpace(dmesh, "CG", 1)
        self.Vdn = VectorFunctionSpace(dmesh, "CG", 1)
        self.dnormalf = dnormal
        self.lb_off = self.param(lb_off)

    def param(self, lb_off):
        # Define trial and test functions
        u = TrialFunction(self.Vd)
        v = TestFunction(self.Vd)

        # Define boundary conditions
        def boundary(x, on_boundary):
            return on_boundary

        bc = DirichletBC(self.Vd, Constant("1.0"), boundary)

        # Define bilinear form
        a = lb_off*inner(grad(u), grad(v)) * dx(self.dmesh) + inner(u, v) * dx(self.dmesh)
        # Define linear form
        L = inner(lb_off, v) * dx(self.dmesh)

        # solve variational problem
        u = Function(self.Vd)
        solve(a == L, u, bc)
        return u


    def eval(self, x):
        # x: corresponds to control in self.Vd
        # print('Extension.vector_laplace_beltrami started.......................')

        parameters["form_compiler"]["cpp_optimize"] = True
        parameters["form_compiler"]["optimize"] = True

        # Define trial and test functions
        u = TrialFunction(self.Vdn)
        v = TestFunction(self.Vdn)

        # Define boundary conditions
        def boundary(x, on_boundary):
            return on_boundary

        bc = DirichletBC(self.Vdn, Constant(("0.0", "0.0")), boundary)

        # Define bilinear form
        a = self.lb_off * inner(grad(u), grad(v)) * dx(self.dmesh) + inner(u, v) * dx(self.dmesh)
        # Define linear form
        L = inner(x * self.dnormalf, v) * dx(self.dmesh)

        # solve variational problem
        u = Function(self.Vdn)
        solve(a == L, u, bc)
        return u

    def chainrule(self, djy):
        # compute derivative of j(vector_laplace_beltrami(x)) under the knowledge of
        # djy = nabla j(y) (gradient)
        # print('Extension.vector_laplace_beltrami_chainrule started.............')

        parameters["form_compiler"]["cpp_optimize"] = True
        parameters["form_compiler"]["optimize"] = True

        # solve adjoint equation
        v = TrialFunction(self.Vdn)
        z = TestFunction(self.Vdn)

        # define boundary conditions
        def boundary(x, on_boundary):
            return on_boundary

        bc = DirichletBC(self.Vdn, Constant(("0.0", "0.0")), boundary)

        # Define bilinear form
        a = self.lb_off * inner(grad(v), grad(z)) * dx(self.dmesh) + inner(v, z) * dx(self.dmesh)
        # Define linear form
        L = inner(djy, z) * dx(self.dmesh)

        # solve variational problem
        v = Function(self.Vdn)
        solve(a == L, v, bc)

        # evaluate dj/dx
        xt = TrialFunction(self.Vd)
        djx = assemble(inner(v, self.dnormalf) * xt * dx(self.dmesh))

        return djx

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
        self.perform_first_order_check(jlist, j0, djx, ds_, epslist)
        return

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

        return