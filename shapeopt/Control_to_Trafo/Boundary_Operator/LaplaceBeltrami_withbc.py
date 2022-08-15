from dolfin import *
from .BoundaryOperator import BoundaryOperator
import numpy as np

class LaplaceBeltrami_withbc(BoundaryOperator):
    def __init__(self, dmesh, dnormal, lb_off):
        super().__init__(dmesh, dnormal, lb_off)

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
