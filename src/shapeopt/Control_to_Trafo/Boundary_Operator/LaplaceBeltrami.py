from dolfin import *
from .BoundaryOperator import BoundaryOperator
import numpy as np

class LaplaceBeltrami(BoundaryOperator):
    def __init__(self, dmesh, dnormal, lb_off):
        super().__init__(dmesh, dnormal, lb_off)

    def eval(self, x):
        # x: corresponds to control in self.Vd
        # print('Extension.vector_laplace_beltrami started.......................')

        parameters["form_compiler"]["cpp_optimize"] = True
        parameters["form_compiler"]["optimize"] = True

        # Define trial and test functions
        u = TrialFunction(self.Vdn)
        v = TestFunction(self.Vdn)

        # Define boundary conditions
        bc = []

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
        bc = []

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
