from dolfin import *
from .BoundaryOperator import BoundaryOperator
import numpy as np

class LaplaceBeltrami(BoundaryOperator):
    def __init__(self, dmesh, dnormal, lb_off):
        super().__init__(dmesh, dnormal, lb_off)

        parameters["form_compiler"]["cpp_optimize"] = True
        parameters["form_compiler"]["optimize"] = True

        # Define trial and test functions
        u = TrialFunction(self.Vdn)
        v = TestFunction(self.Vdn)

        # Define boundary conditions
        self.bc = []

        # Define bilinear form
        a = self.lb_off * inner(grad(u), grad(v)) * dx(self.dmesh) + inner(u, v) * dx(self.dmesh)
        A = assemble(a)
        A = self.apply_bc(self.bc, A) 
        self.solver = PETScLUSolver()
        self.solver.set_operator(A)

    def apply_bc(self, bc, A):
        for i in bc:
            bc.apply(A)
        return A


    def eval(self, x):
        # x: corresponds to control in self.Vd
        # print('Extension.vector_laplace_beltrami started.......................')

        # Define trial and test functions
        v = TestFunction(self.Vdn)

        # Define linear form
        L = inner(x * self.dnormalf, v) * dx(self.dmesh)
        b = assemble(L)
        b = self.apply_bc(self.bc, b)

        # solve variational problem
        u = Function(self.Vdn)
        self.solver.solve(u.vector(), b)
        return u

    def chainrule(self, djy, gradient=True):
        # compute derivative of j(vector_laplace_beltrami(x)) under the knowledge of
        # djy = nabla j(y) (gradient) if gradient = True
        # and djy derivative if gradient = False
        # print('Extension.vector_laplace_beltrami_chainrule started.............')
        if gradient == True:
            # solve adjoint equation
            z = TestFunction(self.Vdn)

            # Define linear form
            L = inner(djy, z) * dx(self.dmesh)
            b = assemble(L)
        else:
            b = djy

        b = self.apply_bc(self.bc, b)

        # solve variational problem
        v = Function(self.Vdn)
        self.solver.solve(v.vector(), b)

        # evaluate dj/dx
        xt = TrialFunction(self.Vd)
        djx = assemble(inner(v, self.dnormalf) * xt * dx(self.dmesh))

        return djx
