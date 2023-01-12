from dolfin import *
from .BoundaryOperator import BoundaryOperator
import numpy as np

class LaplaceBeltrami_withbc(BoundaryOperator):
    def __init__(self, dmesh, dnormal, lb_off):
        super().__init__(dmesh, dnormal, lb_off)

        # set solver for param

        # Define trial and test functions
        u = TrialFunction(self.Vd)
        v = TestFunction(self.Vd)
        # Define boundary conditions
        def boundary(x, on_boundary):
            return on_boundary
        self.bc_param = DirichletBC(self.Vd, Constant("1.0"), boundary)
        # Define bilinear form
        a = lb_off*inner(grad(u), grad(v)) * dx(self.dmesh) + inner(u, v) * dx(self.dmesh)
        A = assemble(a)
        self.bc_param.apply(A)
        self.solver_param = PETScKrylovSolver('cg', 'hypre_amg')

        # set solver for extension

        parameters["form_compiler"]["cpp_optimize"] = True
        parameters["form_compiler"]["optimize"] = True
        # Define trial and test functions
        u = TrialFunction(self.Vdn)
        v = TestFunction(self.Vdn)
        # Define boundary conditions
        def boundary(x, on_boundary):
            return on_boundary
        self.bc = DirichletBC(self.Vdn, Constant(("0.0", "0.0")), boundary)
        # Define bilinear form
        a = self.lb_off * inner(grad(u), grad(v)) * dx(self.dmesh) + inner(u, v) * dx(self.dmesh)
        A = assemble(a)
        self.bc.apply(A)
        self.solver = PETScKrylovSolver('cg', 'hypre_amg')

    def param(self, lb_off):
    
        # Define linear form
        L = inner(lb_off, v) * dx(self.dmesh)
        b = assemble(L)
        self.bc_param.apply(b)

        # solve variational problem
        u = Function(self.Vd)
        self.solver_param(u.vector(), b)

        return u


    def eval(self, x):
        # x: corresponds to control in self.Vd
        # print('Extension.vector_laplace_beltrami started.......................')

        # Define linear form
        L = inner(x * self.dnormalf, v) * dx(self.dmesh)
        b = assemble(L)
        self.bc.apply(b)

        # solve variational problem
        u = Function(self.Vdn)
        self.solver(u.vector(), b)

        return u

    def chainrule(self, djy):
        # compute derivative of j(vector_laplace_beltrami(x)) under the knowledge of
        # djy = nabla j(y) (gradient)
        # print('Extension.vector_laplace_beltrami_chainrule started.............')

        # Define linear form
        L = inner(djy, z) * dx(self.dmesh)
        b = assemble(L)
        self.bc.apply(b)

        # solve variational problem
        v = Function(self.Vdn)
        self.solver(v.vector(), b)
        
        # evaluate dj/dx
        xt = TrialFunction(self.Vd)
        djx = assemble(inner(v, self.dnormalf) * xt * dx(self.dmesh))

        return djx
