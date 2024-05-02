from dolfin import *
from .BoundaryOperator import BoundaryOperator

class LaplaceBeltrami_withbc2(BoundaryOperator):
    def __init__(self, dmesh, dnormal, lb_off):
        super().__init__(dmesh, dnormal, lb_off)

        # set solver for param

        # Define trial and test functions
        u = TrialFunction(self.Vd)
        v = TestFunction(self.Vd)
        # Define boundary conditions
        class Tip(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], (0.15)) and near(x[1], (0.2))

        tip = Tip()
        self.bc_param = [DirichletBC(self.Vd, Constant("1.0"), 'on_boundary')] 
        self.bc_param_post = [DirichletBC(self.Vd, Constant("0.00000001"), tip, method='pointwise')]

        # Define bilinear form
        a = lb_off*inner(grad(u), grad(v)) * dx(self.dmesh) + inner(u, v) * dx(self.dmesh)
        A = assemble(a)
        for bc in self.bc_param:
            bc.apply(A)
        self.solver_param = PETScLUSolver()
        self.solver_param.set_operator(A)

        # set solver for extension

        parameters["form_compiler"]["cpp_optimize"] = True
        parameters["form_compiler"]["optimize"] = True
        # Define trial and test functions
        u = TrialFunction(self.Vdn)
        v = TestFunction(self.Vdn)
        # Define boundary conditions
        self.bc = DirichletBC(self.Vdn, Constant(("0.0", "0.0")), 'on_boundary')
        # Define bilinear form
        a = self.lb_off * inner(grad(u), grad(v)) * dx(self.dmesh) + inner(u, v) * dx(self.dmesh)
        A = assemble(a)
        self.bc.apply(A)
        self.solver = PETScLUSolver()
        self.solver.set_operator(A)

        # space dependent weighting
        self.lb_off = self.param(lb_off)

    def param(self, lb_off):
    
        # Define linear form
        v = TestFunction(self.Vd)
        L = inner(lb_off, v) * dx(self.dmesh)
        b = assemble(L)
        for bc in self.bc_param:
            bc.apply(b)

        # solve variational problem
        u = Function(self.Vd)
        self.solver_param.solve(u.vector(), b)

        for bc in self.bc_param_post:
            bc.apply(u.vector())

        return u


    def eval(self, x):
        # x: corresponds to control in self.Vd
        # print('Extension.vector_laplace_beltrami started.......................')

        # Define linear form
        v = TestFunction(self.Vdn)
        L = inner(x * self.dnormalf, v) * dx(self.dmesh)
        b = assemble(L)
        self.bc.apply(b)

        # solve variational problem
        u = Function(self.Vdn)
        self.solver.solve(u.vector(), b)

        return u

    def chainrule(self, djy):
        # compute derivative of j(vector_laplace_beltrami(x)) under the knowledge of
        # djy = nabla j(y) (gradient)
        # print('Extension.vector_laplace_beltrami_chainrule started.............')

        # Define linear form
        z = TestFunction(self.Vdn)
        L = inner(djy, z) * dx(self.dmesh)
        b = assemble(L)
        self.bc.apply(b)

        # solve variational problem
        v = Function(self.Vdn)
        self.solver.solve(v.vector(), b)

        # evaluate dj/dx
        xt = TrialFunction(self.Vd)
        djx = assemble(inner(v, self.dnormalf) * xt * dx(self.dmesh))

        return djx


