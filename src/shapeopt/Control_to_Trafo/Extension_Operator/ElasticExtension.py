from dolfin import *
from .ExtensionOperator import ExtensionOperator
import numpy as np

class ElasticExtension(ExtensionOperator):
    def __init__(self, mesh, boundaries, params):
        super().__init__(mesh, boundaries, params)

    def eval(self, x):
        # x is a vector valued function in self.Vn that attains values on the design boundary

        parameters["form_compiler"]["cpp_optimize"] = True
        parameters["form_compiler"]["optimize"] = True

        # Define trial and test functions
        u = TrialFunction(self.Vn)
        v = TestFunction(self.Vn)


        # Define boundary conditions
        bc = []
        for i in self.params.keys():
            if i != "design" and not isinstance(self.params[i], bool) and self.params[i] in self.params["boundary_labels"]:
                bc.append(DirichletBC(self.Vn, Constant(("0.0", "0.0")), self.boundaries, self.params[i]))

        # Define bilinear form
        a = self.mu*inner(grad(u) + np.transpose(grad(u)), grad(v)) * dx  # + inner(u,v)*dx
        # Define linear form
        L = inner(x, v)*self.ds(self.params["design"])



        # solve variational problem
        u = Function(self.Vn)
        solve(a == L, u, bc)

        return u

    def param(self):
      # parameter for linear extension equation
      u = TrialFunction(self.V)
      v = TestFunction(self.V)
      bc1 = DirichletBC(self.V, Constant("1.0"), self.boundaries, self.params["inflow"])
      bc2 = DirichletBC(self.V, Constant( "1.0"), self.boundaries, self.params["outflow"])
      bc3 = DirichletBC(self.V, Constant( "1.0"), self.boundaries, self.params["noslip"])
      bc4 = DirichletBC(self.V, Constant("500.0"), self.boundaries, self.params["design"])
      bc = [bc1, bc2, bc3, bc4]

      # Define bilinear form
      a = inner(grad(u), grad(v)) * dx(self.mesh)  # + inner(u,v)*dx
      # Define linear form
      L =Constant("0.0")*v * dx(self.mesh)

      # solve variational problem
      u = Function(self.V)
      solve(a == L, u, bc)
      return u

    def chainrule(self,djy, option, option2):
      # compute derivative of j(linear_elasticity(x)) under the knowledge of
      # djy:
      # option == 1: gradient  nabla j(x)
      # option == 2: deriative j'(x)
      #
      # option2 == 1: djy is gradient
      # option2 == 2: djy is derivative
      #print('Extension.vector_laplace_beltrami_chainrule started.............')

      parameters["form_compiler"]["cpp_optimize"] = True
      parameters["form_compiler"]["optimize"] = True

      # solve adjoint equations
      u = TrialFunction(self.Vn)
      v = TestFunction(self.Vn)

      # Define boundary conditions
      bc = []
      for i in self.params.keys():
          if i != "design" and not isinstance(self.params[i], bool) and self.params[i] in self.params["boundary_labels"]:
              bc.append(DirichletBC(self.Vn, Constant(("0.0", "0.0")), self.boundaries, self.params[i]))

      # Define bilinear form
      a = self.mu*inner(grad(u) + np.transpose(grad(u)), grad(v))*dx
      if option2 ==1:
        # define linear form
        L = inner(djy,v)*dx
        # solve variational problem
        u = Function(self.Vn)
        solve(a==L, u, bc)
      elif option2 == 2:
        A = assemble(a)
        for bci in bc:
           bci.apply(A)
        u = Function(self.Vn)
        dj = Function(self.Vn)
        dj.vector().set_local(djy)
        dj.vector().apply("")
        djy = dj.vector()
        for bci in bc:
           bci.apply(djy)
        solve(A, u.vector(), djy)
      if option == 1:
        return u
      elif option == 2:
        xt = TrialFunction(self.Vn)
        ud = assemble(inner(u,xt)*self.ds(self.params["design"]))
        u = Function(self.Vn)
        u.vector()[:] = ud
        return u


