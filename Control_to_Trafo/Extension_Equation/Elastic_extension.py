from dolfin import *
import numpy as np

class Extension:
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
        # x is a vector valued function in self.Vn that attains values on the design boundary

        parameters["form_compiler"]["cpp_optimize"] = True
        parameters["form_compiler"]["optimize"] = True

        # Define trial and test functions
        u = TrialFunction(self.Vn)
        v = TestFunction(self.Vn)


        # Define boundary conditions
        bc1 = DirichletBC(self.Vn, Constant(("0.0", "0.0")), self.boundaries, self.params["inflow"])
        bc2 = DirichletBC(self.Vn, Constant(("0.0", "0.0")), self.boundaries, self.params["outflow"])
        bc3 = DirichletBC(self.Vn, Constant(("0.0", "0.0")), self.boundaries, self.params["noslip"])
        bc = [bc1, bc2, bc3]

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
      bc1 = DirichletBC(self.Vn, Constant(("0.0","0.0")), self.boundaries, self.params["inflow"])
      bc2 = DirichletBC(self.Vn, Constant(("0.0","0.0")), self.boundaries, self.params["outflow"])
      bc3 = DirichletBC(self.Vn, Constant(("0.0","0.0")), self.boundaries, self.params["noslip"])
      bc = [bc1, bc2, bc3]

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
        bc1.apply(A)
        bc2.apply(A)
        bc3.apply(A)
        u = Function(self.Vn)
        dj = Function(self.Vn)
        dj.vector().set_local(djy)
        dj.vector().apply("")
        djy = dj.vector()
        bc1.apply(djy)
        bc2.apply(djy)
        bc3.apply(djy)
        solve(A, u.vector(), djy)
      if option == 1:
        return u
      elif option == 2:
        xt = TrialFunction(self.Vn)
        ud = assemble(inner(u,xt)*self.ds(self.params["design"]))
        u = Function(self.Vn)
        u.vector()[:] = ud
        return u

    def test(self):
          print('----------------------------------------------------------------')
          print('Extension.test_linear_elasticity started........................')

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
          self.perform_first_order_check(jlist, j0, djx_, ds_, epslist)
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

