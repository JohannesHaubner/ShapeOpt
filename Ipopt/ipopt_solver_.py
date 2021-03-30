from __future__ import print_function

from functools import partial
#from dolfin import *
from dolfin_adjoint import *
from copy import deepcopy
import numpy as np
import backend

import Constraints.volume as Cv
import Constraints.barycenter_ as Cb
import Constraints.determinant as Cd

from pyadjoint.optimization.optimization_solver import OptimizationSolver
from pyadjoint.reduced_functional_numpy import ReducedFunctionalNumPy

import Control_to_Trafo.dof_to_trafo as ctt

import Reduced_Objective.Stokes as Stokes

import ipopt as cyipopt

class _IPOptProblem:
    """API used by cyipopt for wrapping the problem"""
    def __init__(self, objective, gradient, constraints, jacobian):
        self.objective = objective
        self.gradient = gradient
        self.constraints = constraints
        self.jacobian = jacobian

class IPOPTSolver(OptimizationSolver):
    """Use the cyipopt bindings to IPOPT to solve the given optimization problem.

    The cyipopt Problem instance is accessible as solver.ipopt_problem."""

    def __init__(self, problem, Mesh_, param, parameters=None):
        self.Mesh_ = Mesh_
        # plt.figure()
        # plot(Mesh_.get_mesh())
        # plt.show()
        self.boundaries = Mesh_.get_boundaries()
        self.params = Mesh_.get_params()
        self.param = param
        self.scale = 1.0
        self.Vd = self.Mesh_.get_Vd()
        self.counter = 0
        self.mesh = self.Mesh_.get_mesh()

        OptimizationSolver.__init__(self, problem, parameters)
        self.rfn = ReducedFunctionalNumPy(self.problem.reduced_functional)
        self.ncontrols = len(self.rfn.get_controls())
        self.rf = self.problem.reduced_functional
        self.dmesh = self.Mesh_.get_design_boundary_mesh()

        # self.param.reg contains regularization parameter
        print('Initialization of IPOPTSolver finished')

        self.__build_ipopt_problem()

    def test_objective(self):
        # check dof_to_deformation with first order derivative check
        print('Extension.test_dof_to_deformation started.......................')
        xl = self.dmesh.num_vertices()
        x0 = 0.5 * np.ones(xl)
        ds = 1.0 * np.ones(xl)
        # ds = interpolate(Expression('0.2*x[0]', degree=1), self.Vd)
        j0 = self.eval_f(x0)
        djx = self.eval_grad_f(x0)
        epslist = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        jlist = [self.eval_f(x0 + eps * ds) for eps in epslist]
        self.perform_first_order_check(jlist, j0, djx, ds, epslist)
        return

    def test_constraints(self):
        # check dof_to_deformation with first order derivative check
        print('Extension.test_dof_to_deformation started.......................')
        xl = self.dmesh.num_vertices()
        x0 = 0.0 * np.ones(xl)
        ds = 1.0 * np.ones(xl)
        # ds = interpolate(Expression('0.2*x[0]', degree=1), self.Vd)
        j0 = self.eval_g(x0)
        djx = self.eval_jac_g(x0)
        print('j0', j0)
        print('djx', djx)
        # print('djx', djx)
        # print('djx', np.ma.size(djx))
        # exit(0)
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

    def eval_f(self, x):
        #
        # The callback for calculating the objective
        #
        # x to deformation
        print('evaluate objective')
        deformation = ctt.Extension(self.Mesh_).dof_to_deformation_precond(x)

        # move mesh in direction of deformation
        j1 = Stokes.reduced_objective(self.mesh, self.boundaries, self.params, self.param,
                                      control=deformation)  # self.rfn(0*deformation.vector().get_local()) #self.rfn(deformation)

        # add regularization (note that due to preconditioning no matrix is needed)
        j = j1 + 0.5 * self.param["reg"] * np.dot(np.asarray(x), np.asarray(x))  # regularization
        return j

    def eval_grad_f(self, x):
        #
        # The callback for calculating the gradient
        #
        print('evaluate derivative of objective funtion')

        deformation = ctt.Extension(self.Mesh_).dof_to_deformation_precond(x)

        # compute gradient
        j, dJf = Stokes.reduced_objective(self.mesh, self.boundaries, self.params,
                                          self.param, flag=True, control=deformation)

        dJ1 = ctt.Extension(self.Mesh_).dof_to_deformation_precond_chainrule(dJf.vector(), 2)
        dJ = dJ1 + self.param["reg"] * x  # derivative of the regularization
        return dJ

    def eval_g(self, x):
        #
        # The callback for calculating the constraints
        # print('evaluate constraint')
        b_ct = Cb.Barycenter_Constraint(self.Mesh_, self.param).eval(x)
        v_ct = Cv.Volume_Constraint(self.Mesh_, self.param["Vol_DmO"]).eval(x)
        # d_ct = Cd.Determinant_Constraint(self.Mesh_, self.param["det_lb"]).eval(x)
        con = self.scale * np.array((v_ct, b_ct[0], b_ct[1]))  # , d_ct))
        return con

    def eval_jac_g(self, x):
        #
        # The callback for calculating the Jacobian
        #
        # print('evaluate jacobian')
        b_ct_d = Cb.Barycenter_Constraint(self.Mesh_, self.param).grad(x)
        v_ct_d = Cv.Volume_Constraint(self.Mesh_, self.param["Vol_DmO"]).grad(x)
        # d_ct_d = Cv.Volume_Constraint(self.Mesh_, self.param["det_lb"]).grad(x)
        jaccon = self.scale * np.concatenate((v_ct_d, b_ct_d[0], b_ct_d[1]))  # , d_ct_d))
        return jaccon

    def __copy_data(self, m):
        """Returns a deep copy of the given Function/Constant."""
        if hasattr(m, "vector"):
            return backend.Function(m.function_space())
        elif hasattr(m, "value_size"):
            return backend.Constant(m(()))
        else:
            raise TypeError('Unknown control type %s.' % str(type(m)))

    def __build_ipopt_problem(self):
        """Build the ipopt problem from the OptimizationProblem instance."""

        import ipopt

        x0 = Function(self.Vd)
        nvar = np.size(x0.vector().get_local())
        ncon = np.size(self.eval_g(x0.vector().get_local()))
        print('nvar, ncon', nvar, ncon)

        max_float = np.finfo(np.double).max
        min_float = np.finfo(np.double).min

        cr = self.param["relax_eq"]
        g_L = np.array([-1.0*cr]*ncon)
        g_U = np.array([cr]*ncon)

        x_U = np.ones((nvar)) * max_float  # np.array([max_float] * nvar)
        x_L = np.ones((nvar)) * min_float  # np.array([min_float] * nvar)

        (lb, ub) = (x_L, x_U)
        fun_g = self.eval_g
        jac_g = self.eval_jac_g
        J = self.eval_f
        dJ = self.eval_grad_f
        (clb, cub) = (g_L, g_U)

        # A callback that evaluates the functional and derivative.
        J = self.rfn.__call__
        dJ = partial(self.rfn.derivative, forget=False)

        nlp = cyipopt.problem(
            n=len(ub),  # length of control vector
            lb=lb,  # lower bounds on control vector
            ub=ub,  # upper bounds on control vector
            m=ncon,  # number of constraints
            cl=clb,  # lower bounds on constraints
            cu=cub,  # upper bounds on constraints
            problem_obj=_IPOptProblem(
                objective=J,  # to evaluate the functional
                gradient=dJ,  # to evaluate the gradient
                constraints=fun_g,  # to evaluate the constraints
                jacobian=jac_g,  # to evaluate the constraint Jacobian
            ),
        )

        """
        if rank(self.problem.reduced_functional.mpi_comm()) > 0:
            nlp.addOption('print_level', 0)    # disable redundant IPOPT output in parallel
        else:
            nlp.addOption('print_level', 6)    # very useful IPOPT output
        """
        # TODO: Earlier the commented out code above was present.
        # Figure out how to solve parallel output cases like these in pyadjoint.
        nlp.addOption("max_iter", self.param["maxiter_IPOPT"])
        nlp.addOption("print_level", 5) #6)

        self.ipopt_problem = nlp




    def solve(self, x0):
        """Solve the optimization problem and return the optimized controls."""
        guess = x0
        results = self.ipopt_problem.solve(guess)

        return results
