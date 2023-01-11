from dolfin import *
from dolfin_adjoint import *
import numpy as np
import backend

from pyadjoint.optimization.optimization_solver import OptimizationSolver
from pyadjoint.reduced_functional_numpy import ReducedFunctionalNumPy

from pathlib import Path
here = Path(__file__).parent.resolve()
import sys
sys.path.insert(0, str(here.parent.parent) + "/shapeopt")
import Control_to_Trafo.dof_to_trafo as ctt
from Constraints import constraints as constraints_
from Reduced_Objective import reduced_objectives
from Control_to_Trafo import Extension
from Tools.first_order_check import perform_first_order_check

import cyipopt


class IPOPTSolver(OptimizationSolver):
    def __init__(self, problem, Mesh_, param, application, constraint_ids : list, boundary_option, extension_option, parameters=None):
        try:
            import cyipopt
        except ImportError:
            print("You need to install cyipopt. (It is recommended to install IPOPT with HSL support!)")
            raise
        self.Mesh_ = Mesh_
        #plt.figure()
        #plot(Mesh_.get_mesh())
        #plt.show()
        self.param = param
        self.scalingfactor = 1.0
        self.Vd = self.Mesh_.get_Vd()
        OptimizationSolver.__init__(self, problem, parameters)
        self.rfn = ReducedFunctionalNumPy(self.problem.reduced_functional)
        self.ncontrols = len(self.rfn.get_controls())
        self.rf = self.problem.reduced_functional
        self.dmesh = self.Mesh_.get_design_boundary_mesh()
        self.boundary_option = boundary_option
        self.extension_option = extension_option
        self.application = application
        self.constraint_ids = constraint_ids
        self.problem_obj = self.create_problem_obj(self)
        
        #self.param.reg contains regularization parameter
        print('Initialization of IPOPTSolver finished')

    def create_problem_obj(self, outer):
        return IPOPTSolver.shape_opt_prob(outer)

    def test_objective(self):
        # check dof_to_deformation with first order derivative check
        print('Extension.test_objective started.......................')
        xl = self.dmesh.num_vertices()
        x0 = 0.5*np.ones(xl)
        ds = 1.0*np.ones(xl)
        #ds = interpolate(Expression('0.2*x[0]', degree=1), self.Vd)
        j0 = self.problem_obj.objective(x0) 
        djx = self.problem_obj.gradient(x0)
        epslist = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        jlist = [self.problem_obj.objective(x0+eps*ds) for eps in epslist]
        order, diff = perform_first_order_check(jlist, j0, djx, ds, epslist)
        return order, diff

    def test_constraints(self):
        # check dof_to_deformation with first order derivative check
        print('test_constraints started.......................')
        xl = self.dmesh.num_vertices()
        x0 = interpolate(Constant(0.01), self.Vd).vector().get_local()
        ds = interpolate(Constant(100.0), self.Vd).vector().get_local()
        #ds = interpolate(Expression('0.2*x[0]', degree=1), self.Vd)
        j0 = self.problem_obj.constraints(x0)
        djx = self.problem_obj.jacobian(x0)
        print('j0', j0)
        xl = self.dmesh.num_vertices()
        x0 = 0.5 * np.ones(xl)
        ds = 1.0 * np.ones(xl)
        # ds = interpolate(Expression('0.2*x[0]', degree=1), self.Vd)
        #if only one constraint
        j0 = self.problem_obj.constraints(x0)
        djx = self.problem_obj.jacobian(x0)
        epslist = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        jlist = [self.problem_obj.constraints(x0 + eps * ds) for eps in epslist]
        order, diff = perform_first_order_check(jlist, j0, djx, ds, epslist)
        return order, diff


    class shape_opt_prob(object):
        def __init__(self, outer):
            self.Mesh_ = outer.Mesh_
            self.rfn = outer.rfn
            self.rf = outer.rf
            self.param = outer.param
            self.Vd = outer.Vd
            self.scale = outer.scalingfactor
            self.bo = outer.boundary_option
            self.eo = outer.extension_option
            self.application = outer.application
            self.constraint_ids = outer.constraint_ids
            self.mesh = self.Mesh_.get_mesh()
            self.domains = self.Mesh_.get_domains()
            self.boundaries = self.Mesh_.get_boundaries()
            self.params = self.Mesh_.get_params()

        def objective(self, x):
            #
            # The callback for calculating the objective
            #
            # x to deformation
            print('evaluate objective')
            deformation = Extension(self.Mesh_, self.param, self.bo, self.eo).dof_to_deformation_precond(self.Mesh_.vec_to_Vd(x))
            #deformation = project(deformation, self.mesh)

            # move mesh in direction of deformation
            j1 = reduced_objectives[self.application].eval(self.mesh, self.domains, self.boundaries, self.params, self.param,
                                          control=deformation, flag=False)  #
            #j1 =  self.rfn(deformation.vector()) #self.rfn(deformation)

            # add regularization (note that due to preconditioning no matrix is needed)
            j = j1 + 0.5 * self.param["reg"] * np.dot(x, x)  # regularization
            return j

        def gradient(self, x):
            #
            # The callback for calculating the gradient
            #
            print('evaluate derivative of objective function')
            deformation = Extension(self.Mesh_, self.param, self.bo, self.eo).dof_to_deformation_precond(self.Mesh_.vec_to_Vd(x))
            # deformation = project(deformation, self.mesh)

            # compute gradient
            j, dJf = reduced_objectives[self.application].eval(self.mesh, self.domains, self.boundaries, self.params,
                                              self.param, flag=True, control=deformation)
            #new_params = [self.__copy_data(p.data()) for p in self.rfn.controls]
            #self.rfn.set_local(new_params, deformation.vector().get_local())
            #dJf = self.rfn.derivative(forget=False, project = False)
            #dJf = self.Mesh_.vec_to_Vn(dJf)

            # ufile = File("./Output/Forward/dJf2.pvd")
            # ufile << dJf

            dJ1 = Extension(self.Mesh_, self.param, self.bo, self.eo).dof_to_deformation_precond_chainrule(dJf.vector(), 2)
            dJ = dJ1 + self.param["reg"] * x  # derivative of the regularization

            return dJ

        def constraints(self, x):
            #
            # The callback for calculating the constraints
            # print('evaluate constraint')
            cs = []
            for c in self.constraint_ids:
                cs_ = constraints_[c](self.Mesh_, self.param, self.bo, self.eo).eval(self.Mesh_.vec_to_Vd(x))
                if isinstance(cs_, float):
                    cs.append([cs_])
                else:
                    cs.append(cs_)
            con = self.scale * np.concatenate(cs, axis=0, out=None)
            return con

        def jacobian(self, x):
            #
            # The callback for calculating the Jacobian
            #
            # print('evaluate jacobian')
            #b_ct_d = constraints_['barycenter'](self.Mesh_, self.param, self.bo, self.eo).grad(self.Mesh_.vec_to_Vd(x))
            #v_ct_d = constraints_['volume'](self.Mesh_, self.param, self.bo, self.eo).grad(self.Mesh_.vec_to_Vd(x))
            #jaccon1 = self.scale * np.concatenate((v_ct_d, b_ct_d[0], b_ct_d[1]))  # , d_ct_d))
            cs = []
            for c in self.constraint_ids:
                cs_ = constraints_[c](self.Mesh_, self.param, self.bo, self.eo).grad(self.Mesh_.vec_to_Vd(x))
                if isinstance(cs_, list):
                    for i in range(len(cs_)):
                        cs.append(cs_[i])
                else:
                    cs.append(cs_)
            jaccon = self.scale * np.concatenate(cs, axis=0, out=None)
            return jaccon


        #def hessianstructure(self):
        #    #
        #    # The structure of the Hessian
        #    # Note:
        #    # The default hessian structure is of a lower triangular matrix. Therefore
        #    # this function is redundant. I include it as an example for structure
        #    # callback.
        #    #
        #    global hs
        #
        #    hs = sps.coo_matrix(np.tril(np.ones((4, 4))))
        #    return (hs.col, hs.row)
        #
        #def hessian(self, x, lagrange, obj_factor):
        #    #
        #    # The callback for calculating the Hessian
        #    #
        #    H = obj_factor*np.array((
        #            (2*x[3], 0, 0, 0),
        #            (x[3],   0, 0, 0),
        #            (x[3],   0, 0, 0),
        #            (2*x[0]+x[1]+x[2], x[0], x[0], 0)))
        #
        #    H += lagrange[0]*np.array((
        #            (0, 0, 0, 0),
        #            (x[2]*x[3], 0, 0, 0),
        #            (x[1]*x[3], x[0]*x[3], 0, 0),
        #            (x[1]*x[2], x[0]*x[2], x[0]*x[1], 0)))
        #
        #    H += lagrange[1]*2*np.eye(4)
        #
        #    #
        #    # Note:
        #    #
        #    #
        #    return H[hs.row, hs.col]

        def intermediate(
                self,
                alg_mod,
                iter_count,
                obj_value,
                inf_pr,
                inf_du,
                mu,
                d_norm,
                regularization_size,
                alpha_du,
                alpha_pr,
                ls_trials
                ):

            #
            # Example for the use of the intermediate callback.
            #
            print("Objective value at iteration ", iter_count, " is ", obj_value)
            return

        def __copy_data(self, m):
            """Returns a deep copy of the given Function/Constant."""
            if hasattr(m, "vector"):
                return backend.Function(m.function_space())
            elif hasattr(m, "value_size"):
                return backend.Constant(m(()))
            else:
                raise TypeError('Unknown control type %s.' % str(type(m)))


    def solve(self,x0):
        # x0 is a function in Vd
        max_float = np.finfo(np.double).max
        min_float = np.finfo(np.double).min

        cr = self.param["relax_eq"]
        cl = [] #[0.0, -cr, -cr] #, min_float]
        cu = [] #[0.0, cr, cr] #, 0.0]
        for c in self.constraint_ids:
            dim = constraints_[c](self.Mesh_, self.param, self.boundary_option, self.extension_option).output_dim()
            cl += [-cr] * dim
            cu += [ cr] * dim

        ub = np.array([max_float] * len(x0))
        lb = np.array([min_float] * len(x0))

        from IPython import embed; embed()

        nlp = cyipopt.Problem(
                        n=len(x0),
                        m=len(cl),
                        problem_obj=self.problem_obj,
                        lb=lb,
                        ub=ub,
                        cl=cl,
                        cu=cu
                        )

        nlp.add_option('mu_strategy', 'adaptive')
        #nlp.add_option('derivative_test', 'first-order')
        nlp.add_option('hessian_approximation', 'limited-memory')
        nlp.add_option('limited_memory_update_type', 'bfgs')
        nlp.add_option('point_perturbation_radius', 0.0)
        nlp.add_option('max_iter', self.param["maxiter_IPOPT"])
        nlp.add_option('tol', 1e-5)

        x, info = nlp.solve(x0)
        return x, info
