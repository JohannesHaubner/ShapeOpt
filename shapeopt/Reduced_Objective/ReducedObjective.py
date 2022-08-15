from dolfin import *
from dolfin_adjoint import *
import numpy as np
from pyadjoint import annotate_tape, stop_annotating
from pyadjoint.overloaded_type import create_overloaded_object
import matplotlib.pyplot as plt

stop_annotating()

class ReducedObjective():
    def __init__(self):
        pass

    def eval(self, mesh, domains, boundaries, params, param, flag=False, red_func=False, control=False):
        raise NotImplementedError

    def test(self, Mesh_, param):
        mesh = Mesh_.get_mesh()
        params = Mesh_.get_params()
        domains = Mesh_.get_domains()
        boundaries = Mesh_.get_boundaries()
        stop_annotating()
        V1 = VectorElement("CG", mesh.ufl_cell(), 1)
        VC = FunctionSpace(mesh, V1)

        tu = interpolate(Expression(("0.0", "0.0"), name='Control', degree=1), VC)
        J, dJ = self.eval(mesh, domains, boundaries, params, param, flag=True)

        # compute disturb direction
        g = Expression(("(x[0]-0.2)*(x[0]-0.2)", "(x[1]-0.2)*(x[1]-0.2)"), degree=2)
        zero = Constant(("0.0", "0.0"))
        ds = TestFunction(VC)
        v = TrialFunction(VC)
        a = inner(grad(ds), grad(v)) * dx
        l = inner(zero, v) * dx
        bc_inflow = DirichletBC(VC, zero, boundaries, params["inflow"])
        bc_design = DirichletBC(VC, g, boundaries, params["design"])
        bc_noslip = DirichletBC(VC, zero, boundaries, params["noslip"])
        bcs = [bc_inflow, bc_design, bc_noslip]

        w = Function(VC)
        solve(a == l, w, bcs)

        # epslist
        epslist = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        jlist = []

        for eps in epslist:
            # ew = project((eps*w), VC, annotate=False)
            # ewi = project((-eps*w), VC, annotate=False)
            # ALE.move(mesh, ew, annotate=False)
            # jlist.append(reduced_objective(mesh, boundaries, params))
            # ALE.move(mesh, ewi)
            ew = project((eps * w), VC, annotate=False)
            jlist.append(self.eval(mesh, domains, boundaries, params, param, control=ew))
        # print(dJ.vector().get_local())
        # print(w.vector().get_local())
        ndof = dJ.vector().size()
        dj = dJ.vector().gather(range(ndof))
        w = w.vector().gather(range(ndof))
        self.perform_first_order_check(jlist, J, dj, w, epslist)

        ## plot solution
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.subplot(1,2,1)
        # plot(mesh, color="k", linewidth=0.2, zorder=0)
        # plot(w[0], zorder=1, scale=20)
        # plt.axis("off")
        # plt.subplot(1,2,2)
        # plot(w[1], zorder=1)
        # plt.axis("off")
        # plt.savefig("Output/ReducedObjective/disturb.png", dpi=800, bbox_inches="tight", pad_inches=0)
        return

    @staticmethod
    def perform_first_order_check(jlist, j0, gradj0, ds, epslist):
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