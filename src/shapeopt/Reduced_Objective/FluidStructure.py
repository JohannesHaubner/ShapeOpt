from dolfin import *
from dolfin_adjoint import *
import numpy as np
from pyadjoint import annotate_tape, stop_annotating
from pyadjoint.overloaded_type import create_overloaded_object
import matplotlib.pyplot as plt

from .ReducedObjective import ReducedObjective

from pathlib import Path
here = Path(__file__).parent.resolve()
save_directory = str(here.parent.parent) + "/example/FSI/Output/Forward"
import os
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

stop_annotating()

class FluidStructure(ReducedObjective):
    def __init__(self):
        super().__init__()

    def meanflow_function(self, mesh, boundaries, params):
        # compute function that is (1, 0) on the obstacles boundary and 0 on the outer boundary
        V1 = VectorElement("CG", mesh.ufl_cell(), 1)
        VC = FunctionSpace(mesh, V1)

        u = TrialFunction(VC)
        psiu = TestFunction(VC)

        bc1 = DirichletBC(VC, Constant((1.0, 0.0)), boundaries, params["interface"])
        bc2 = DirichletBC(VC, Constant((0.0, 0.0)), boundaries, params["noslip"])
        bc3 = DirichletBC(VC, Constant((1.0, 0.0)), boundaries, params["noslip_obstacle"])
        bc4 = DirichletBC(VC, Constant((0.0, 0.0)), boundaries, params["inflow"])
        bc5 = DirichletBC(VC, Constant((0.0, 0.0)), boundaries, params["outflow"])
        bc6 = DirichletBC(VC, Constant((1.0, 0.0)), boundaries, params["obstacle"])
        bcs = [bc1, bc2, bc3, bc4, bc5, bc6]

        a = inner(grad(u), grad(psiu))*dx(mesh)
        L = Constant(0.0)*psiu[0]*dx(mesh)

        u = Function(VC)

        solve(a == L, u, bcs)

        #file = File("./Output/meanflow_direction.pvd")
        #file << u

        return u


    def eval(self, mesh, domains, boundaries, params, param, flag=False, red_func=False, control=False, add_penalty=True, visualize=False, vis_folder=str(), fallback_strategy=False, point=[0.6, 0.2]):
        # mesh generated
        # params dictionary, includes labels for boundary parts:
        # params.inflow
        # params.outflow
        # params.noslip
        # params.obstacle

        print("Use FluidStructure to compute reduced objective",flush=True)

        #parameters["adjoint"]["stop_annotating"] = False

        # compute help function for evaluation of objective

        dx = Measure('dx', domain=mesh, subdomain_data=domains)
        dS = Measure('dS', domain=mesh, subdomain_data=boundaries)
        ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
        n = FacetNormal(mesh)
        dim = mesh.geometric_dimension()

        dxf = dx(mesh)(params['fluid'])
        dxs = dx(mesh)(params['solid'])

        # function spaces
        V2 = VectorElement("CG", mesh.ufl_cell(), dim)
        V1 = VectorElement("CG", mesh.ufl_cell(), 1)
        S1 = FiniteElement("CG", mesh.ufl_cell(), 1)
        W  = FunctionSpace(mesh, MixedElement(V2, S1, V2, V2))
        WE = FunctionSpace(mesh, MixedElement(V1, V1))
        U1 = FunctionSpace(mesh, V1)
        VC = FunctionSpace(mesh, V1)
        U = FunctionSpace(mesh, V1)
        P = FunctionSpace(mesh, S1)

        phiv = self.meanflow_function(mesh, boundaries, params)

        func = interpolate(Constant(1.0), P)
        bc_inter = DirichletBC(P, Constant(0.0), boundaries, params["interface"])
        bc_inter.apply(func.vector())

        
        # charFunc
        if visualize:
            C = FunctionSpace(mesh, "DG", 0)
            chfun = Function(C, name="charfunc")
            psi = TestFunction(C)
            u = TrialFunction(C)
            L = Constant(1.0)*psi*dx(mesh)(params["solid"]) + Constant(0.0)*psi*dx(mesh)(params["fluid"]) 
            a = u * psi *dx
            solve(a == L, chfun, [])

        stop_annotating()
        set_working_tape(Tape())
        annotate_tape()

        # parameters
        Ubar = Constant(1.0)
        lambdas = Constant(2.0e6)
        mys = Constant(0.5e6)
        rhos = Constant(1.0e4)
        rhof = Constant(1.0e3)
        nyf = Constant(1.0e-3)

        auhat = Constant(1e-9)
        azhat = Constant(1e-9)
        aphat = Constant(1e-9)

        t = 0.0
        T = param["T"]
        deltat = param["deltat"]
        k = Constant(deltat)
        theta = Constant(0.5 + 0.5 * deltat)

        INH = False

        # Expressions
        (x, y) = SpatialCoordinate(mesh)
        V_01 = Expression(("1.5*Ubar*4.0*x[1]*(0.41 -x[1])/ 0.1681*0.5*(1-cos(pi/2*t))", "0.0"), Ubar=Ubar, \
                          t=t, degree=2)
        V_02 = Expression(("1.5*Ubar*4.0*x[1]*(0.41 -x[1])/ \
                   0.1681", "0.0"), Ubar=Ubar, t=t, degree=2)
        V_1 = Constant([0.0]*dim)

        # output files
        if visualize:
            save_directory = str(here.parent.parent.parent) + "/example/FSI/Output/Forward" + vis_folder
            fssim = save_directory + "/"
            import os
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            vstring = fssim + 'velocity.pvd'
            v2string = fssim + 'velocity2.pvd'
            pstring = fssim + 'pressure.pvd'
            dstring = fssim + 'displacementy.txt'
            tstring = fssim + 'times.txt'
            charstring = fssim + 'char_sol.pvd'
            vfile = File(vstring)
            pfile = File(pstring)
            v2file = File(v2string)
            charfile = File(charstring)
            displacementy = []
            times = []

        # run forward model
        counter = -1

        J = 0

        if not fallback_strategy:

            tu = interpolate(Expression(("0.0","0.0"), name = 'Control', degree =1), VC)
            if control:
                tu.vector().set_local(control.vector().get_local())
                tu.vector().apply("")
                if flag == True:
                    print(tu.vector().get_local(),flush=True)

            # test and trial functions
            w = Function(W, name="state")
            (v, p, u, z) = split(w)

            w_ = Function(W, name="old_state")
            (v_, p_, u_, z_) = split(w_)

            psi = TestFunction(W)
            (psiv, psip, psiu, psiz) = split(psi)

            # weak form
            I = Identity(2)
            tFhat = I + grad(tu)
            tFhatt = tFhat.T
            tFhati = inv(tFhat)
            tFhatti = tFhati.T
            tJhat = det(tFhat)
            Fhat = I + grad(u) * tFhati
            Fhatt = Fhat.T
            Fhati = inv(Fhat)
            Fhatti = Fhati.T
            Ehat = 0.5 * (Fhatt * Fhat - I)
            Jhat = det(Fhat)

            # stress tensors
            def sigmafp(p):
                return -p * I

            def sigmafv(v):
                return rhof * nyf * (grad(v) * tFhati * Fhati + Fhatti * tFhatti \
                                    * grad(v).T)

            def sigmasp(p):
                if INH:
                    return -p * I  # INH
                else:
                    return Constant(0.0)  # STVK

            def sigmasv(v):
                if INH:
                    return mys * (Fhat * Fhatt - I)  # INH
                else:
                    return inv(Jhat) * Fhat * (lambdas * tr(Ehat) * I \
                                            + 2.0 * (mys) * Ehat) * Fhatt  # STVK

            # INH or STVK setting for solid material
            if INH == False:
                inh_f = Constant(0.0)
            else:
                inh_f = Constant(1.0)

            # variables for previous time-step
            Fhat_ = I + grad(u_) * tFhati
            Fhatt_ = Fhat_.T
            Fhati_ = inv(Fhat_)
            Fhatti_ = Fhati_.T
            Ehat_ = 0.5 * (Fhatt_ * Fhat_ - I)
            Jhat_ = det(Fhat_)
            Jhattheta = theta * Jhat + (1.0 - theta) * Jhat_

            def sigmafv_(v_):
                return rhof * nyf * (grad(v_) * tFhati * Fhati_ + Fhatti_ * tFhatti \
                                    * grad(v_).T)

            def sigmasv_(v_):
                if INH:
                    return mys * (Fhat_ * Fhatt_ - I)  # INH
                else:
                    return inv(Jhat_) * Fhat_ * (lambdas * tr(Ehat_) * I \
                                                + 2.0 * (mys) * Ehat_) * Fhatt_  # STVK

            # terms with time derivatives
            A_T = (1.0 / k * inner(rhof * Jhattheta * tJhat * (v - v_), psiv) * dxf
                - 1.0 / k * inner(rhof * Jhat * tJhat * grad(v) * tFhati * Fhati * (u - u_), psiv)
                * dxf + 1.0 / k * inner(tJhat * rhos * (v - v_), psiv) * dxs
                + 1.0 / k * inner( tJhat * rhos * (u - u_), psiu) * dxs)

            # pressure terms
            A_P = (inner(tJhat * Jhat * tFhati * Fhati * sigmafp(p),
                        grad(psiv).T) * dxf + inh_f * inner(tJhat * Jhat * tFhati * Fhati * sigmasp(p)
                                                                            , grad(psiv)) * dxs)

            # implicit terms (e.g. incompressibiliy)
            A_I = (
                    -inner( tJhat * z, psiz) * dxf + inner(tJhat * tFhati * tFhatti * grad(u).T,
                                                                    grad(psiz).T) * dxf
                    + inner(azhat * tJhat * tFhati * tFhatti * grad(z).T,
                                                                    grad(func * psiz).T) * dxs
                    #- inner(azhat * tJhat("+") * grad(z)("+") * tFhati("+") * tFhati("+") * n("+"), psiz("+"))*dS(mesh)(params["interface"]) 
                    + inh_f * inner(Jhat - Constant(1.0), psip) * dxs
                    + inner(tJhat * tr(tFhatti * grad(Jhat * Fhati * v).T), psip) * dxf
                    + inner(aphat * tJhat * tFhati * tFhatti * (grad(p)), (grad(psip))) * dxs
                    + inner(aphat * tJhat * p, psip) * dxs
        )

            # remaining explicit terms
            A_E = (inner(auhat * tJhat * tFhati * tFhatti * grad(z).T, grad(func * psiu).T) * dxf
                   #- inner(auhat* tJhat("-") * grad(z)("-")* tFhati("-") * tFhatti("-") * n("-"), psiu("-"))*dS(mesh)(params["interface"]) #  grad(u) = Du
                + inner(rhof * tJhat * Jhat * grad(v) * tFhati * Fhati * v, psiv) * dxf
                + inner(tJhat * Jhat * tFhati * Fhati * sigmafv(v), grad(psiv).T)
                * dxf - inner( tJhat * rhos * v, psiu) * dxs
                + inner(tJhat * Jhat * tFhati * Fhati * sigmasv(v), grad(psiv).T)
                * dxs)

            # explicit terms of previous time-step
            A_E_rhs = (inner(auhat * tJhat * tFhati * tFhatti * grad(z_).T, grad(func * psiu).T) * dxf
                       #- inner(auhat * tJhat("-") * grad(z_)("-") * tFhati("-") * tFhatti("-") * n("-"), psiu("-"))*dS(mesh)(params["interface"]) 
                    + inner(rhof * tJhat * Jhat_ * grad(v_) * tFhati * Fhati_ * v_, psiv)
                    * dxf + inner(tJhat * Jhat_ * tFhati * Fhati_ * sigmafv_(v_),grad(psiv).T) * dxf
                    - inner( tJhat * rhos * v_, psiu) * dxs + inner(tJhat * Jhat_ * tFhati * Fhati_ * sigmasv_(v_)
                                                                            , grad(psiv).T) * dxs)

            # shifted crank nicolson scheme
            F = A_T + A_P + A_I + theta * A_E + (Constant(1.0) - theta) * A_E_rhs

            # output files
            if visualize == True:
                fssim = save_directory + "/"
                vstring = fssim + 'velocity.pvd'
                v2string = fssim + 'velocity2.pvd'
                pstring = fssim + 'pressure.pvd'
                dstring = fssim + 'displacementy.txt'
                charstring = fssim + 'char_sol.pvd'
                vfile = File(vstring)
                pfile = File(pstring)
                v2file = File(v2string)
                charfile = File(charstring)
                displacementy = []

            class Projector():
                def __init__(self, V):
                    self.v = TestFunction(V)
                    u = TrialFunction(V)
                    form = inner(u, self.v)*dx
                    self.A = assemble(form, annotate=False)
                    self.solver = LUSolver(self.A)
                    self.uh = Function(V)
                def project(self, f):
                    L = inner(f, self.v)*dx
                    b = assemble(L, annotate=False)
                    self.solver.solve(self.uh.vector(), b)
                    return self.uh

            projectorU = Projector(U)
            projectorU1 = Projector(U1)
            projectorP = Projector(P)

            # run forward model
            counter = -1
            if visualize:
                # append displacementy
                u_p = projectorU1.project(u)
                u_p.rename("projection", "projection")
                try:
                    displacementy.append(u_p(Point(point[0], point[1]))[1])
                    times.append(t)
                    np.savetxt(dstring, displacementy)
                    np.savetxt(tstring, times)
                except:
                    pass

                # plot transformed mesh
                if abs(counter / 4.0 - int(counter / 4.0)) == 0:
                    u_p_inv = Function(U1)
                    u_p_inv.vector().axpy(-1.0, u_p.vector())
                    ALE.move(mesh, u_p)
                    vp = projectorU.project(v)
                    pp = projectorP.project(p)
                    vp.rename("velocity", "velocity")
                    pp.rename("pressure", "pressure")
                    pfile << pp
                    vfile << vp
                    v2file << vp
                    charfile << chfun
                    u_p = ALE.move(mesh, u_p_inv)

            # boundary conditions
            bc_in_0_1 = DirichletBC(W.sub(0), V_01, boundaries, params["inflow"])  # in   v
            bc_in_0_2 = DirichletBC(W.sub(0), V_02, boundaries, params["inflow"])  # in   v
            bc_ns_0 = DirichletBC(W.sub(0), V_1, boundaries, params["noslip"])  # ns   v
            bc_d_0 = DirichletBC(W.sub(0), V_1, boundaries, params["obstacle"])  # ns   v
            bc_in_2 = DirichletBC(W.sub(2), V_1, boundaries, params["inflow"])  # in   u
            bc_ns_2 = DirichletBC(W.sub(2), V_1, boundaries, params["noslip"])  # ns   u
            bc_d_2 = DirichletBC(W.sub(2), V_1, boundaries, params["obstacle"])  # ns   u
            bc_nso_0 = DirichletBC(W.sub(0), V_1, boundaries, params["noslip_obstacle"])  # ns   v
            bc_nso_2 = DirichletBC(W.sub(2), V_1, boundaries, params["noslip_obstacle"])  # ns   v

            # update pressure boundary condition
            bc1 = [bc_in_0_1, bc_ns_0, bc_d_0, bc_in_2, bc_ns_2, bc_d_2, bc_nso_0, bc_nso_2]
            bc2 = [bc_in_0_2, bc_ns_0, bc_d_0, bc_in_2, bc_ns_2, bc_d_2, bc_nso_0, bc_nso_2]
            Jac = derivative(F, w)
            problem1 = NonlinearVariationalProblem(F, w, bc1, J=Jac)
            problem2 = NonlinearVariationalProblem(F, w, bc2, J=Jac)

            solver1 = NonlinearVariationalSolver(problem1)
            solver2 = NonlinearVariationalSolver(problem2) 

            solver_parameters = {"nonlinear_solver": "newton", "newton_solver": {"maximum_iterations": 25}}

            solver1.parameters.update(solver_parameters)
            solver2.parameters.update(solver_parameters)


            while t < T - 0.5 * deltat:
                print("t = \t", t + deltat, "\n", flush=True)
                w_.assign(w)
                counter += 1
                t += deltat
                V_01.t = t
                V_02.t = t

                if t <= 2.0:
                    solver1.solve()
                else:
                    solver2.solve()

                if visualize:
                    # append displacementy
                    u_p = projectorU1.project(u)
                    u_p.rename("projection", "projection")
                    try:
                        displacementy.append(u_p(Point(point[0], point[1]))[1])
                        times.append(t)
                        np.savetxt(dstring, displacementy)
                        np.savetxt(tstring, times)
                    except:
                        pass

                    # plot transformed mesh
                    if abs(counter / 4.0 - int(counter / 4.0)) == 0:
                        u_p_inv = Function(U1)
                        u_p_inv.vector().axpy(-1.0, u_p.vector())
                        ALE.move(mesh, u_p)
                        up = projectorU.project(u)
                        vp = projectorU.project(v)
                        pp = projectorP.project(p)
                        vp.rename("velocity", "velocity")
                        pp.rename("pressure", "pressure")
                        pfile << pp
                        vfile << vp
                        v2file << vp
                        charfile << chfun
                        ALE.move(mesh, u_p_inv)

                    ##########################
                J += assemble(float(deltat)*(-1.0 / T * (inner(tJhat * Jhat * rhof * (
                        (v - v_) / float(deltat) + ((grad(v) * tFhati * Fhati * (v - (u - u_) / float(deltat))))), phiv) * dxf
                                        - Jhat * tJhat * p * tr(grad(phiv) * tFhati * Fhati) * dxf)
                        + 2.0 * nyf * inner(Jhat * tJhat * (grad(v) * tFhati * Fhati + Fhatti * tFhatti * grad(v).T),
                        (grad(phiv) * tFhati * Fhati + Fhatti * tFhatti * grad(phiv).T)) * dxf ))

            def smoothmax(r, eps=1e-4):
                return conditional(gt(r, eps), r - eps / 2, conditional(lt(r, 0), 0, r ** 2 / (2 * eps)))

            #objective function
            if add_penalty:
                print("Objective value without penalization is ", J)
                J += assemble(0.5*Constant(param["gammaP"]) * 1.0/(tJhat - Constant(param["det_lb"]))*dx(mesh))

        else:
            J += assemble((tu[0] + tu[1]) * 10e9 * dx(mesh)) + assemble(0.5*Constant(param["gammaP"]) * 1.0/(tJhat - Constant(param["det_lb"]))*dx(mesh)) # fallback strategy if ipopt wants to evaluate on mesh with bad qualities

        if flag:
          dJ = compute_gradient(J,Control(tu))

        ## plot solution
        #import matplotlib.pyplot as plt
        #plt.figure()
        #plt.subplot(1,2,1)
        #plot(mesh, color="k", linewidth=0.2, zorder=0)
        #plot(dJ[1], zorder=1, scale=20)
        #plt.axis("off")
        #plt.subplot(1,2,2)
        #plot(u[0], zorder=1)
        #plt.axis("off")
        #plt.savefig("Output/ReducedObjective/initial.png", dpi=800, bbox_inches="tight", pad_inches=0)
        stop_annotating
        if red_func:
          m = Control(tu)
          return ReducedFunctional(J, m)
        else:
          if flag:
            return J, dJ
          else:
            return J

