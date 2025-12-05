from dolfin import *
from dolfin_adjoint import *
import numpy as np
from pyadjoint import annotate_tape, stop_annotating
from pyadjoint.overloaded_type import create_overloaded_object
import matplotlib.pyplot as plt
from petsc4py import PETSc

from .dolfin_adjoint_files.petsc_solver import SNESSolver
from .ReducedObjective import ReducedObjective

from pathlib import Path
here = Path(__file__).parent.resolve()
save_directory = str(here.parent.parent) + "/example/FSI/Output/Forward"
import os
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

stop_annotating()

class SNESProblem():
    def __init__(self, F, u, bc):
        V = u.function_space()
        du = TrialFunction(V)
        self.L = F
        self.a = derivative(F, u, du)
        self.bcs = bc
        self.u = u
        self.du = du
        return

    def F(self, snes, x, F):
        x = PETScVector(x)
        F = PETScVector(F)
        assemble(self.L, tensor=F)
        for bc in self.bcs:
            bc.apply(F, x)  
        return            

    def J(self, snes, x, J, P):
        J = PETScMatrix(J)
        assemble(self.a, tensor=J)
        for bc in self.bcs:
            bc.apply(J)
        return

class Write_to_XDMF(object):
    def __init__(self, output_directory, mesh):
        self.output_directory = output_directory
        self.mesh = mesh
        self._initialize()
        self.append = False

    def _initialize(self):
        fssim = self.output_directory + "/"
        Path(fssim).mkdir(parents=True, exist_ok=True)

        names = []
        names.append(fssim + 'velocity.xdmf')
        names.append(fssim + 'pressure.xdmf')
        names.append(fssim + 'characteristic_function.xdmf')
        names.append(fssim + 'deformation.xdmf')
        self.files = []
        for i in names:
            self.files.append(XDMFFile(self.mesh.mpi_comm(), i))
        for file in self.files:
            file.parameters["functions_share_mesh"] = False
            file.parameters["rewrite_function_mesh"] = True

    
    def write(self, vp, pp, chfun, u_p, t):
        vp.rename("v", "v")
        pp.rename("p", "p")
        chfun.rename("c", "c")
        u_p.rename("d", "d")
        self.files[0].write(vp, t)
        self.files[1].write(pp, t)
        self.files[2].write(chfun, t)
        self.files[3].write(u_p, t)
    

    def close(self):
        for file in self.files:
            file.close()


class FluidStructure(ReducedObjective):
    def __init__(self, drag=True, min=True):
        super().__init__()
        self.drag = drag
        self.min = min

    def meanflow_function(self, mesh, boundaries, params, drag):
        # compute function that is (1, 0) on the obstacles boundary and 0 on the outer boundary
        dX = Measure('dx', domain=mesh, metadata={"quadrature_degree": 2})
        V1 = VectorElement("CG", mesh.ufl_cell(), 1)
        VC = FunctionSpace(mesh, V1)

        u = TrialFunction(VC)
        psiu = TestFunction(VC)

        if mesh.topology().dim() == 2:
            if drag:
                func = Constant((1.0, 0.0))
            else:
                if self.min:
                    func = Constant((0.0, 1.0))
                else:
                    func = Constant((0.0, -1.0))
        elif mesh.topology().dim() == 3:
            if drag:
                func = Constant((1.0, 0.0, 0.0))
            else:
                if self.min:
                    func = Constant((0.0, 1.0, 0.0))
                else:
                    func = Constant((0.0, -1.0, 0.0))

        bcs = []
        dim = mesh.topology().dim()
        for i in ["interface", "noslip_obstacle", "obstacle"]:
            if i in params:
                bcs.append(DirichletBC(VC, func, boundaries, params[i]))
        for j in ["noslip", "inflow", "outflow"]:
            if j in params:
                bcs.append(DirichletBC(VC, Constant([0.0]*dim), boundaries, params[j]))


        a = inner(grad(u), grad(psiu))*dX(mesh)
        L = Constant(0.0)*psiu[0]*dX(mesh)

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

        ##parameters["adjoint"]["stop_annotating"] = False
        #parameters["form_compiler"]["cpp_optimize"] = True
        #parameters["form_compiler"]["optimize"] = True

        ##parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -fno-math-errno -march=native'        
        ##parameters['form_compiler']['quadrature_degree'] = 20   

        # compute help function for evaluation of objective

        dX = Measure('dx', domain=mesh, subdomain_data=domains, metadata={"quadrature_degree": 7})
        dS = Measure('dS', domain=mesh, subdomain_data=boundaries, metadata={"quadrature_degree": 7})
        ds = Measure('ds', domain=mesh, subdomain_data=boundaries, metadata={"quadrature_degree": 7})
        n = FacetNormal(mesh)
        dim = mesh.geometric_dimension()

        dxf = dX(mesh)(params['fluid'], metadata={"quadrature_degree": 7})
        dxs = dX(mesh)(params['solid'], metadata={"quadrature_degree": 7})

        # function spaces
        V2 = VectorElement("CG", mesh.ufl_cell(), 2)
        V1 = VectorElement("CG", mesh.ufl_cell(), 1)
        S1 = FiniteElement("CG", mesh.ufl_cell(), 1)
        W  = FunctionSpace(mesh, MixedElement(V2, S1, V2))
        WE = FunctionSpace(mesh, MixedElement(V1, V1))
        U1 = FunctionSpace(mesh, V1)
        VC = FunctionSpace(mesh, V1)
        U = FunctionSpace(mesh, V1)
        P = FunctionSpace(mesh, S1)

        phiv = self.meanflow_function(mesh, boundaries, params, drag=self.drag)

        func = interpolate(Constant(1.0), P)
        bc_inter = DirichletBC(P, Constant(0.0), boundaries, params["interface"])
        bc_inter.apply(func.vector())

        
        # charFunc
        if visualize:
            C = FunctionSpace(mesh, "DG", 0)
            chfun = Function(C, name="charfunc")
            psi = TestFunction(C)
            u = TrialFunction(C)
            L = Constant(1.0)*psi*dxs + Constant(0.0)*psi*dxf
            a = u * psi *dX(mesh)
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

        if dim == 3:
            # Ubar = Constant(1.75)
            # lambdas = Constant(8.0e6)
            # mys = Constant(2e6)
            # rhos = Constant(1.0e3)
            # rhof = Constant(1.0e3)
            # nyf = Constant(1.0e-3)
            Ubar = Constant(1.5)
            lambdas = Constant(2.0e6)
            mys = Constant(0.5e6)
            rhos = Constant(1.0e4)
            rhof = Constant(1.0e3)
            nyf = Constant(1.0e-3)



        auhat = Constant(1e-9)
        aphat = Constant(1e-9)

        t = 0.0
        T = param["T"]
        deltat = param["deltat"]
        k = Constant(deltat)
        theta = Constant(0.5 + 0.5 * deltat)

        INH = False

        # Expressions
        if dim == 2:
            (x, y) = SpatialCoordinate(mesh)
            V_01 = Expression(("(t < 2)*1.5*Ubar*4.0*x[1]*(0.41 -x[1])/ 0.1681*0.5*(1-cos(pi/2*t)) +(1 - (t < 2))*1.5*Ubar*4.0*x[1]*(0.41 -x[1])/ \
                    0.1681", "0.0"), Ubar=Ubar, \
                            t=t, degree=2)

        elif dim == 3:
            V_01 =  Expression(("(t < 2) * Ubar*x[1]*(H -x[1])*x[2]*(B - x[2])/ (0.001764)*0.5*(1-cos(pi/2*t)) + (1-(t<2))*Ubar*x[1]*(H -x[1])*x[2]*(B - x[2])/ (0.001764)", "0.0", "0.0"), Ubar=Ubar, \
                            H=param["H"], B=param["B"], t=t, degree=2)
            
        V_1 = Constant([0.0]*dim)  

        # output files
        if visualize:
            save_directory = str(here.parent.parent.parent) + "/example/FSI/Output/Forward" + vis_folder
            write_to_xdmf = Write_to_XDMF(save_directory, mesh)
            dstring = save_directory + '/displacementy.txt'
            tstring = save_directory + '/times.txt'
            displacementy = []
            times = []

        # run forward model
        counter = -1

        J = 0

        if dim == 2:
            tu = interpolate(Expression(("0.0","0.0"), name = 'Control', degree =1), VC) 
        elif dim == 3:
            tu = interpolate(Expression(("0.0","0.0","0.0"), name = 'Control', degree =1), VC)
        if control:
            tu.vector().set_local(control.vector().get_local())
            tu.vector().apply("")
            if flag == True:
                #print(tu.vector().get_local(),flush=True)
                pass

        if not fallback_strategy:

            # test and trial functions
            w = Function(W, name="state")
            (v, p, u) = split(w)

            w_ = Function(W, name="old_state")
            (v_, p_, u_) = split(w_)

            psi = TestFunction(W)
            (psiv, psip, psiu) = split(psi)

            # weak form
            I = Identity(dim)
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
                    #- inner(azhat * tJhat("+") * grad(z)("+") * tFhati("+") * tFhati("+") * n("+"), psiz("+"))*dS(mesh)(params["interface"]) 
                    + inh_f * inner(Jhat - Constant(1.0), psip) * dxs
                    + inner(tJhat * tr(tFhatti * grad(Jhat * Fhati * v).T), psip) * dxf
                    + inner(aphat * tJhat * tFhati * tFhatti * (grad(p)), (grad(psip))) * dxs
                    + inner(aphat * tJhat * p, psip) * dxs
                    )

            # remaining explicit terms
            A_E = (inner(auhat * tJhat * tFhati * tFhatti * grad(u).T, grad(func * psiu).T) * dxf
                   #- inner(auhat* tJhat("-") * grad(z)("-")* tFhati("-") * tFhatti("-") * n("-"), psiu("-"))*dS(mesh)(params["interface"]) #  grad(u) = Du
                + inner(rhof * tJhat * Jhat * grad(v) * tFhati * Fhati * v, psiv) * dxf
                + inner(tJhat * Jhat * tFhati * Fhati * sigmafv(v), grad(psiv).T)
                * dxf - inner( tJhat * rhos * v, psiu) * dxs
                + inner(tJhat * Jhat * tFhati * Fhati * sigmasv(v), grad(psiv).T)
                * dxs)

            # explicit terms of previous time-step
            A_E_rhs = (inner(auhat * tJhat * tFhati * tFhatti * grad(u_).T, grad(func * psiu).T) * dxf
                       #- inner(auhat * tJhat("-") * grad(z_)("-") * tFhati("-") * tFhatti("-") * n("-"), psiu("-"))*dS(mesh)(params["interface"]) 
                    + inner(rhof * tJhat * Jhat_ * grad(v_) * tFhati * Fhati_ * v_, psiv)
                    * dxf + inner(tJhat * Jhat_ * tFhati * Fhati_ * sigmafv_(v_),grad(psiv).T) * dxf
                    - inner( tJhat * rhos * v_, psiu) * dxs + inner(tJhat * Jhat_ * tFhati * Fhati_ * sigmasv_(v_)
                                                                            , grad(psiv).T) * dxs)

            # shifted crank nicolson scheme
            F = A_T + A_P + A_I + theta * A_E + (Constant(1.0) - theta) * A_E_rhs

            class Projector():
                def __init__(self, V):
                    self.v = TestFunction(V)
                    u = TrialFunction(V)
                    form = inner(u, self.v)*dX(mesh)
                    self.A = assemble(form, annotate=False)
                    self.solver = LUSolver(self.A)
                    self.func = Function(V)
                def project(self, f):
                    L = inner(f, self.v)*dX(mesh)
                    b = assemble(L, annotate=False)
                    self.solver.solve(self.func.vector(), b)
                    return self.func

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

            # boundary conditions
            bc1 = []
            bc2 = []
            if "inflow" in params:
                bc1.append(DirichletBC(W.sub(0), V_01, boundaries, params["inflow"]))  # in   v
                bc1.append(DirichletBC(W.sub(2), V_1, boundaries, params["inflow"]))  # in   u
            if "obstacle" in params:
                bc1.append(DirichletBC(W.sub(0), V_1, boundaries, params["obstacle"]))  # ns   v
                bc1.append(DirichletBC(W.sub(2), V_1, boundaries, params["obstacle"]))  # ns   u
            if "noslip" in params:
                bc1.append(DirichletBC(W.sub(0), V_1, boundaries, params["noslip"]))  # ns   v
                bc1.append(DirichletBC(W.sub(2), V_1, boundaries, params["noslip"]))  # ns   u
            if "noslip_obstacle" in params:
                bc1.append(DirichletBC(W.sub(0), V_1, boundaries, params["noslip_obstacle"]))  # ns   v
                bc1.append(DirichletBC(W.sub(2), V_1, boundaries, params["noslip_obstacle"]))  # ns   u

            # # pressure BC
            # class PressureB(SubDomain):
            #     def inside(self, x, on_boundary):
            #         return near(x[0], (0.0)) and near(x[1], (0.0))
            # pressureB = PressureB()
            # bc1.append(DirichletBC(W.sub(1), Constant(0.0), pressureB, method='pointwise'))
            # bc2.append(DirichletBC(W.sub(1), Constant(0.0), pressureB, method='pointwise'))
        

            direct_solver = False

            if direct_solver:
                Jac = derivative(F, w)
                problem1 = NonlinearVariationalProblem(F, w, bc1, J=Jac)
                PETScOptions.set("pc_type", "lu")
                PETScOptions.set("pc_factor_mat_solver_type", "mumps")
                #PETScOptions.set("mat_mumps_icntl_4", 3) #verbosity
                PETScOptions.set("mat_mumps_icntl_14", 400)
                PETScOptions.set("mat_mumps_icntl_28", 2) #parallel ordering
                PETScOptions.set("mat_mumps_icntl_35", 1)
                PETScOptions.set("mat_mumps_cntl_7", 1e-8)

                solver1 = NonlinearVariationalSolver(problem1)

                #list_linear_solver_methods()

                solver_parameters = {"nonlinear_solver": "newton", "newton_solver": {"maximum_iterations": 25, "linear_solver": "mumps"}}

                solver1.parameters.update(solver_parameters)
            else:
                problem1 = SNESProblem(F, w, bc1)
                solver1 = SNESSolver(PETSc.SNES().create(mesh.mpi_comm()), problem1)

                def get_dofs(W):
                    # sort dofs by states and subdomains
                    w = Function(W)
                    w = interpolate(Constant(('1.0', )*w.ufl_shape[0]), W)
                    psi = TestFunction(W)

                    (v, p, u) = split(w)
                    (psiv, psip, psiu) = split(psi)
                    psi_ = [psiv, psip, psiu]
                    w_ = [v, p, u]

                    state = {"velocity": 0, "pressure": 1, "deformation":2}
                    domain = {"interface": 0, "fluid": 1, "solid": 2}

                    interface_dofs = []
                    fluid_dofs = []
                    solid_dofs = []

                    dsi = dS(mesh)(params['interface'])

                    dofmap = W.dofmap()

                    dx_ = [dxf, dxs]
                    dofs_ = [fluid_dofs, solid_dofs]

                    for i in range(W.num_sub_spaces()):
                        # interface dofs
                        vec = assemble(inner(avg(w_[i]), avg(psi_[i]))*dsi) # assemble vector which has nonzeros at interface
                        indices = np.nonzero(vec)[0]
                        interface_dofs.append(np.array(indices))

                        for j in range(2):
                            vec = assemble(inner(w_[i], psi_[i])*dx_[j]) # assemble vector which has nonzeros at interface, fluid+ interface, solid+interface
                            indicesj = np.nonzero(vec)[0] # indices of vec which are nonzero
                            dofs_[j].append(np.setdiff1d(indicesj, indices)) # substract interface dofs

                    return [interface_dofs] + dofs_, state, domain

                def __test1(W):
                    indexset,_,_ = get_dofs(W)

                    # test
                    k = 0
                    for j in range(len(indexset)):
                        for i in range(len(indexset[j])):
                            k += len(indexset[j][i])
                    assert(k == len(w.vector()[:]))

                def __test2(W):
                    indexset,_, _ = get_dofs(W)

                    k = []
                    for j in range(len(indexset)):
                        for i in range(len(indexset[j])):
                            k = np.concatenate((k, indexset[j][i]), axis=0)
                    l = len(set(k))
                    assert(l == len(w.vector()[:]))

                dofs, state, domain = get_dofs(W) #resort in other bins

                bins = []
                bins.append({"velocity": ["fluid", "interface"], "pressure": ["fluid", "interface"], "deformation": []})
                bins.append({"velocity": ["solid"], "pressure": [], "deformation": ["solid", "interface"]})
                bins.append({"velocity": [], "pressure": [], "deformation": ["fluid"]})
                bins.append({"velocity": [], "pressure": ["solid"], "deformation": []})

                def collect_dofs(dofs, bins, states, domains):
                    dof_bins = []
                    for i in range(len(bins)):
                        bi = np.asarray([])
                        for j in states:
                            if len(bins[i][j])> 0:
                                for k in bins[i][j]:
                                    bi= np.concatenate((bi, dofs[states[j]][domains[k]]), axis = 0)
                        dof_bins.append(PETSc.IS().createGeneral(bi.astype('int32')))
                    return dof_bins

                dof_bins = collect_dofs(dofs, bins, state, domain)
                #from IPython import embed; embed()

                opts = PETSc.Options()
                #opts.setValue('ksp_rtol', 1E-8)
                #opts.setValue('ksp_view_pre', None)
                opts.setValue('snes_monitor', None)
                #opts.setValue('snes_linesearch_monitor', None)
                #opts.setValue('ksp_monitor_true_residual', None)
                #opts.setValue('ksp_converged_reason', None)
                opts.setValue('snes_converged_reason', None)
                opts.setValue('snes_type', 'newtonls')
                opts.setValue('snes_divergence_tolerance', 1e2)
                opts.setValue('snes_linesearch_type', 'l2')
                opts.setValue('snes_max_it', 30)
                opts.setValue('snes_view', None)
                opts.setValue('ksp_atol', 1E-8)
                opts.setValue('ksp_max_it', 1000)
                opts.setValue('ksp_monitor', None)

                option_itsol = 1

                if option_itsol == 0:

                    solver1.snes.setErrorIfNotConverged(True)
                    ksp = solver1.snes.getKSP()
                    ksp.setType('preonly')
                    ksp.getPC().setType('lu')
                    ksp.getPC().setFactorSolverType('mumps')

                    ksp.setFromOptions()
                    solver1.snes.setFromOptions()

                elif option_itsol == 1:

                    ksp = solver1.snes.getKSP()
                    ksp.setType('fgmres')
                    pc = ksp.getPC()
                    pc.setFieldSplitIS(*[(f"{i:d}", dofs_i.sort()) for i, dofs_i in enumerate(dof_bins)])
                    pc.setType(PETSc.PC.Type.FIELDSPLIT)
                    #pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
                    pc.setSPAIVerbose(3)

                    opt_schur = True
                    if opt_schur:
                        #pts.setValue('pc_type', 'fieldsplit')
                        opts.setValue('fieldsplit_type', 'schur')
                        opts.setValue('pc_fieldsplit_0_fields', '0')    # fields in split 0
                        opts.setValue('pc_fieldsplit_1_fields', '1,2,3')    # fields in split 1
                        opts.setValue('fieldsplit_0_ksp_type', 'preonly')
                        opts.setValue('fieldsplit_0_pc_type', 'lu')
                        opts.setValue('fieldsplit_1_ksp_type', 'preonly')
                        opts.setValue('fieldsplit_1_pc_type', 'lu')
                        opts.setValue('fieldsplit_1_pc_type', 'fieldsplit')
                        opts.setValue('fieldsplit_1_pc_fieldsplit_type', 'schur')
                        opts.setValue('fieldsplit_1_pc_fieldsplit_0_fields', '1')
                        opts.setValue('fieldsplit_1_pc_fieldsplit_1_fields', '2,3')
                        opts.setValue('fieldsplit_1_pc_fieldsplit_0_ksp_type', 'preonly')
                        opts.setValue('fieldsplit_1_pc_fieldsplit_0_pc_type', 'lu')
                        opts.setValue('fieldsplit_1_pc_fieldsplit_0_pc_factor_mat_solver_type', 'mumps')
                        opts.setValue('fieldsplit_1_pc_fieldsplit_1_ksp_type', 'preonly')
                        opts.setValue('fieldsplit_1_pc_fieldsplit_1_pc_type', 'lu')
                        opts.setValue('fieldsplit_1_pc_fieldsplit_1_fieldsplit_type', 'additive')
                        opts.setValue('fieldsplit_1_pc_fieldsplit_1_pc_factor_mat_solver_type', 'mumps')
                        opts.setValue('fieldsplit_1_pc_fieldsplit_1_pc_fieldsplit_0_fields', '2')
                        opts.setValue('fieldsplit_1_pc_fieldsplit_1_pc_fieldsplit_1_fields', '3')
                        opts.setValue('fieldsplit_1_pc_fieldsplit_1_pc_fieldsplit_0_ksp_type', 'preonly')
                        opts.setValue('fieldsplit_1_pc_fieldsplit_1_pc_fieldsplit_0_pc_type', 'lu')
                        opts.setValue('fieldsplit_1_pc_fieldsplit_1_pc_fieldsplit_0_pc_factor_mat_solver_type', 'mumps')
                        opts.setValue('fieldsplit_1_pc_fieldsplit_1_pc_fieldsplit_1_ksp_type', 'preonly')
                        opts.setValue('fieldsplit_1_pc_fieldsplit_1_pc_fieldsplit_1_pc_type', 'lu')
                        opts.setValue('fieldsplit_1_pc_fieldsplit_1_pc_fieldsplit_1_pc_factor_mat_solver_type', 'mumps')
                    else:
                        opts.setValue('pc_type', 'fieldsplit')
                        opts.setValue('pc_fieldsplit_type', 'additive')

                        for i in range(len(dof_bins)):
                            opts.setValue(f'fieldsplit_{i}_ksp_type', 'preonly')
                            opts.setValue(f'fieldsplit_{i}_pc_type', 'lu')
                            opts.setValue(f'fieldsplit_{i}_pc_factor_mat_solver_type', 'mumps')

                    pc.setFromOptions()
                    ksp.setFromOptions()
                    solver1.snes.setFromOptions()

                b = PETScVector()  # same as b = PETSc.Vec()
                J_mat = PETScMatrix()   
                solver1.snes.setFunction(problem1.F, b.vec())
                solver1.snes.setJacobian(problem1.J, J_mat.mat())



            while t < T - 0.5 * deltat:
                print("t = \t", t + deltat, "\n", flush=True)
                w_.assign(w)
                counter += 1
                t += deltat
                V_01.t = t
                #V_02.t = t

                #if t <= 2.0:
                if direct_solver:
                    solver1.solve()
                else:
                    solver1.solve(None, problem1.u.vector().vec())


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
                        # take care of bc that might not be fulfilled by projection
                        bcv = []
                        bcu = []
                        if "inflow" in params:
                            #if t < 2. :
                            bcv.append(DirichletBC(U, V_01, boundaries, params["inflow"]))  # in   v
                            #else: 
                            #    bcv.append(DirichletBC(U, V_02, boundaries, params["inflow"]))  # in   v
                            bcu.append(DirichletBC(U, V_1, boundaries, params["inflow"]))  # in   u
                        if "obstacle" in params:
                            bcv.append(DirichletBC(U, V_1, boundaries, params["obstacle"]))  # ns   v
                            bcu.append(DirichletBC(U, V_1, boundaries, params["obstacle"]))  # ns   u
                        if "noslip" in params:
                            bcv.append(DirichletBC(U, V_1, boundaries, params["noslip"]))  # ns   v
                            bcu.append(DirichletBC(U, V_1, boundaries, params["noslip"]))  # ns   u
                        if "noslip_obstacle" in params:
                            bcv.append(DirichletBC(U, V_1, boundaries, params["noslip_obstacle"]))  # ns   v
                            bcu.append(DirichletBC(U, V_1, boundaries, params["noslip_obstacle"]))  # ns   u
                        for bc in bcu:
                            bc.apply(u_p.vector())
                        u_p_inv = Function(U1)
                        u_p_inv.vector().axpy(-1.0, u_p.vector())
                        ALE.move(mesh, u_p)
                        vp = projectorU.project(v)
                        for bc in bcv:
                            bc.apply(vp.vector())
                        pp = projectorP.project(p)
                        write_to_xdmf.write(vp, pp, chfun, u_p, t)
                        u_p = ALE.move(mesh, u_p_inv)

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
                J += assemble(0.5*Constant(param["gammaP"]) * 1.0/(tJhat - Constant(param["det_lb"]))*dX(mesh))

        else:
            I = Identity(2)
            tFhat = I + grad(tu)
            tFhatt = tFhat.T
            tFhati = inv(tFhat)
            tFhatti = tFhati.T
            tJhat = det(tFhat)
            J += assemble((tu[0] + tu[1]) * 10e9 * dX(mesh)) + assemble(0.5*Constant(param["gammaP"]) * 1.0/(tJhat - Constant(param["det_lb"]))*dX(mesh)) # fallback strategy if ipopt wants to evaluate on mesh with bad qualities

        if visualize:
            write_to_xdmf.close()

        if flag:
          print('compute dJ')
          dJ = compute_gradient(J, Control(tu))
          print('end compute dJ')

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

