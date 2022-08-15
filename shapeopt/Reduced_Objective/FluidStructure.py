#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 09:11:45 2021

@author: Johannes Haubner
"""

from dolfin import *
from dolfin_adjoint import *
import numpy as np
from pyadjoint import annotate_tape, stop_annotating
from pyadjoint.overloaded_type import create_overloaded_object
import matplotlib.pyplot as plt

stop_annotating()

def meanflow_function(mesh, boundaries, params):
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
    bc6 = DirichletBC(VC, Constant((1.0, 0.0)), boundaries, params["design"])
    bcs = [bc1, bc2, bc3, bc4, bc5, bc6]

    a = inner(grad(u), grad(psiu))*dx(mesh)
    L = Constant(0.0)*psiu[0]*dx(mesh)

    u = Function(VC)

    solve(a == L, u, bcs)

    #file = File("./Output/meanflow_direction.pvd")
    #file << u

    return u


def reduced_objective(mesh, domains, boundaries, params, param, flag =False, red_func = False, control = False):
    # mesh generated 
    # params dictionary, includes labels for boundary parts:
    # params.inflow
    # params.outflow
    # params.noslip
    # params.design

    print("Use FluidStructure to compute reduced objective")

    #parameters["adjoint"]["stop_annotating"] = False

    # compute help function for evaluation of objective

    dx = Measure('dx', domain=mesh, subdomain_data=domains)
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

    phiv = meanflow_function(mesh, boundaries, params)

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
    atuhat = Constant(1e-9)
    atzhat = Constant(1.0)
    azhat = Constant(-1.0)
    aphat = Constant(1e-9)

    t = 0.0
    T = 0.02
    deltat = 0.01
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

    tu = interpolate(Expression(("0.0","0.0"), name = 'Control', degree =1), VC)
    if control:
        tu.vector().set_local(control.vector().get_local())
        tu.vector().apply("")
        if flag == True:
            print(tu.vector().get_local())
    
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
           + 1.0 / k * inner(tJhat * rhos * (u - u_), psiu) * dxs)

    # pressure terms
    A_P = (inner(tJhat * Jhat * tFhati * Fhati * sigmafp(p),
                 grad(psiv).T) * dxf + inh_f * inner(tJhat * Jhat * tFhati * Fhati * sigmasp(p)
                                                                      , grad(psiv)) * dxs)

    # implicit terms (e.g. incompressibiliy)
    A_I = (
            inner(azhat * tJhat * z, psiz) * dx(mesh) - inner(azhat * tJhat * tFhati * tFhatti * grad(u).T,
                                                              grad(psiz).T) * dx(mesh)
            + inh_f * inner(Jhat - Constant(1.0), psip) * dxs
            + inner(tJhat * tr(tFhatti * grad(Jhat * Fhati * v).T), psip) * dxf
            + inner(aphat * tJhat * tFhati * tFhatti * (grad(p)), (grad(psip))) * dxs
   )

    # remaining explicit terms
    A_E = (inner(auhat * tJhat * tFhati * tFhatti * grad(z).T, grad(psiu).T) * dxf
           + inner(rhof * tJhat * Jhat * grad(v) * tFhati * Fhati * v, psiv) * dxf
           + inner(tJhat * Jhat * tFhati * Fhati * sigmafv(v), grad(psiv).T)
           * dxf - inner(tJhat * rhos * v, psiu) * dxs
           + inner(tJhat * Jhat * tFhati * Fhati * sigmasv(v), grad(psiv).T)
           * dxs)

    # explicit terms of previous time-step
    A_E_rhs = (inner(auhat * tJhat * tFhati * tFhatti * grad(z_).T, grad(psiu).T) * dxf
               + inner(rhof * tJhat * Jhat_ * grad(v_) * tFhati * Fhati_ * v_, psiv)
               * dxf + inner(tJhat * Jhat_ * tFhati * Fhati_ * sigmafv_(v_),grad(psiv).T) * dxf
               - inner(tJhat * rhos * v_, psiu) * dxs + inner(tJhat * Jhat_ * tFhati * Fhati_ * sigmasv_(v_)
                                                                      , grad(psiv).T) * dxs)

    # shifted crank nicolson scheme
    F = A_T + A_P + A_I + theta * A_E + (1.0 - theta) * A_E_rhs

    # output files
    saveoption = False
    if saveoption == True:
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

    # run forward model
    counter = -1

    J = 0

    while t < T - 0.5 * deltat:
        print("t = \t", t + deltat, "\n" )
        w_.assign(w)
        counter += 1
        t += deltat
        V_01.t = t
        V_02.t = t

        # boundary conditions
        bc_in_0_1 = DirichletBC(W.sub(0), V_01, boundaries, params["inflow"])  # in   v
        bc_in_0_2 = DirichletBC(W.sub(0), V_02, boundaries, params["inflow"])  # in   v
        bc_ns_0 = DirichletBC(W.sub(0), V_1, boundaries, params["noslip"])  # ns   v
        bc_d_0 = DirichletBC(W.sub(0), V_1, boundaries, params["design"])  # ns   v
        bc_in_2 = DirichletBC(W.sub(2), V_1, boundaries, params["inflow"])  # in   u
        bc_ns_2 = DirichletBC(W.sub(2), V_1, boundaries, params["noslip"])  # ns   u
        bc_d_2 = DirichletBC(W.sub(2), V_1, boundaries, params["design"])  # ns   u
        bc_nso_0 = DirichletBC(W.sub(0), V_1, boundaries, params["noslip_obstacle"])  # ns   v
        bc_nso_2 = DirichletBC(W.sub(2), V_1, boundaries, params["noslip_obstacle"])  # ns   v

        # update pressure boundary condition
        bc1 = [bc_in_0_1, bc_ns_0, bc_d_0, bc_in_2, bc_ns_2, bc_d_2, bc_nso_0, bc_nso_2]
        bc2 = [bc_in_0_2, bc_ns_0, bc_d_0, bc_in_2, bc_ns_2, bc_d_2, bc_nso_0, bc_nso_2]
        if t <= 2.0:
            solve(F == 0, w, bc1,
                  solver_parameters={"nonlinear_solver": "newton", "newton_solver": {"maximum_iterations": 10}})
        else:
            solve(F == 0, w, bc2,
                  solver_parameters={"nonlinear_solver": "newton", "newton_solver": {"maximum_iterations": 10}})

        if saveoption == True:
            # append displacementy
            u_p = project(u, U1, annotate=False, name="projection")
            try:
                displacementy.append(u_p(Point(0.6, 0.2))[1])
                np.savetxt(dstring, displacementy)
            except:
                pass

            # plot transformed mesh
            if abs(counter / 4.0 - int(counter / 4.0)) == 0:
                # u_p = project(u,U2, annotate=False)
                u_p_inv = project((-1.0 * u), U1, annotate=False)
                ALE.move(mesh, u_p)
                up = project(u, U, annotate=False)
                vp = project(v, U, annotate=False)
                pp = project(p, P, annotate=False)
                vp.rename("velocity", "velocity")
                pp.rename("pressure", "pressure")
                pfile << pp
                vfile << vp
                # vp modified
                # vp.vector()[:] *= charFunc.vector()
                charfile << charFunc
                v2file << vp

                ALE.move(mesh, u_p_inv)

            ##########################
        J += assemble(float(deltat)*(-1.0 / T * (inner(tJhat * Jhat * rhof * (
                (v - v_) / float(deltat) + ((grad(v) * tFhati * Fhati * (v - (u - u_) / float(deltat))))), phiv) * dxf
                                - Jhat * tJhat * p * tr(grad(phiv) * tFhati * Fhati) * dxf)
                 + 2.0 * nyf * inner(Jhat * tJhat * (grad(v) * tFhati * Fhati + Fhatti * tFhatti * grad(v).T),
                 (grad(phiv) * tFhati * Fhati + Fhatti * tFhatti * grad(phiv).T)) * dxf ))

    gammaP = 1e5
    etaP = 0.05

    def smoothmax(r, eps=1e-4):
        return conditional(gt(r, eps), r - eps / 2, conditional(lt(r, 0), 0, r ** 2 / (2 * eps)))

    #objective function
    J += assemble(0.5*gammaP * smoothmax(etaP - tJhat)**2*dx(mesh))

    flag = True
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
      return ReducedFunctional(J,m)
    else:
      if flag:
        return J, dJ
      else:
        return J

def test(Mesh_, param):
    mesh = Mesh_.get_mesh()
    params = Mesh_.get_params()
    boundaries = Mesh_.get_boundaries()
    stop_annotating()
    V1 = VectorElement("CG", mesh.ufl_cell(), 1)
    VC = FunctionSpace(mesh, V1)
    
    tu = interpolate(Expression(("0.0","0.0"), name = 'Control', degree =1), VC)
    J, dJ = reduced_objective(mesh, domains, boundaries, params, param, flag=True)
    
    # compute disturb direction
    g = Expression(("(x[0]-0.2)*(x[0]-0.2)", "(x[1]-0.2)*(x[1]-0.2)"), degree =2)
    zero = Constant(("0.0", "0.0"))
    ds = TestFunction(VC)
    v = TrialFunction(VC)
    a = inner(grad(ds),grad(v))*dx
    l = inner(zero, v)*dx
    bc_inflow = DirichletBC(VC, zero, boundaries, params["inflow"])
    bc_design = DirichletBC(VC, g, boundaries, params["design"])
    bc_noslip = DirichletBC(VC, zero, boundaries, params["noslip"])
    bcs = [bc_inflow, bc_design, bc_noslip]
    
    w = Function(VC)
    solve(a==l, w, bcs)
    
    # epslist
    epslist = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    jlist = []
    
    for eps in epslist:
        #ew = project((eps*w), VC, annotate=False)
        #ewi = project((-eps*w), VC, annotate=False)
        #ALE.move(mesh, ew, annotate=False)
        #jlist.append(reduced_objective(mesh, boundaries, params))
        #ALE.move(mesh, ewi)
        ew = project((eps * w), VC, annotate=False)
        jlist.append(reduced_objective(mesh, domains, boundaries, params, param, control = ew))
    #print(dJ.vector().get_local())
    #print(w.vector().get_local())
    ndof = dJ.vector().size()
    dj = dJ.vector().gather(range(ndof))
    w = w.vector().gather(range(ndof))
    perform_first_order_check(jlist, J, dj, w, epslist)
    
    ## plot solution
    #import matplotlib.pyplot as plt
    #plt.figure()
    #plt.subplot(1,2,1)
    #plot(mesh, color="k", linewidth=0.2, zorder=0)
    #plot(w[0], zorder=1, scale=20)
    #plt.axis("off")
    #plt.subplot(1,2,2)
    #plot(w[1], zorder=1)
    #plt.axis("off")
    #plt.savefig("Output/ReducedObjective/disturb.png", dpi=800, bbox_inches="tight", pad_inches=0)
    return
    
    
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
        di1 = je - j0 - eps*np.dot(gradj0,ds)
        diff0.append(abs(di0))
        diff1.append(abs(di1))
        if i == 0:
            order0.append(0.0)
            order1.append(0.0)
        if i > 0:
            order0.append(np.log(diff0[i-1]/diff0[i])/np.log(epslist[i-1]/epslist[i]))
            order1.append(np.log(diff1[i-1]/diff1[i])/np.log(epslist[i-1]/epslist[i]))
        i = i+1
    for i in range(len(epslist)):
        print('eps\t', epslist[i], '\t\t check continuity\t', order0[i], '\t\t diff0 \t', diff0[i], '\t\t check derivative \t', order1[i], '\t\t diff1 \t', diff1[i], '\n'),
        
    return
    
