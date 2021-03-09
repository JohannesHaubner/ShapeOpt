#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 08:39:51 2020

@author: Johannes Haubner
"""

from dolfin import *
from dolfin_adjoint import *
import numpy as np
from pyadjoint import annotate_tape, stop_annotating
import matplotlib.pyplot as plt

stop_annotating

def reduced_objective(mesh, boundaries, params, flag =False, red_func = False):
    # mesh generated 
    # params dictionary, includes labels for boundary parts:
    # params.inflow
    # params.outflow
    # params.noslip
    # params.design
    
    #parameters["adjoint"]["stop_annotating"] = False
    
    annotate_tape
    set_working_tape(Tape())
    
    dim = mesh.geometric_dimension()
    
    # function spaces
    V2 = VectorElement("CG", mesh.ufl_cell(), dim)
    V1 = VectorElement("CG", mesh.ufl_cell(), 1)
    S1 = FiniteElement("CG", mesh.ufl_cell(), 1)
    VP = FunctionSpace(mesh, V2*S1)
    VC = FunctionSpace(mesh, V1)
    #
    tu = interpolate(Expression(("0.0","0.0"), name = 'Control', degree =1), VC)
    
    # test and trial functions
    w = Function(VP, name = "Mixed State Solutions")
    (u,p) = split(w)
    (v, q) = TestFunctions(VP)
    
    # Expressions
    (x,y) = SpatialCoordinate(mesh)
    g = Expression(("0.1*x[1]*(6-x[1])", "0"), degree = 2)
    zero = Constant([0]*dim)
    
    # weak form
    tFhat = Identity(dim) + grad(tu)
    tFhati = inv(tFhat)
    tJhat = det(tFhat)
    F = inner(grad(u)*tFhati, grad(v)*tFhati)*tJhat*dx - tr(grad(u)*tFhati)*q*tJhat*dx - tr(grad(v)*tFhati)*p*tJhat*dx - inner(zero, v)*tJhat*dx
    
    print(params["inflow"])
    
    # boundary conditions
    bc_inflow = DirichletBC(VP.sub(0), g, boundaries, params["inflow"])
    bc_design = DirichletBC(VP.sub(0), zero, boundaries, params["design"])
    bc_noslip = DirichletBC(VP.sub(0), zero, boundaries, params["noslip"])
    bcs = [bc_inflow, bc_design, bc_noslip]
    
    # solve equations
    
    solve(F==0, w, bcs)
    u, p = w.split()

    # plot solution to check
    #plt.figure()
    #plot(u[0])
    #plt.show()

    gammaP = 1e5
    etaP = 0.05

    def smoothmax(r, eps=1e-4):
        return conditional(gt(r, eps), r - eps / 2, conditional(lt(r, 0), 0, r ** 2 / (2 * eps)))
    
    #objective function
    J=assemble(inner(grad(u)*tFhati, grad(u)*tFhati)*tJhat*dx) #+0.5*gammaP * smoothmax(etaP - tJhat)**2*dx)  
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

def test(mesh, boundaries, params):
    stop_annotating
    V1 = VectorElement("CG", mesh.ufl_cell(), 1)
    VC = FunctionSpace(mesh, V1)
    
    tu = interpolate(Expression(("0.0","0.0"), name = 'Control', degree =1), VC)
    J, dJ = reduced_objective(mesh, boundaries, params, True)
    
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
        ew = project((eps*w), VC, annotate=False)
        ewi = project((-eps*w), VC, annotate=False)
        ALE.move(mesh, ew, annotate=False)
        jlist.append(reduced_objective(mesh, boundaries, params))
        ALE.move(mesh, ewi)
    #print(dJ.vector().get_local())
    #print(w.vector().get_local())
    perform_first_order_check(jlist, J, dJ.vector().get_local(), w.vector().get_local(), epslist)
    
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
    

