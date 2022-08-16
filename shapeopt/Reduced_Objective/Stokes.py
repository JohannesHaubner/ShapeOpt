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
from pyadjoint.overloaded_type import create_overloaded_object
import matplotlib.pyplot as plt
from .ReducedObjective import ReducedObjective

stop_annotating()

class Stokes(ReducedObjective):
    def __init__(self):
        super().__init__()

    def eval(self, mesh, domains, boundaries, params, param, flag=False, red_func=False, control=False):
        # mesh generated
        # params dictionary, includes labels for boundary parts:
        # params.inflow
        # params.outflow
        # params.noslip
        # params.design

        #parameters["adjoint"]["stop_annotating"] = False

        stop_annotating()
        set_working_tape(Tape())
        annotate_tape()

        dim = mesh.geometric_dimension()


        # function spaces
        V2 = VectorElement("CG", mesh.ufl_cell(), dim)
        V1 = VectorElement("CG", mesh.ufl_cell(), 1)
        S1 = FiniteElement("CG", mesh.ufl_cell(), 1)
        VP = FunctionSpace(mesh, V2*S1)
        VC = FunctionSpace(mesh, V1)
        VU = FunctionSpace(mesh, V2)


        # Expressions
        zero = Constant([0.0]*dim)
        (x, y) = SpatialCoordinate(mesh)
        g = Expression(("4/(H*H)*x[1]*(H-x[1])", "0"), degree=2, H=param["H"])


        tu = interpolate(Expression(("0.0","0.0"), name = 'Control', degree =1), VC)
        if control:
            tu.vector().set_local(control.vector().get_local())
            tu.vector().apply("")
            if flag == True:
                print(tu.vector().get_local())

        # test and trial functions
        w = Function(VP, name = "Mixed State Solutions")
        (u,p) = split(w)
        (v, q) = TestFunctions(VP)

        # weak form
        tFhat = Identity(dim) + grad(tu)
        tFhati = inv(tFhat)
        tJhat = det(tFhat)
        F = inner(grad(u)*tFhati, grad(v)*tFhati)*tJhat*dx(mesh) - tr(grad(u)*tFhati)*q*tJhat*dx(mesh) \
            - tr(grad(v)*tFhati)*p*tJhat*dx(mesh) - inner(zero, v)*tJhat*dx(mesh)


        # boundary conditions
        bc_inflow = DirichletBC(VP.sub(0), g, boundaries, params["inflow"])
        bc_design = DirichletBC(VP.sub(0), zero, boundaries, params["design"])
        bc_noslip = DirichletBC(VP.sub(0), zero, boundaries, params["noslip"])
        bcs = [bc_inflow, bc_design, bc_noslip]

        # solve equations

        solve(F == 0, w, bcs) #, solver_parameters={'newton_solver':{'linear_solver':'mumps'}})
        stop_annotating()
        u, p = w.split()

        # plot solution to check
        # save to pvd file for testing
        ufile = File("./Output/Forward/velocity.pvd")
        up = project(u, VU)
        ufile << up
        #plt.figure()
        #plot(u[0])
        #plt.show()

        gammaP = 1e5
        etaP = 0.05

        def smoothmax(r, eps=1e-4):
            return conditional(gt(r, eps), r - eps / 2, conditional(lt(r, 0), 0, r ** 2 / (2 * eps)))

        #objective function
        J = assemble(inner(grad(u)*tFhati, grad(u)*tFhati)*tJhat*dx(mesh)
                     + 0.5*gammaP * smoothmax(etaP - tJhat)**2*dx(mesh))
        if flag:
          dJ = compute_gradient(J, Control(tu))

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
        stop_annotating()
        if red_func:
          m = Control(tu)
          return ReducedFunctional(J, m)
        else:
          if flag:
            return J, dJ
          else:
            return J

