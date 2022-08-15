#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 15:48:29 2020

@author: haubnerj
"""

from dolfin import *
#from dolfin_adjoint import *
import numpy as np
import src.Control_to_Trafo.dof_to_trafo as ctt

class Volume_Constraint():
    def __init__(self, Mesh_, param):
        # Consider constraint of the form volume >= V
        self.Mesh_ = Mesh_
        self.dim = self.Mesh_.mesh.geometric_dimension()
        self.Vd = Mesh_.get_Vd()
        self.Vn = Mesh_.get_Vn()
        self.V = param["Vol_DmO"]
        self.param = param
        self.scalingfactor = 1.0
        
    def eval(self,x):
        # x dof
        # evaluate g(x) = V - volume(x)
        deformation = ctt.Extension(self.Mesh_, self.param).dof_to_deformation_precond(x)
        dF = Identity(self.dim) + grad(deformation)
        Jhat = det(dF)
        vol = self.scalingfactor * (assemble(Jhat*dx) -self.V)
        return vol
    
    def grad(self,x):
        deformation = ctt.Extension(self.Mesh_, self.param).dof_to_deformation_precond(x)
        form = det(Identity(self.dim)+grad(deformation))*dx
        dform = assemble(derivative(form, deformation))
        dvolx = self.scalingfactor*ctt.Extension(self.Mesh_, self.param).dof_to_deformation_precond_chainrule(dform, 2)
        return dvolx
    
    def test(self):
        # check volume and gradient computation with first order derivative check
        print('Volume_Constraint.test started................................')
        x0 = interpolate(Constant(0.01), self.Vd).vector().get_local()
        ds = interpolate(Constant(100.0), self.Vd).vector().get_local()
        #ds = interpolate(Expression('0.2*x[0]', degree=1), self.Vd)
        j0 = self.eval(self.Mesh_.vec_to_Vd(x0))
        djx = self.grad(self.Mesh_.vec_to_Vd(x0))
        epslist = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        ylist = [self.Mesh_.vec_to_Vd(x0+eps*ds) for eps in epslist]
        jlist = [self.eval(y) for y in ylist]
        ds_ = ds#.vector().get_local()
        self.perform_first_order_check(jlist, j0, djx, ds_, epslist)
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
        di1 = je - j0 - eps*np.dot(gradj0,ds)
        diff0.append(abs(di0))
        diff1.append(abs(di1))
        if i == 0:
            order0.append(0.0)
            order1.append(0.0)
        if i > 0:
            order0.append(np.log(diff0[i-1]/diff0[i])/ np.log(epslist[i-1]/epslist[i]))
            order1.append(np.log(diff1[i-1]/diff1[i])/ np.log(epslist[i-1]/epslist[i]))
        i = i+1
      for i in range(len(epslist)):
        print('eps\t', epslist[i], '\t\t check continuity\t', order0[i], '\t\t diff0 \t', diff0[i], '\t\t check derivative \t', order1[i], '\t\t diff1 \t', diff1[i], '\n'),
        
      return
