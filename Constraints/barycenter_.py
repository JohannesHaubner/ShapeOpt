#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 15:48:29 2020

@author: haubnerj
"""

from dolfin import *
#from dolfin_adjoint import *
import numpy as np
import Control_to_Trafo.dof_to_trafo as ctt

class Barycenter_Constraint():
    def __init__(self, Mesh_, param):
        # Volume_mesh_and_obstacle is a constant that has to be given manually and
        # describes the volume of the hold all domain (mesh + obstacle to be optimized)
        # Barycenter_mesh_and_obstacle is a vector that describes the barycenter
        # of the hold all domain
        self.scalingfactor = 1.0
        self.Mesh_ = Mesh_
        self.mesh = Mesh_.get_mesh()
        self.dim = self.Mesh_.mesh.geometric_dimension()
        self.Vd = Mesh_.get_Vd()
        self.Vn = Mesh_.get_Vn()
        self.V = Mesh_.get_V()
        self.param = param
        self.Bary_O = param["Bary_O"]
        self.Vol = param["Vol_DmO"]
        self.L = param["L"]
        self.H = param["H"]

        
    def eval(self,y):
        L = self.L
        H = self.H
        deformation = ctt.Extension(self.Mesh_, self.param).dof_to_deformation_precond(y)
        x = SpatialCoordinate(self.mesh)
        dF = Identity(self.dim) + grad(deformation)
        Jhat = det(dF)
        bc1 = (L**2 * H / 2 - assemble((x[0]+deformation[0])*Jhat * dx))/ (L * H - self.Vol) - self.Bary_O[0]
        bc2 = (L * H**2 / 2 - assemble((x[1]+deformation[1])*Jhat * dx))/ (L * H - self.Vol) - self.Bary_O[1]
        bc = [bc1, bc2]
        return bc
    
    def grad(self,y):
        L = self.L
        H = self.H
        deformation = ctt.Extension(self.Mesh_, self.param).dof_to_deformation_precond(y)
        x = SpatialCoordinate(self.mesh)
        form1 = (x[0]+deformation[0])*det(Identity(self.dim) + grad(deformation)) * dx
        form2 = (x[1]+deformation[1])*det(Identity(self.dim) + grad(deformation)) * dx
        df1 = -1.0/ (L * H - self.Vol)*assemble(derivative(form1, deformation))
        df2 = -1.0/ (L * H - self.Vol)*assemble(derivative(form2, deformation))
        cgf1 = ctt.Extension(self.Mesh_,self.param).dof_to_deformation_precond_chainrule(df1, 2)
        cgf2 = ctt.Extension(self.Mesh_,self.param).dof_to_deformation_precond_chainrule(df2, 2)
        return [cgf1, cgf2]
    
    def test(self):
        # check volume and gradient computation with first order derivative check
        print('Barycenter_Constraint.test started............................')
        x0 = interpolate(Expression("(x[0]-0.2)", degree =1), self.Vd).vector().get_local()
        ds = interpolate(Expression("10000*(x[0]-0.2)", degree =1), self.Vd).vector().get_local()
        #ds = interpolate(Expression('0.2*x[0]', degree=1), self.Vd)
        j0 = self.eval(x0)[0]
        djx = self.grad(x0)[0]
        #print(djx.max())
        epslist = [0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001, 0.0000005, 0.0000001, 0.00000005, 0.00000001]
        ylist = [x0+eps*ds for eps in epslist]
        jlist = [self.eval(y)[0] for y in ylist]
        ds_ = ds
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
