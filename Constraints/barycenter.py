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
    def __init__(self, Mesh_, Volume_mesh_and_obstacle, Barycenter_mesh_and_obstacle):
        # Volume_mesh_and_obstacle is a constant that has to be given manually and
        # describes the volume of the hold all domain (mesh + obstacle to be optimized)
        # Barycenter_mesh_and_obstacle is a vector that describes the barycenter
        # of the hold all domain
        self.scalingfactor = 1.0
        self.Mesh_ = Mesh_
        self.dim = self.Mesh_.mesh.geometric_dimension()
        self.Vd = Mesh_.get_Vd()
        self.Vn = Mesh_.get_Vn()
        self.V = Mesh_.get_V()
        self.volume_ref = assemble(interpolate(Constant(1.0), self.V)*dx)
        self.functions_x = []
        self.bary_int = []
        self.barycenter_ref = []
        for i in range(self.dim):
            self.functions_x.append(Expression("x[ci]", ci = i, element = self.V.ufl_element()))
            self.bary_int.append(assemble(self.functions_x[i]*interpolate(Constant(1.0), self.V)*dx))
            self.barycenter_ref.append(1.0/self.volume_ref*self.bary_int[i])
        print(self.volume_ref)
        print(self.barycenter_ref)
        self.Volume_D = Volume_mesh_and_obstacle
        self.Bary_D = Barycenter_mesh_and_obstacle
        self.Volume_obs_ref = self.Volume_D - self.volume_ref
        self.Bary_obs = []
        for i in range(self.dim):
          self.Bary_obs.append(1.0/(self.Volume_D - self.volume_ref)*(self.Bary_D[i]*self.Volume_D - self.bary_int[i]))
        
    def eval(self,x,eps):
        # inequality constraint of the form ||bc_obs(x) - bc_ref||^2 - eps <= 0
        # (bc_ref = bc_obs(0))
        # is reformulated to
        # ||self.Bary_D*self.Volume_D - int_Omega_obs x dx - bc_ref*vol(Omega_obs)||^2 - eps*vol(Omega_obs)^2 <= 0
        # x dof
        # more precisely, we define forms:
        # bary_def[i] is the form to (int_Omega_obs x dx + bc_ref*vol(Omega_obs))[i] 
        # volobs is the form corresponding to vol(Omega_obs)
        deformation = ctt.Extension(self.Mesh_).dof_to_deformation_precond(x)
        #print(deformation.vector().max())
        volume =  det(Identity(self.dim)+grad(deformation))*dx
        volobs = 1.0/Constant(self.volume_ref/self.Volume_D)*interpolate(Constant(1.0), self.V)*dx - volume 
        #print(assemble(volobs), self.Volume_obs_ref)
        bary_def = []
        vol_bary = []
        volobs_bary = []
        sum_bc = 0
        for i in range(self.dim):
          vol_bary.append(self.Bary_obs[i]*det(Identity(self.dim)+grad(deformation))*dx)
          volobs_bary.append(self.Bary_obs[i]/Constant(self.volume_ref/self.Volume_D)*interpolate(Constant(1.0), self.V)*dx - vol_bary[i] )
          bary_def.append((self.functions_x[i] + deformation[i])*det(Identity(self.dim)+grad(deformation))*dx + volobs_bary[i])
          bdi = assemble(bary_def[i])
          sum_bc = sum_bc + (self.Bary_D[i]*self.Volume_D - bdi)**2
        vobs = assemble(volobs)
        sum_bc = sum_bc - vobs*vobs*eps
        return self.scalingfactor*sum_bc
    
    def grad(self,x,eps):
        # inequality constraint of the form ||bc_obs(x) - bc_ref||^2 - eps <= 0
        # (bc_ref = bc_obs(0))
        # is reformulated to
        # ||self.Bary_D*self.Volume_D - int_Omega_obs x dx - bc_ref*vol(Omega_obs)||^2 - eps*vol(Omega_obs)^2 <= 0
        # x dof
        # more precisely, we define forms:
        # bary_def[i] is the form to (int_Omega_obs x dx + bc_ref*vol(Omega_obs))[i] 
        # volobs is the form corresponding to vol(Omega_obs)
        deformation = ctt.Extension(self.Mesh_).dof_to_deformation_precond(x)
        #print(deformation.vector().max())
        volume =  det(Identity(self.dim)+grad(deformation))*dx
        volobs = 1.0/Constant(self.volume_ref/self.Volume_D)*interpolate(Constant(1.0), self.V)*dx - volume 
        #print(assemble(volobs), self.Volume_obs_ref)
        bary_def = []
        vol_bary = []
        volobs_bary = []
        vobs = assemble(volobs)
        der = -2.0*eps*vobs*assemble(derivative(volobs,deformation))
        for i in range(self.dim):
          vol_bary.append(self.Bary_obs[i]*det(Identity(self.dim)+grad(deformation))*dx)
          volobs_bary.append(self.Bary_obs[i]/Constant(self.volume_ref/self.Volume_D)*interpolate(Constant(1.0), self.V)*dx - vol_bary[i] )
          bary_def.append((self.functions_x[i] + deformation[i])*det(Identity(self.dim)+grad(deformation))*dx + volobs_bary[i])
          bdi = assemble(bary_def[i])
          der = der - 2.0*(self.Bary_D[i]*self.Volume_D - bdi)*(assemble(derivative(bary_def[i],deformation)))
          dbcx = ctt.Extension(self.Mesh_).dof_to_deformation_precond_chainrule(der, 2)
        return self.scalingfactor*dbcx
    
    def test(self,tol):
        # check volume and gradient computation with first order derivative check
        print('Barycenter_Constraint.test started............................')
        x0 = interpolate(Expression("(x[0]-0.2)", degree =1), self.Vd).vector().get_local()
        ds = interpolate(Expression("10000*(x[0]-0.2)", degree =1), self.Vd).vector().get_local()
        #ds = interpolate(Expression('0.2*x[0]', degree=1), self.Vd)
        j0 = self.eval(x0,tol)
        djx = self.grad(x0,tol)
        print(djx.max())
        epslist = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
        ylist = [x0+eps*ds for eps in epslist]
        jlist = [self.eval(y,tol) for y in ylist]
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
