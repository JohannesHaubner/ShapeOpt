#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 08:45:52 2020

@author: haubnerj
"""
from dolfin import *
#from dolfin_adjoint import *
import numpy as np
import matplotlib.pyplot as plt

class Extension():
    def __init__(self, Mesh_):
      # mesh: reference mesh
      # boundaries: labels for boundary
      # dmesh: design boundary mesh
      # Vd: function space in which the control lives
      # params: params.design, params.inflow, params.outflow, params.noslip
      mesh = Mesh_.get_mesh()
      dmesh = Mesh_.get_design_boundary_mesh()
      boundaries = Mesh_.get_boundaries()
      params = Mesh_.get_params()
      
      # normal vector on mesh
      n = Mesh_.get_n()
      
      self.Mesh_ = Mesh_
      self.mesh = mesh
      
      self.dmesh = dmesh

      # Laplace Beltrami off
      self.lb_off = Constant('0.1') #1.0') #1.0 on, 0.0 off
      
      # define function spaces
      self.V = Mesh_.get_V()
      self.Vd = Mesh_.get_Vd()
      self.Vn = Mesh_.get_Vn()
      self.Vdn = Mesh_.get_Vdn()
      self.dnormalf = Mesh_.get_dnormalf()
      self.params = params
      self.boundaries = boundaries
      self.ds = Measure("ds", subdomain_data=boundaries)
      
      # lumped mass matrix for IPOPT
      v = TestFunction(self.Vd)
      u = TrialFunction(self.Vd)
      mass_form = v*u*dx(self.dmesh)
      mass_action_form = action(mass_form, interpolate(Constant(1.0),self.Vd))
      M_lumped = assemble(mass_form)
      M_lumped_m05= assemble(mass_form)
      M_lumped.zero()
      M_lumped_m05.zero()
      M_diag = assemble(mass_action_form)
      M_diag_m05 = assemble(mass_action_form)
      M_diag_m05.set_local(np.ma.power(M_diag.get_local(), -0.5))
      M_diag_m05.apply("")
      M_lumped.set_diagonal(M_diag)
      M_lumped_m05.set_diagonal(M_diag_m05)
      self.M_lumped = M_lumped
      self.M_lumped_m05 = M_lumped_m05
      ## test matrix
      #func = interpolate(Constant(1.0), self.Vd)
      #print((self.M_lumped_m05 * func.vector()).get_local())
      #exit(0)
      self.mu = Function(self.V)
      self.mu.assign(self.parameter_lin_elas()) #Constant("1.0")) #self.parameter_lin_elas())

    def parameter_lin_elas(self):
      # parameter for linear extension equation
      u = TrialFunction(self.V)
      v = TestFunction(self.V)
      bc1 = DirichletBC(self.V, Constant("0.1"), self.boundaries, self.params["inflow"])
      bc2 = DirichletBC(self.V, Constant( "0.1"), self.boundaries, self.params["outflow"])
      bc3 = DirichletBC(self.V, Constant( "0.1"), self.boundaries, self.params["noslip"])
      bc4 = DirichletBC(self.V, Constant("500.0"), self.boundaries, self.params["design"])
      bc = [bc1, bc2, bc3, bc4]

      # Define bilinear form
      a = inner(grad(u), grad(v)) * dx(self.mesh)  # + inner(u,v)*dx
      # Define linear form
      L =Constant("0.0")*v* dx(self.mesh)

      # solve variational problem
      u = Function(self.V)
      solve(a == L, u, bc)
      return u
      
    def vec_to_func_precond_chainrule(self, v):
      vc = self.M_lumped_m05 * v.vector()
      #print(v.vector())
      vcf = Function(self.Vd)
      vcf.vector().set_local(vc)
      vcf.vector().apply("")
      x = self.Mesh_.Vd_to_vec(vcf)
      return x
  
    def vec_to_func_precond(self,v):
      """ takes a Vd function, and multiplies with (lumped mass matrix)^0.5 """
      #v = self.Mesh_.vec_to_Vd(x)
      vc = self.M_lumped_m05 * v.vector()
      v.vector().set_local(vc.get_local())
      v.vector().apply("")
      return v
  
    def dof_to_deformation_precond(self,x):
      xd = self.vec_to_func_precond(x)
      v = self.dof_to_deformation(xd)
      return v
  
    def dof_to_deformation_precond_chainrule(self,djy, option2):
      djy = self.dof_to_deformation_chainrule(djy, option2)
      djyy = Function(self.Vd)
      djyy.vector().set_local(djy)
      djyy.vector().apply("")
      djx = self.vec_to_func_precond_chainrule(djyy)
      return djx
  
    def test_dof_to_deformation_precond(self):
      # check dof_to_deformation with first order derivative check
      #print('Extension.test_dof_to_deformation started.......................')
      xl = self.dmesh.num_vertices()
      x0 = 0.5*np.ones(xl)
      ds = 1.0*np.ones(xl)
      #ds = interpolate(Expression('0.2*x[0]', degree=1), self.Vd)
      y0 = self.dof_to_deformation_precond(self.Mesh_.vec_to_Vd(x0))
      j0 = assemble(0.5*inner(y0,y0)*dx)
      djy = y0
      djx = self.dof_to_deformation_precond_chainrule(djy, 1)
      epslist = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
      ylist = [self.dof_to_deformation_precond(self.Mesh_.vec_to_Vd(x0+eps*ds)) for eps in epslist]
      jlist = [assemble(0.5*inner(y, y)*dx) for y in ylist]
      self.perform_first_order_check(jlist, j0, djx, ds, epslist)
      return
      
      
    def dof_to_deformation(self, x):
      # x: corresponds to control in self.Vd
      xd = self.vector_laplace_beltrami(x)
      deformation = self.linear_elasticity(xd)
      return deformation
  
    def dof_to_deformation_chainrule(self,djy, option2):
      # compute derivative of j(dof_to_deformation(x)) under the knowledge of
      # djy = nabla j(y) (gradient) (if option2 ==1) or derivative j'(y) (if option2 == 2)
      djxd = self.linear_elasticity_chainrule(djy,1, option2)
      djy = self.vector_laplace_beltrami_chainrule(djxd)
      return djy
  
    def test_dof_to_deformation(self):
      # check dof_to_deformation with first order derivative check
      #print('Extension.test_dof_to_deformation started.......................')
      x0 = interpolate(Constant(0.5), self.Vd)
      ds = interpolate(Constant(1.0), self.Vd)
      #ds = interpolate(Expression('0.2*x[0]', degree=1), self.Vd)
      y0 = self.dof_to_deformation(x0)
      j0 = assemble(0.5*inner(y0,y0)*dx(self.mesh))
      print(j0)
      exit(0)
      djy = y0
      djx = self.dof_to_deformation_chainrule(djy,1).get_local()
      epslist = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
      ylist = [self.dof_to_deformation(x0+eps*ds) for eps in epslist]
      jlist = [assemble(0.5*inner(y, y)*dx) for y in ylist]
      ds_ = ds.vector().get_local()
      self.perform_first_order_check(jlist, j0, djx, ds_, epslist)
      return
     
        
      
    def test_design_boundary_mesh(self):
      # this test is used to check if in 2d the length of the design boundary is
      # correctly obtained
      print('----------------------------------------------------------------')
      print('Extension.test_design_boundary_mesh started.....................')
      v = interpolate(Constant(("1.0")), self.Vd)
      print('output: ', assemble(inner(v,v)*dx))
      print('test finished...................................................')
      pass

      
      
    def linear_elasticity(self,x):
      # x: corresponds to vector valued function in self.Vdn
      #print('Extension.vector_laplace_beltrami started.......................')
      
      parameters["form_compiler"]["cpp_optimize"] = True
      parameters["form_compiler"]["optimize"] = True
      
      # Define trial and test functions
      u = TrialFunction(self.Vn)
      v = TestFunction(self.Vn)
      
      # project x on domain function
      xd = self.Mesh_.Vdn_to_Vn(x)
      
      # Define boundary conditions
      bc1 = DirichletBC(self.Vn, Constant(("0.0","0.0")), self.boundaries, self.params["inflow"])
      bc2 = DirichletBC(self.Vn, Constant(("0.0","0.0")), self.boundaries, self.params["outflow"])
      bc3 = DirichletBC(self.Vn, Constant(("0.0","0.0")), self.boundaries, self.params["noslip"])
      bc = [bc1, bc2, bc3]
      
      # Define bilinear form
      a = self.mu*0.5*inner(grad(u) + np.transpose(grad(u)), grad(v))*dx #+ inner(u,v)*dx
      # Define linear form
      L = inner(xd,v)*self.ds(self.params["design"])
      
      # solve variational problem
      u = Function(self.Vn)
      solve(a == L, u, bc)

      return u
  
    def test_linear_elasticity(self):
      print('----------------------------------------------------------------')
      print('Extension.test_linear_elasticity started........................')

      x0 = interpolate(Constant(("1.0","0.5")), self.Vdn)
      ds = interpolate(Constant(("0.5","0.2")), self.Vdn)
      y0 = self.linear_elasticity(x0)
      j0 = assemble(0.5*inner(y0,y0)*dx)
      djy = y0
      djx = self.linear_elasticity_chainrule(djy,2,1)
      epslist = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
      ylist = []
      for eps in epslist:
          xeps = Function(self.Vdn)
          xeps.assign(x0)
          xeps.vector().axpy(eps, ds.vector())
          ylist.append(self.linear_elasticity(xeps))
      #ylist = [self.linear_elasticity(x0+eps*ds) for eps in epslist]
      jlist = [assemble(0.5*inner(y, y)*dx) for y in ylist]
      ds_ = ds.vector().get_local()
      self.perform_first_order_check(jlist, j0, djx.get_local(), ds_, epslist)
      return
  
    def linear_elasticity_chainrule(self,djy, option, option2): 
      # compute derivative of j(linear_elasticity(x)) under the knowledge of
      # djy:
      # option == 1: gradient  nabla j(x)
      # option == 2: deriative j'(x)
      #
      # option2 == 1: djy is gradient 
      # option2 == 2: djy is derivative
      #print('Extension.vector_laplace_beltrami_chainrule started.............')
      
      parameters["form_compiler"]["cpp_optimize"] = True
      parameters["form_compiler"]["optimize"] = True
      
      # solve adjoint equations
      u = TrialFunction(self.Vn)
      v = TestFunction(self.Vn)
      
      # Define boundary conditions
      bc1 = DirichletBC(self.Vn, Constant(("0.0","0.0")), self.boundaries, self.params["inflow"])
      bc2 = DirichletBC(self.Vn, Constant(("0.0","0.0")), self.boundaries, self.params["outflow"])
      bc3 = DirichletBC(self.Vn, Constant(("0.0","0.0")), self.boundaries, self.params["noslip"])
      bc = [bc1, bc2, bc3]
      
      # Define bilinear form
      a = self.mu*0.5*inner(grad(u) + np.transpose(grad(u)), grad(v))*dx #+ inner(u,v)*dx
      if option2 ==1:
        # define linear form
        L = inner(djy,v)*dx
        # solve variational problem
        u = Function(self.Vn)
        solve(a==L, u, bc)
      elif option2 == 2:
        A = assemble(a)
        bc1.apply(A)
        bc2.apply(A)
        bc3.apply(A)
        u = Function(self.Vn)
        dj = Function(self.Vn)
        dj.vector().set_local(djy)
        dj.vector().apply("")
        djy = dj.vector()
        bc1.apply(djy)
        bc2.apply(djy)
        bc3.apply(djy)
        solve(A, u.vector(), djy)
      if option == 1:
        djx = self.Mesh_.Vn_to_Vdn(u)
        return djx
      elif option == 2:
        xt = TrialFunction(self.Vn)
        ud = assemble(inner(u,xt)*self.ds(self.params["design"]))
        u = Function(self.Vn)
        u.vector()[:] = ud
        djx = self.Mesh_.Vn_to_Vdn(u)
        return djx.vector()
  
  
    def vector_laplace_beltrami(self,x):   
      # x: corresponds to control in self.Vd
      #print('Extension.vector_laplace_beltrami started.......................')
      
      parameters["form_compiler"]["cpp_optimize"] = True
      parameters["form_compiler"]["optimize"] = True

      ds = self.Mesh_.get_ds()
      
      # Define trial and test functions
      u = TrialFunction(self.Vdn)
      v = TestFunction(self.Vdn)
      
      # Define boundary conditions
      bc = []
      
      # Define bilinear form
      a = self.lb_off*inner(grad(u), grad(v))*dx(self.dmesh) + inner(u,v)*dx(self.dmesh)
      # Define linear form
      L = inner(x*self.dnormalf,v)*dx(self.dmesh)
      
      # solve variational problem
      u = Function(self.Vdn)
      solve(a == L, u, bc)
      return u

    def vector_laplace_beltrami_chainrule(self,djy):
      # compute derivative of j(vector_laplace_beltrami(x)) under the knowledge of
      # djy = nabla j(y) (gradient)
      #print('Extension.vector_laplace_beltrami_chainrule started.............')
      
      parameters["form_compiler"]["cpp_optimize"] = True
      parameters["form_compiler"]["optimize"] = True

      # solve adjoint equation
      v = TrialFunction(self.Vdn)
      z = TestFunction(self.Vdn)
      
      # define boundary conditions
      bc = []
      
      # Define bilinear form
      a = self.lb_off*inner(grad(v), grad(z))*dx(self.dmesh) + inner(v,z)*dx(self.dmesh)
      # Define linear form
      L = inner(djy, z)*dx(self.dmesh)
      
      # solve variational problem
      v = Function(self.Vdn)
      solve(a == L, v, bc)
      
      # evaluate dj/dx
      xt = TrialFunction(self.Vd)
      djx = assemble(inner(v, self.dnormalf)*xt*dx(self.dmesh))
      
      return djx
  
    def test_vector_laplace_beltrami(self):
      # check laplace beltrami equation with first order derivative check
      print('Extension.test_vector_laplace_beltrami started..................')
      x0 = interpolate(Constant(0.15), self.Vd)
      ds = interpolate(Constant(1.0), self.Vd)
      #ds = interpolate(Expression('0.2*x[0]', degree=1), self.Vd)
      y0 = self.vector_laplace_beltrami(x0)
      bdfile = File(MPI.comm_self, "./Output/Tests/vectorLaplaceBeltrami.pvd")
      bdfile << self.Mesh_.Vdn_to_Vn(y0)
      j0 = assemble(0.5 * inner(y0, y0) * dx(self.dmesh))
      # correction since assemble adds up values for all processes
      rank = MPI.comm_world.Get_size()
      j0 = j0/rank
      djy = y0
      djx = self.vector_laplace_beltrami_chainrule(djy).get_local()
      epslist = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
      ylist = [self.vector_laplace_beltrami(x0+eps*ds) for eps in epslist]
      jlist = [assemble(0.5*inner(y, y)*dx)/rank for y in ylist] #includes correction
      ds_ = ds.vector().get_local()
      self.perform_first_order_check(jlist, j0, djx, ds_, epslist)
      return
  
    def test_dof_to_precond_Vd_to_vector_laplace_beltrami(self):
      print('Extension.test_dof_to_precond_Vd_to_vector_laplace_beltrami started...')
      x0 = Function(self.Vd)
      n = x0.vector().size()
      x0 = 0.5*np.ones(n,1)
      ds = 1.0*np.ones(n,1)
      y0 = self.vector_laplace_beltrami()
      pass
      
      
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
      
