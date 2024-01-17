from dolfin import *
import numpy as np

from .Boundary_Operator import boundary_operators
from .Extension_Operator import extension_operators

from pathlib import Path
here = Path(__file__).parent
import sys
sys.path.insert(0, str(here.parent))
from Tools.first_order_check import perform_first_order_check

class Extension():
    def __init__(self, Mesh_, param, boundary_operator, extension_operator):
      """
      # mesh: reference mesh
      # params: params.design, params.inflow, params.outflow, params.noslip
      # boundary_option: boundary operator 
      # extension_option: extension operator 
      """
      mesh = Mesh_.get_mesh()
      dmesh = Mesh_.get_design_boundary_mesh()
      boundaries = Mesh_.get_boundaries()
      domains = Mesh_.get_domains()
      params = Mesh_.get_params()
      
      # normal vector on mesh
      n = Mesh_.get_n()
      
      self.Mesh_ = Mesh_
      self.mesh = mesh
      
      self.dmesh = dmesh

      # Laplace Beltrami off
      lb_off_p = param["lb_off_p"]
      self.lb_off = Expression(("lb_off"), degree = 0, lb_off = lb_off_p) #Constant("0.1") #Constant('0.0') # 0.0 = multiply with n, 1.0 = Laplace Beltrami
      
      # define function spaces
      self.V = Mesh_.get_V()
      self.Vd = Mesh_.get_Vd()
      self.Vn = Mesh_.get_Vn()
      self.Vdn = Mesh_.get_Vdn()
      self.dnormalf = Mesh_.get_dnormalf()
      self.params = params
      self.domains = domains
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

      self.boundary_operator = boundary_operator
      self.extension_operator = extension_operator

    @staticmethod
    def print_options():
      print('Options for boundary_option:')
      for key, value in boundary_operators.items():
        print('...', key)
      print('\nOptions for extension_option:')
      for key, value in extension_operators.items():
        print('...', key)

    def dof_to_deformation(self, x):
      # x: corresponds to control in self.Vd
      #xd = boundary.Boundary_Operator(self.dmesh, self.dnormalf, self.lb_off).eval(x)
      #xd = self.Mesh_.Vdn_to_Vn(xd)
      #xd = extension.Extension(self.mesh, self.boundaries, self.params).eval(xd)
      #deformation = extension.Extension(self.mesh, self.boundaries, self.params).eval(xd)

      ## strategy 3
      xd = self.boundary_operator.eval(x) #self.lb_off).eval(x)
      xd = self.Mesh_.Vdn_to_Vn(xd)
      deformation = self.extension_operator.eval(xd)
      ###
      return deformation

    def dof_to_deformation_chainrule(self, djy, option2):
      # compute derivative of j(dof_to_deformation(x)) under the knowledge of
      # djy = nabla j(y) (gradient) (if option2 ==1) or derivative j'(y) (if option2 == 2)
      #djxd = extension.Extension(self.mesh, self.boundaries, self.params).chainrule(djy, 2, option2)
      #djxd = extension.Extension(self.mesh, self.boundaries, self.params).chainrule(djxd.vector(), 1, 2)
      #djxdf = self.Mesh_.Vn_to_Vdn(djxd)
      #djy = boundary.Boundary_Operator(self.dmesh, self.dnormalf, self.lb_off).chainrule(djxdf)

      ### strategy 3
      djxd = self.extension_operator.chainrule(djy, 1, option2)
      djxdf = self.Mesh_.Vn_to_Vdn(djxd)
      djy = self.boundary_operator.chainrule(djxdf)
      ###
      return djy


    def vec_to_func_precond_chainrule(self, v):
      vc = self.M_lumped_m05 * v.vector()
      #print(v.vector())
      vcf = Function(self.Vd)
      vcf.vector().set_local(vc)
      vcf.vector().apply("")
      x = self.Mesh_.Vd_to_vec(vcf)
      return x
  
    def vec_to_func_precond(self, v):
      """ takes a Vd function, and multiplies with (lumped mass matrix)^0.5 """
      #v = self.Mesh_.vec_to_Vd(x)
      vc = self.M_lumped_m05 * v.vector()
      v.vector().set_local(vc.get_local())
      v.vector().apply("")
      return v
  
    def dof_to_deformation_precond(self, x):
      xd = self.vec_to_func_precond(x)
      v = self.dof_to_deformation(xd)
      return v
  
    def dof_to_deformation_precond_chainrule(self, djy, option2):
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
      order, diff = perform_first_order_check(jlist, j0, djx, ds, epslist)
      return order, diff

  
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
      perform_first_order_check(jlist, j0, djx, ds_, epslist)
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

      

  
    def test_dof_to_precond_Vd_to_vector_laplace_beltrami(self):
      print('Extension.test_dof_to_precond_Vd_to_vector_laplace_beltrami started...')
      x0 = Function(self.Vd)
      n = x0.vector().size()
      x0 = 0.5*np.ones(n,1)
      ds = 1.0*np.ones(n,1)
      y0 = self.vector_laplace_beltrami()
      pass

      
