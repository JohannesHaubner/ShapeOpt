from dolfin import *
#from dolfin_adjoint import *
from .Constraint import Constraint
from pathlib import Path
here = Path(__file__).parent.resolve()
import sys
sys.path.insert(0, str(here.parent.parent))
from shapeopt.Tools.first_order_check import perform_first_order_check

class Barycenter(Constraint):
    def __init__(self, Mesh_, param, dof_to_trafo):
        # Volume_mesh_and_obstacle is a constant that has to be given manually and
        # describes the volume of the hold all domain (mesh + obstacle to be optimized)
        # Barycenter_mesh_and_obstacle is a vector that describes the barycenter
        # of the hold all domain
        super().__init__(Mesh_, param, dof_to_trafo)
        self.scalingfactor = 1.0
        self.Bary_O = param["Bary_O"]
        self.Vol = param["Vol_DmO"]
        self.L = param["L"]
        self.H = param["H"]
        self.dim = Mesh_.mesh.topology().dim()
        self.dx = Measure('dx', subdomain_data=Mesh_.domains, metadata={"quadrature_degree": 10})
        if self.dim == 3:
            self.B = param["B"]

    def output_dim(self):
        return self.mesh.geometry().dim()
        
    def eval(self, y):
        L = self.L
        H = self.H
        deformation = self.dof_to_trafo.dof_to_deformation_precond(y)
        x = SpatialCoordinate(self.mesh)
        dF = Identity(self.dim) + grad(deformation)
        Jhat = det(dF)
        if self.dim == 2:
            bc1 = (L**2 * H / 2 - assemble((x[0]+deformation[0])*Jhat * self.dx))/ (L * H - self.Vol) - self.Bary_O[0]
            bc2 = (L * H**2 / 2 - assemble((x[1]+deformation[1])*Jhat * self.dx))/ (L * H - self.Vol) - self.Bary_O[1]
            bc = [bc1, bc2]
        elif self.dim == 3:
            B = self.B
            bc1 = (L**2 * H * B / 2 - assemble((x[0]+deformation[0])*Jhat * self.dx))/ (L * H * B - self.Vol) - self.Bary_O[0]
            bc2 = (L * H * B**2 / 2 - assemble((x[0]+deformation[0])*Jhat * self.dx))/ (L * H * B - self.Vol) - self.Bary_O[1]
            bc3 = (L * H**2 * B / 2 - assemble((x[1]+deformation[1])*Jhat * self.dx))/ (L * H * B - self.Vol) - self.Bary_O[2]
            bc = [bc1, bc2, bc3]
        return bc
    
    def grad(self, y):
        L = self.L
        H = self.H
        fac = L * H
        if self.dim == 3:
            B = self.B 
            fac = fac * B
        deformation = self.dof_to_trafo.dof_to_deformation_precond(y)
        x = SpatialCoordinate(self.mesh)
        form1 = (x[0]+deformation[0])*det(Identity(self.dim) + grad(deformation)) * self.dx
        form2 = (x[1]+deformation[1])*det(Identity(self.dim) + grad(deformation)) * self.dx
        if self.dim == 3:
            form3 = (x[2]+deformation[2])*det(Identity(self.dim) + grad(deformation)) * self.dx
        df1 = -1.0/ (fac - self.Vol)*assemble(derivative(form1, deformation))
        df2 = -1.0/ (fac - self.Vol)*assemble(derivative(form2, deformation))
        if self.dim == 3:
            df3 = -1.0/ (fac - self.Vol)*assemble(derivative(form3, deformation))
        cgf1 = self.dof_to_trafo.dof_to_deformation_precond_chainrule(df1, 2)
        cgf2 = self.dof_to_trafo.dof_to_deformation_precond_chainrule(df2, 2)
        cgf = [cgf1, cgf2]
        if self.dim == 3:
            cgf.append(self.dof_to_trafo.dof_to_deformation_precond_chainrule(df3, 2))
        return cgf

    def test(self):
        # check eval and gradient computation with first order derivative check (test2 better for Barycenter, e.g.)
        print('Barycenter.Constraint.test started............................')
        x0 = interpolate(Expression("(x[0]-0.2)", degree=1), self.Vd).vector().get_local()
        ds = interpolate(Expression("10000*(x[0]-0.2)", degree=1), self.Vd).vector().get_local()
        # ds = interpolate(Expression('0.2*x[0]', degree=1), self.Vd)
        j0 = self.eval(self.Mesh_.vec_to_Vd(x0))[0]
        djx = self.grad(self.Mesh_.vec_to_Vd(x0))[0]
        # print(djx.max())
        epslist = [0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001, 0.0000005, 0.0000001, 0.00000005, 0.00000001]
        ylist = [self.Mesh_.vec_to_Vd(x0 + eps * ds) for eps in epslist]
        jlist = [self.eval(y)[0] for y in ylist]
        ds_ = ds
        order, diff = perform_first_order_check(jlist, j0, djx, ds_, epslist)
        return order, diff
