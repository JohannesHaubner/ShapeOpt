from dolfin import *
#from dolfin_adjoint import *
from .Constraint import Constraint
from pathlib import Path
here = Path(__file__).parent.resolve()
import sys
sys.path.insert(0, str(here.parent.parent) + "/shapeopt")
import Control_to_Trafo.dof_to_trafo as ctt

class Determinant(Constraint):
    def __init__(self, Mesh_, eta):
        # Consider constraint of the form volume >= V
        super().__init__(Mesh_, param)
        self.eta = eta
        self.scalingfactor = 1.0
        
    def eval(self,x):
        # x dof
        # evaluate g(x) = V - volume(x)
        deformation = ctt.Extension(self.Mesh_).dof_to_deformation_precond(x)
        dF = Identity(self.dim) + grad(deformation)
        Jhat = det(dF)
        vol = self.scalingfactor * (assemble(1.0/self.eta*(1/Jhat - 1/self.eta*Constant('1.0'))*dx))
        return vol
    
    def grad(self,x):
        deformation = ctt.Extension(self.Mesh_).dof_to_deformation_precond(x)
        dF = Identity(self.dim) + grad(deformation)
        Jhat = det(dF)
        form = 1.0/self.eta*(1/Jhat - 1/self.eta*Constant('1.0'))*dx
        dform = assemble(derivative(form, deformation))
        dvolx = self.scalingfactor*ctt.Extension(self.Mesh_).dof_to_deformation_precond_chainrule(dform, 2)
        return dvolx
