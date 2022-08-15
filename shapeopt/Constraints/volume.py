from dolfin import *
#from dolfin_adjoint import *
from .Constraint import Constraint
from pathlib import Path
here = Path(__file__).parent.resolve()
import sys
sys.path.insert(0, str(here.parent.parent) + "/shapeopt")
import Control_to_Trafo.dof_to_trafo as ctt

class Volume(Constraint):
    def __init__(self, Mesh_, param, boundary_option, extension_option):
        # Consider constraint of the form volume >= V
        super().__init__(Mesh_, param, boundary_option, extension_option)
        self.V = param["Vol_DmO"]
        self.scalingfactor = 1.0
        
    def eval(self,x):
        # x dof
        # evaluate g(x) = V - volume(x)
        deformation = ctt.Extension(self.Mesh_, self.param, self.boundary_option, self.extension_option).dof_to_deformation_precond(x)
        dF = Identity(self.dim) + grad(deformation)
        Jhat = det(dF)
        vol = self.scalingfactor * (assemble(Jhat*dx) -self.V)
        return vol
    
    def grad(self,x):
        deformation = ctt.Extension(self.Mesh_, self.param, self.boundary_option, self.extension_option).dof_to_deformation_precond(x)
        form = det(Identity(self.dim)+grad(deformation))*dx
        dform = assemble(derivative(form, deformation))
        dvolx = self.scalingfactor*ctt.Extension(self.Mesh_, self.param, self.boundary_option, self.extension_option).dof_to_deformation_precond_chainrule(dform, 2)
        return dvolx

