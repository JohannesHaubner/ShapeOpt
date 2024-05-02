from dolfin import *
#from dolfin_adjoint import *
from .Constraint import Constraint
from pathlib import Path
here = Path(__file__).parent.resolve()
import sys
sys.path.insert(0, str(here.parent.parent) + "/shapeopt")
import Control_to_Trafo.dof_to_trafo as ctt

class VolumeSolid(Constraint):
    def __init__(self, Mesh_, param, dof_to_trafo):
        # Consider constraint of the form volume >= V
        super().__init__(Mesh_, param, dof_to_trafo)
        self.V = param["Vol_solid"]
        self.scalingfactor = 1.0
        self.dx = Measure('dx', subdomain_data=Mesh_.domains)
        self.id = self.param["solid"]

    def output_dim(self):
        return 1
        
    def eval(self,x):
        # x dof
        # evaluate g(x) = V - volume(x)
        deformation = self.dof_to_trafo.dof_to_deformation_precond(x)
        dF = Identity(self.dim) + grad(deformation)
        Jhat = det(dF)
        vol = self.scalingfactor * (assemble(Jhat*self.dx(self.id)) -self.V)
        return vol
    
    def grad(self,x):
        deformation = self.dof_to_trafo.dof_to_deformation_precond(x)
        form = det(Identity(self.dim)+grad(deformation))*self.dx(self.id)
        dform = assemble(derivative(form, deformation))
        dvolx = self.scalingfactor*self.dof_to_trafo.dof_to_deformation_precond_chainrule(dform, 2)
        return dvolx

