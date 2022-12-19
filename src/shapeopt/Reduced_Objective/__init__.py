from .Stokes import Stokes
from .FluidStructure import FluidStructure

reduced_objectives = {
    'fluid_structure': FluidStructure(),
    'stokes': Stokes(),
}

