from .Stokes import Stokes
from .FluidStructure import FluidStructure
from .FluidStructure_harmonic import FluidStructure as FluidStructure_harmonic

reduced_objectives = {
    'fluid_structure': FluidStructure,
    'fluid_structure_harmonic': FluidStructure_harmonic,
    'stokes': Stokes,
}

