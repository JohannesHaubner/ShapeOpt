from .LaplaceBeltrami import LaplaceBeltrami
from .LaplaceBeltrami_withbc import LaplaceBeltrami_withbc
from .LaplaceBeltrami_withbc2 import LaplaceBeltrami_withbc2 
from .NoBCOperator import NoBCO

boundary_operators = {'laplace_beltrami': LaplaceBeltrami,
                      'laplace_beltrami_withbc': LaplaceBeltrami_withbc,
                      'laplace_beltrami_withbc2': LaplaceBeltrami_withbc2,
                      'no_boundary_operator': NoBCO,
                      }