from .LaplaceBeltrami import LaplaceBeltrami
from .LaplaceBeltrami_withbc import LaplaceBeltrami_withbc

boundary_operators = {'laplace_beltrami': LaplaceBeltrami,
                      'laplace_beltrami_withbc': LaplaceBeltrami_withbc,
                      }