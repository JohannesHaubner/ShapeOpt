import pyvista
from colormaps import cmap_1, cmap_2

# specify output directory 
from pathlib import Path
here = Path(__file__).parent.parent.resolve()

import sys, os
sys.path.insert(0, str(here))

# specify filename and read mesh    
filename = str(here) + "/Output/Forward/init/velocity2.pvd"
print(filename)

reader = pyvista.get_reader(filename)

reader.time_values

mesh = reader.read()
mesh.plot()