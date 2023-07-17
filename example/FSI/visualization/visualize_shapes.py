import pyvista
from pyvista import examples

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# specify output directory 
from pathlib import Path
here = Path(__file__).parent.parent.resolve()

import sys, os
sys.path.insert(0, str(here) + '/mesh')

# specify filename and read mesh    
filename = str(here) + "/mesh/mesh_triangles.xdmf"
print(filename)

reader = pyvista.get_reader(filename)
mesh = reader.read()

# specify colormaps
# colormap 1
cdict = {
    'red': (
        (0.0, 0.0, 0.0),
        (0.5, 0.85490196100000004, 0.85490196100000004),
        (1.0, 0.890196078, 0.890196078),
    ),
    'green': (
        (0.0, 0.39607843100000001, 0.39607843100000001),
        (0.5, 0.84313725500000003, 0.84313725500000003),
        (1.0,  0.44705882400000002, 0.44705882400000002),
    ),
    'blue': (
        (0.0, 0.74117647099999995, 0.74117647099999995),
        (0.5, 0.79607843099999998, 0.79607843099999998),
        (1.0, 0.133333333, 0.133333333),
    )
}
cmap_1 = LinearSegmentedColormap('map1', cdict)

# colormap 2
cdict = {
    'red': (
        (0.0, 0.85490196100000004, 0.85490196100000004),
        (1.0, 0.0, 0.0),
    ),
    'green': (
        (0.0, 0.84313725500000003, 0.84313725500000003,),
        (1.0,  0.20000000000000001, 0.20000000000000001),
    ),
    'blue': (
        (0.0, 0.79607843099999998, 0.79607843099999998),
        (1.0, 0.34901960784000002, 0.34901960784000002),
    )
}
cmap_2 = LinearSegmentedColormap('map1', cdict)

# plot region of interest

roi = pyvista.Cube(center=(0.375, 0.2, 0.0), x_length=0.65, y_length=0.3, z_length=0.1)
extracted = mesh.clip_box(roi, invert=False)

pl = pyvista.Plotter(window_size=[4000,3000])
_ = pl.add_mesh(extracted,  cmap=cmap_2, show_edges=True, line_width=3, edge_color="white")
pl.view_xy()
pl.remove_scalar_bar()
pl.enable_image_style()
pl.reset_camera_clipping_range()
pl.camera.tight()
pl.save_graphic("init_plot.eps")
pl.show()

