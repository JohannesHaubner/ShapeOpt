import pyvista
from pyvista import examples
from .colormaps import cmap_2

# specify output directory 
from pathlib import Path
here = Path(__file__).parent.parent.resolve()

import sys, os
sys.path.insert(0, str(here))

def visualize_shape(fname : str, save_fname : str, factor : float = 1.0):

    # specify filename and read mesh    
    filename = str(here) + "/" + fname
    save_filename = str(here) + "/" + save_fname
    print(filename)

    reader = pyvista.get_reader(filename)
    mesh = reader.read()
    print('mesh loaded')

    # plot region of interest
    assert factor > 0
    cx = 0.2 + (0.375-0.2)/(factor)
    roi = pyvista.Cube(center=(cx, 0.2, 0.0), x_length=0.65/factor, y_length=0.2/factor, z_length=0.1/factor)
    extracted = mesh.clip_box(roi, invert=False)

    pyvista.start_xvfb()
    pl = pyvista.Plotter(off_screen=True)
    _ = pl.add_mesh(extracted,  cmap=cmap_2, show_edges=True, line_width=.1, edge_color="white")
    pl.view_xy()
    try:
        pl.remove_scalar_bar()
    except:
        pass
    pl.enable_image_style()
    pl.reset_camera_clipping_range()
    pl.camera.tight()
    pl.screenshot(save_filename)
    print('plot saved')
    #pl.show()

if __name__ == "__main__":
    filename = "mesh_triangles.xdmf"
    save_filename = "init_plot.png"
    visualize_shape(filename, save_filename )