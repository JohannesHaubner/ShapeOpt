import pyvista
from pyvista import examples
from colormaps import cmap_2

# specify output directory 
from pathlib import Path
here = Path(__file__).parent.parent.resolve()

import sys, os
sys.path.insert(0, str(here) + '/mesh')

def visualize_shape(fname : str, save_filename : str)

    # specify filename and read mesh    
    filename = str(here) + "/mesh/" + fname
    print(filename)

    reader = pyvista.get_reader(filename)
    mesh = reader.read()

    # plot region of interest

    roi = pyvista.Cube(center=(0.375, 0.2, 0.0), x_length=0.65, y_length=0.3, z_length=0.1)
    extracted = mesh.clip_box(roi, invert=False)

    pl = pyvista.Plotter(window_size=[4000,3000])
    _ = pl.add_mesh(extracted,  cmap=cmap_2, show_edges=True, line_width=1.5, edge_color="white")
    pl.view_xy()
    pl.remove_scalar_bar()
    pl.enable_image_style()
    pl.reset_camera_clipping_range()
    pl.camera.tight()
    pl.save_graphic(save_filename)
    pl.show()

if __name__ == "__main__":
    filename = "mesh_triangles.xdmf"
    save_filename = "init_plot.eps"
    visualize_shape(filename, save_filename )