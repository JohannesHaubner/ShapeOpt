import pyvista
import numpy as np
from .colormaps import cmap_1, cmap_2

# specify output directory 
from pathlib import Path
here = Path(__file__).parent.parent.resolve()

def make_movie(foldername : str, gifname : str):

    # specify filename and read mesh    
    filename = str(here) + "/Output/Forward/" + foldername + "/velocity2.pvd"
    filename2 = str(here) + "/Output/Forward/" + foldername + "/char_sol.pvd"
    print(filename)

    reader = pyvista.get_reader(filename)
    reader2 = pyvista.get_reader(filename2)

    # filename for mp4
    gif_fn = gifname

    import os
    os.system("export RDMAV_FORK_SAFE=0")

    # plotter
    pyvista.start_xvfb()
    pl = pyvista.Plotter(window_size=([1024, 768]),off_screen=True)
    # open a movie file

    n = 2

    pl.open_gif(gif_fn, fps=50/n)

    # test
    reader.set_active_time_point(0)
    print('active time value:', reader.active_time_value)
    mesh = reader.read()[0]
    mesh2 = reader2.read()[0]
    mag = np.sum(np.abs(mesh["velocity"])**2, axis=-1)**(1./2)
    mesh.point_data.set_scalars(mag, name="magnitude")
    mesh.point_data.remove("velocity")
    print(reader.active_datasets)
    pl.set_background('white')
    pl.view_xy()
    pl.add_mesh(mesh, cmap=cmap_1, clim=[0., 2.5], lighting=False, smooth_shading=True)
    pl.add_mesh(mesh2, cmap=cmap_2, clim=[0., 1.], opacity="linear", lighting=False, smooth_shading=True)
    pl.enable_image_style()
    pl.reset_camera_clipping_range()
    pl.remove_scalar_bar("magnitude")
    pl.remove_scalar_bar()
    pl.camera.tight(-0.005)
    print('setting set')
    # Run through each frame
    pl.show(auto_close=False)
    pl.write_frame()  # write initial data

    for i in range(int(len(reader.time_values)/n)):
        pl.clear()
        reader.set_active_time_point(i*n)
        reader2.set_active_time_point(i*n)
        print('active time value:', reader.active_time_value)
        mesh = reader.read()[0]
        mesh2 = reader2.read()[0]
        mag = np.sum(np.abs(mesh["velocity"])**2, axis=-1)**(1./2)
        mesh.point_data.set_scalars(mag, name="magnitude")
        mesh.point_data.remove("velocity")
        print(reader.active_datasets)
        pl.set_background('white')
        pl.view_xy()
        pl.add_mesh(mesh, cmap=cmap_1, clim=[0., 2.5], lighting=False, smooth_shading=True)
        pl.add_mesh(mesh2, cmap=cmap_2, clim=[0., 1.], opacity="linear", lighting=False, smooth_shading=True)
        pl.enable_image_style()
        pl.reset_camera_clipping_range()
        pl.remove_scalar_bar("magnitude")
        pl.remove_scalar_bar()
        pl.camera.tight(-0.005)
        print('setting set')
        # Run through each frame
        pl.update()
        pl.show(auto_close=False)
        pl.write_frame()  # write initial data
    
    pl.close()
    pass

if __name__ == "__main__":
    foldername = "Init"
    outname = "init_fsi.mp4"
    make_movie(foldername, outname)