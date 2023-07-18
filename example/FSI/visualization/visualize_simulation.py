import pyvista
import numpy as np
from colormaps import cmap_1, cmap_2

# specify filename and read mesh    
filename = "../Output/Forward/Init/velocity2.pvd"
filename2 = "../Output/Forward/Init/char_sol.pvd"
print(filename)

reader = pyvista.get_reader(filename)
reader2 = pyvista.get_reader(filename2)

# filename for mp4
mp4_fn = "init_fsi.mp4"

# plotter
pl = pyvista.Plotter()
# open a movie file
pl.open_movie(mp4_fn)

# test
reader.set_active_time_point(2)
print('active time value:', reader.active_time_value)
mesh = reader.read()[0]
mag = np.sum(np.abs(mesh["velocity"])**2, axis=-1)**(1./2)
mesh.point_data.set_scalars(mag, name="magnitude")
mesh.point_data.remove("velocity")
print(mesh.active_scalars)
print(mesh.array_names)
print(reader.active_datasets)

for i in range(len(reader.time_values)):
    reader.set_active_time_point(i)
    reader2.set_active_time_point(i)
    print('active time value:', reader.active_time_value)
    mesh = reader.read()[0]
    mesh2 = reader2.read()[0]
    mag = np.sum(np.abs(mesh["velocity"])**2, axis=-1)**(1./2)
    mesh.point_data.set_scalars(mag, name="magnitude")
    mesh.point_data.remove("velocity")
    print(reader.active_datasets)
    pl.view_xy()
    pl.add_mesh(mesh, cmap=cmap_1, clim=[0., 2.5])
    pl.add_mesh(mesh2, cmap=cmap_2, clim=[7., 8.], opacity="linear")
    pl.enable_image_style()
    pl.reset_camera_clipping_range()
    pl.remove_scalar_bar("magnitude")
    pl.remove_scalar_bar()
    pl.camera.tight()
    pl.write_frame()
    
pl.close()