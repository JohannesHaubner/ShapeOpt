import pyvista
import numpy as np
from colormaps import cmap_1, cmap_2

# specify filename and read mesh    
filename = "../Output/Forward/init/velocity2.pvd"
print(filename)

reader = pyvista.get_reader(filename)

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
for i in range(2): #range(len(reader.time_values)):
    reader.set_active_time_point(i)
    print('active time value:', reader.active_time_value)
    mesh = reader.read()[0]
    mag = np.sum(np.abs(mesh["velocity"])**2, axis=-1)**(1./2)
    mesh.point_data.set_scalars(mag, name="magnitude")
    mesh.point_data.remove("velocity")
    print(reader.active_datasets)
    pl.view_xy()
    pl.add_mesh(mesh, cmap=cmap_1)
    pl.enable_image_style()
    pl.reset_camera_clipping_range()
    pl.camera.tight()
    pl.write_frame()
    
pl.close()