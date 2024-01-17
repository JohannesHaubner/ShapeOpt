import numpy as np
from ogs5py import OGS
import pygmsh, meshio
import h5py
import os

from pathlib import Path
here = Path(__file__).parent.resolve()

if not os.path.exists(str(here) + "/mesh2"):
    os.makedirs(str(here) + "/mesh2")

# resolution
resolution = 0.025  #0.05 #1 # 0.005 #0.1
alp = 0.2

# geometric properties
L = 2.5 #2.5 #20            # length of channel
H = 0.41 #0.4 #6           # heigth of channel
c = [0.5, 0.2, 0]  #[0.2, 0.2, 0] #[10, 3, 0]  # position of object
r = 0.05 #0.05 #0.5 # radius of object

# labels
boundary_labels = [1, 2, 3, 4, 5] # has to contain all labels for boundary parts
inflow = 1
outflow = 2
walls = 3
obstacle = 4
interface = 5
fluid = 7
solid = 8
noslipobstacle = 6

params = {"inflow" : inflow,
          "outflow": outflow,
          "noslip": walls,
          "noslip_obstacle": noslipobstacle,
          "design": interface,
          "interface": interface,
          "mesh_parts": True,
          "fluid": fluid,
          "solid": solid,
          "boundary_labels": boundary_labels
          }

vol = L*H
vol_D_minus_obs = vol - np.pi*r*r
geom_prop = {"barycenter_hold_all_domain": [0.5*L, 0.5*H],
             "volume_hold_all_domain": vol,
             "volume_D_minus_obstacle": vol_D_minus_obs,
             "volume_obstacle": np.pi*r*r,
             "length_pipe": L,
             "heigth_pipe": H,
             "barycenter_obstacle": [ c[0], c[1]],
             }
np.save(str(here) + '/mesh2/params.npy', params)
np.save(str(here) + '/mesh2/geom_prop.npy', geom_prop)

# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.geo.Geometry()
# Fetch model we would like to add data to
model = geometry.__enter__()
# Add circle
pc = model.add_point(c)
sin = 0.5 # sin(30°)
cos = np.sqrt(3)/2 # cos(30°)
pc0 = model.add_point(c)
obstacle = model.add_circle(c, r, mesh_size=alp*resolution)
interface = model.add_circle(c, 2*r, mesh_size=alp*resolution)

# Add outer points 
points = [model.add_point((0, 0, 0), mesh_size=resolution),
          model.add_point((L, 0, 0), mesh_size=resolution), #5*resolution
          model.add_point((L, H, 0), mesh_size=resolution), #5*resolution
          model.add_point((0, H, 0), mesh_size=resolution)]

# Add lines between all points creating the rectangle
channel_lines = [model.add_line(points[i], points[i+1])
                 for i in range(-1, len(points)-1)]

# Create a line loop and plane surface for meshing
channel_loop = model.add_curve_loop(channel_lines)
plane_surface = model.add_plane_surface(
    channel_loop, holes=[interface.curve_loop])
plane_surface2 = model.add_plane_surface(
    interface.curve_loop, holes=[obstacle.curve_loop])

# Call gmsh kernel before add physical entities
model.synchronize()

volume_marker = 6
model.add_physical([channel_lines[0]], "inflow") # mark inflow boundary with 1
model.add_physical([channel_lines[2]], "outflow") # mark outflow boundary with 2
model.add_physical([channel_lines[1], channel_lines[3]], "walls") # mark walls with 3
model.add_physical([obstacle.curve_loop], "obstacle") # mark obstacle 
model.add_physical([interface.curve_loop], "interface") # mark interface 
model.add_physical([plane_surface], "fluid") # mark solid domain 
model.add_physical([plane_surface2], "solid") # mark solid domain 

geometry.generate_mesh(dim=2)
import gmsh
gmsh.write(str(here) + "/mesh2/mesh.msh")
gmsh.clear()
geometry.__exit__()

import meshio
mesh_from_file = meshio.read(str(here) + "/mesh2/mesh.msh")

import numpy
def create_mesh(mesh: meshio.Mesh, cell_type: str, data_name: str = "name_to_read",
                prune_z: bool = False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={
                           data_name: [cell_data]})
    return out_mesh #https://fenicsproject.discourse.group/t/what-is-wrong-with-my-mesh/7504/8

line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
meshio.write(str(here) + "/mesh2/facet_mesh.xdmf", line_mesh)

triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
meshio.write(str(here) + "/mesh2/mesh_triangles.xdmf", triangle_mesh)

#mesh = line_mesh
#mesh_boundary = meshio.Mesh(points=mesh.points,
#                                               cells={"line": mesh.get_cells_type("line")},
#                                               cell_data={"name_to_read": [mesh.get_cell_data("gmsh:physical", "line")]})
#meshio.write("../Output/Mesh_Generation/mesh_boundary.xdmf", mesh_boundary)


#model = OGS()
#model.msh.generate("gmsh", geo_object=geom)
#model.msh.show()