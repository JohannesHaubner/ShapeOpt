#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 08:39:51 2020

@author: Johannes Haubner
"""

import numpy as np
from ogs5py import OGS
import pygmsh, meshio
import h5py



# geometric properties
L = 2.5            # length of channel
H = 0.4            # heigth of channel
c_x, c_y = L/2, H/2 # position of object
r_x = 0.05 # radius of object

# labels
inflow = 1
outflow = 2
walls = 3
obstacle = 4

params = {"inflow" : inflow, 
          "outflow": outflow,
          "noslip": walls,
          "design": obstacle
          }
np.save('../Mesh_Generation/params.npy', params)

geom = pygmsh.geo.Geometry()

circle = geom.add_circle(
    x0=[0.2, 0.2, 0.0], radius=0.05, mesh_size=0.01, num_sections=4, make_surface=False
)

rectangle = geom.add_rectangle(0.0, L, 0.0, H, 0.0, mesh_size=0.005, holes=[circle])

flow_list = [rectangle.line_loop.lines[3]]
walls_list = [rectangle.line_loop.lines[0], rectangle.line_loop.lines[2]] 
geom.add_physical(rectangle.surface,label=12)
geom.add_physical(flow_list, label=inflow)
geom.add_physical(walls_list, label=walls)
geom.add_physical([rectangle.line_loop.lines[1]], label=outflow)
geom.add_physical(circle.line_loop.lines, label=obstacle)

mesh = pygmsh.generate_mesh(geom, prune_z_0=True, geo_filename="../Output/Mesh_Generation/mesh.geo")

#meshio.write("mesh.xdmf", mesh)
mesh_triangle = meshio.Mesh(points=mesh.points, cells={"triangle": mesh.get_cells_type("triangle")})
meshio.write("../Output/Mesh_Generation/mesh_triangles. xdmf", mesh_triangle)
mesh_boundary = meshio.Mesh(points=mesh.points, 
                                               cells={"line": mesh.get_cells_type("line")}, 
                                               cell_data={"name_to_read": [mesh.get_cell_data("gmsh:physical", "line")]})
meshio.write("../Output/Mesh_Generation/mesh_boundary.xdmf", mesh_boundary)


#model = OGS()
#model.msh.generate("gmsh", geo_object=geom)
#model.msh.show()
