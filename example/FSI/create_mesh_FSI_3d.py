import numpy as np
import pygmsh, meshio
import gmsh
import os

from pathlib import Path
here = Path(__file__).parent.resolve()

if not os.path.exists(str(here) + "/mesh3d"):
    os.makedirs(str(here) + "/mesh3d")

# based on: https://jsdokken.com/src/tutorial_gmsh.html

gmsh.initialize()
gmsh.model.add("3d FSI")

L, H, B, r = 1.5, 0.41, 0.3, 0.05 #2.5, 0.41, 0.41, 0.05
length_flap = 0.4
channel = gmsh.model.occ.addBox(0, 0, 0, L, H, B)
#
flap_width = B/11
left = 5*B/11
center_cylinder = L/2.5*0.5
flap = gmsh.model.occ.addBox(center_cylinder, 0.19, left, length_flap, 0.02, flap_width)
cylinder = gmsh.model.occ.addCylinder(center_cylinder, 0.2, 0, 0, 0, B, r)
fluid, _ = gmsh.model.occ.cut([(3, channel)], [(3, flap), (3, cylinder)])
flap2 = gmsh.model.occ.addBox(center_cylinder, 0.19, left, length_flap, 0.02, flap_width)
cylinder2 = gmsh.model.occ.addCylinder(center_cylinder, 0.2, 0, 0, 0, B, r)
solid, _ = gmsh.model.occ.cut([(3, flap2)], [(3, cylinder2)])
c = [0.5, 0.2, B/2] #Barycenter of cylinder


gmsh.model.occ.synchronize()
volumes = gmsh.model.getEntities(dim=3)
#assert (volumes == fluid[0])
fluid_marker = 13
solid_marker = 15
gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
gmsh.model.addPhysicalGroup(volumes[1][0], [volumes[1][1]], solid_marker)
gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid volume")
gmsh.model.setPhysicalName(volumes[1][0], solid_marker, "Solid volume")

surfaces = gmsh.model.getEntities(dim=2)
inlet_marker, outlet_marker, wall_marker, obstacle_marker, obstacle_solid_marker, interface_marker = 1, 3, 5, 7, 9, 11
walls = []
obstacles = []
obstacle_solid = []
interface = []
fine_resolution_surface_ = []
obstacle_resolution_surface_ = []
for surface in surfaces:
    com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
    if np.isclose(com[0], 0):
        gmsh.model.addPhysicalGroup(surface[0], [surface[1]], inlet_marker)
        inlet = surface[1]
        gmsh.model.setPhysicalName(surface[0], inlet_marker, "Fluid inlet")
    elif np.isclose(com[0], L):
        gmsh.model.addPhysicalGroup(surface[0], [surface[1]], outlet_marker)
        gmsh.model.setPhysicalName(surface[0], outlet_marker, "Fluid outlet")
    elif np.isclose(com[1], 0) or np.isclose(com[2], B) or np.isclose(com[1], H) or np.isclose(com[2], 0):
        walls.append(surface[1])
    elif 0.19 <= com[1] <= 0.21 and center_cylinder < com[0] <= center_cylinder + r and left <= com[2] <= left+flap_width:
        obstacle_solid.append(surface[1])
        fine_resolution_surface_.append(surface[1])
    elif  com[0] <= center_cylinder + r:
        obstacles.append(surface[1])
        obstacle_resolution_surface_.append(surface[1])
    else:
        interface.append(surface[1])
        fine_resolution_surface_.append(surface[1])

gmsh.model.addPhysicalGroup(2, walls, wall_marker)
gmsh.model.setPhysicalName(2, wall_marker, "Walls")
gmsh.model.addPhysicalGroup(2, interface, interface_marker)
gmsh.model.setPhysicalName(2, interface_marker, "Interface")
gmsh.model.addPhysicalGroup(2, obstacles, obstacle_marker)
gmsh.model.setPhysicalName(2, obstacle_marker, "Obstacle")
gmsh.model.addPhysicalGroup(2, obstacle_solid, obstacle_solid_marker)
gmsh.model.setPhysicalName(2, obstacle_solid_marker, "Obstacle_solid")

#from IPython import embed; embed()
boundary_labels = [1, 3, 5, 7, 9, 11] 
bdry_labels = {
          "inflow" : inlet_marker,
          "outflow": outlet_marker,
          "noslip": wall_marker,
          "noslip_obstacle": obstacle_solid_marker,
          "interface": interface_marker, 
          "obstacle_solid": obstacle_solid_marker,
          "obstacle": obstacle_marker,
      }

# dictionary of tags for the subdomains
subdom_labels = {
    "fluid": fluid_marker,
    "solid": solid_marker,
}

# Dictionary with facet-labels from the boundary of each subdomain
subdomain_boundaries = {
    "fluid": ("noslip", "obstacle", "inflow", "outflow", "interface"),
    "solid": ("interface", "obstacle_solid"),
}

params = {"inflow" : inlet_marker,
          "outflow": outlet_marker,
          "noslip": wall_marker,
          "noslip_obstacle": obstacle_solid_marker, #noslipobstacle=obstacle_solid
          "obstacle": obstacle_marker,
          "design": obstacle_marker,
          "interface": interface_marker,
          "mesh_parts": True,
          "fluid": fluid_marker,
          "solid": solid_marker,
          "boundary_labels": boundary_labels,
          "bdry_labels": bdry_labels,
          "subdom_labels": subdom_labels,
          "subdomain_boundaries": subdomain_boundaries,
          "obstacle_solid": obstacle_solid_marker
          }

vol = L * B * H
vol_D_minus_obs = vol - np.pi*r*r * B
geom_prop = {"barycenter_hold_all_domain": [0.5*L, 0.5*B, 0.5*H],
             "volume_hold_all_domain": vol,
             "volume_D_minus_obstacle": vol_D_minus_obs,
             "volume_obstacle": np.pi*r*r*B,
             "length_pipe": L,
             "width_pipe": B,
             "heigth_pipe": H,
             "barycenter_obstacle": c,
             }
np.save(str(here) + '/mesh3d/params.npy', params)
np.save(str(here) + '/mesh3d/geom_prop.npy', geom_prop)


distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "FacesList", fine_resolution_surface_)
#gmsh.model.mesh.field.setNumbers(distance, "FacesList", obstacle_solid)

resolution = r/8
alpha = 1.5
beta = 2.5 #4
threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
gmsh.model.mesh.field.setNumber(threshold, "LcMin", resolution)
gmsh.model.mesh.field.setNumber(threshold, "LcMax", alpha*5*resolution)
gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.5*r)
gmsh.model.mesh.field.setNumber(threshold, "DistMax", r)

inlet_dist = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(inlet_dist, "FacesList", [inlet])
inlet_thre = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(inlet_thre, "IField", inlet_dist)
gmsh.model.mesh.field.setNumber(inlet_thre, "LcMin", alpha*5*resolution)
gmsh.model.mesh.field.setNumber(inlet_thre, "LcMax", alpha*5*resolution)
gmsh.model.mesh.field.setNumber(inlet_thre, "DistMin", 0.1)
gmsh.model.mesh.field.setNumber(inlet_thre, "DistMax", 0.5)

obstacle_dist = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(obstacle_dist, "FacesList", obstacle_resolution_surface_)
obstacle_thre = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(obstacle_thre, "IField", obstacle_dist)
gmsh.model.mesh.field.setNumber(obstacle_thre, "LcMin", beta*resolution)
gmsh.model.mesh.field.setNumber(obstacle_thre, "LcMax", alpha*5*resolution)
gmsh.model.mesh.field.setNumber(obstacle_thre, "DistMin", 0.5*r)
gmsh.model.mesh.field.setNumber(obstacle_thre, "DistMax", 0.5)

minimum = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(
    minimum, "FieldsList", [threshold, inlet_thre, obstacle_thre])
gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)

gmsh.write(str(here) + "/mesh3d/mesh.msh")

import meshio
mesh_from_file = meshio.read(str(here) + "/mesh3d/mesh.msh")

def get_cell_data_(self, name: str, cell_type: str):
        #from IPython import embed; embed()
        return np.concatenate(
            [d for c, d in zip(self.cells, self.cell_data[name]) if c.type == cell_type]
        )

import numpy
def create_mesh(mesh: meshio.Mesh, cell_type: str, data_name: str = "name_to_read",
                prune_z: bool = False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = get_cell_data_(mesh, "gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={
                           data_name: [cell_data]})
    return out_mesh #https://fenicsproject.discourse.group/t/what-is-wrong-with-my-mesh/7504/8

facet_mesh = create_mesh(mesh_from_file, "triangle", prune_z=False)
meshio.write(str(here) + "/mesh3d/facet_mesh.xdmf", facet_mesh)

cell_mesh = create_mesh(mesh_from_file, "tetra", prune_z=False)
meshio.write(str(here) + "/mesh3d/mesh_triangles.xdmf", cell_mesh)