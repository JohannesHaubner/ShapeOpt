from dolfin import *
#import gmsh
import pygmsh
#import sys
#import meshio
#import itertools
import numpy as np
import meshio
#import numpy
import gmsh
import os
import mpi4py


from pathlib import Path
here = Path(__file__).parent.resolve()
import sys, os
sys.path.insert(0, str(here.parent.parent) + '/src')
from shapeopt.Tools.subdomains import SubMeshCollection

# restricted to FSI example

def remesh(mesh_ending_read: str, mesh_ending_write: str, pathmesh: str):
    """
    :param mesh_ending_read: reads in files in mr_mesh, mr_dom and mr_bound (see below)
    :param mesh_ending_write: writes remeshed facet and boundary mesh in files defined in mw_facet, mw_boundary (see below)
    :param pathmesh: path where mesh is stored
    """

    mr = mesh_ending_read
    mw = mesh_ending_write

    mr_mesh = pathmesh + '/mesh_triangles' + mr + '.xdmf'
    mr_bound = pathmesh + '/facet_mesh' + mr + '.xdmf'

    mw_facet = pathmesh + '/facet_mesh' + mw + '.xdmf'
    mw_boundary = pathmesh + '/mesh_triangles' + mw + '.xdmf'

    if mpi4py.MPI.COMM_WORLD.rank == 0:

        # load mesh
        mesh = Mesh(MPI.comm_self)
        with XDMFFile(MPI.comm_self, mr_mesh) as infile:
            infile.read(mesh)

        # load markers
        mvc = MeshValueCollection('size_t', mesh, 2)
        with XDMFFile(MPI.comm_self, mr_mesh) as infile:
            infile.read(mvc)
        domains = MeshFunction('size_t', mesh, mvc)

        mvc = MeshValueCollection('size_t', mesh, 1)
        with XDMFFile(MPI.comm_self, mr_bound) as infile:
            infile.read(mvc)
        boundaries = MeshFunction('size_t', mesh, mvc)


        # boundary labels
        boundary_labels = [1, 2, 3, 4, 5, 6] # has to contain all labels for boundary parts
        inflow = 1
        outflow = 2
        walls = 3
        noslipobstacle = 4
        obstacle = 5
        interface = 6
        fluid = 7
        solid = 8

        fluid_bl = [1, 2, 3, 5, 6]
        solid_bl = [4, 6]

        #
        boundary_labels = {
            "inflow": inflow,
            "outflow": outflow,
            "walls": walls,
            "obstacle_fluid": obstacle,
            "obstacle_solid": noslipobstacle,
            "interface": interface,
        }

        resolution = {
            "inflow": 0.1,
            "outflow": 0.1,
            "walls": 0.1,
            "obstacle_fluid": 0.005,
            "obstacle_solid": 0.005,
            "interface": 0.005,
        }

        subdomain_labels = {
            "fluid": fluid,
            "solid": solid,
        }

        subdomain_boundaries = {
            "fluid": ("inflow", "outflow", "walls", "obstacle_fluid", "interface"),
            "solid": ("interface", "obstacle_solid"),
        }

        # mesh view create
        meshes = SubMeshCollection(domains, boundaries, subdomain_labels, boundary_labels, subdomain_boundaries)

        # resolution
        resolution = 0.05  #0.05 #1 # 0.005 #0.1

        # geometric properties
        L = 2.5 #2.5 #20            # length of channel
        H = 0.41 #0.4 #6           # heigth of channel
        c = [0.2, 0.2, 0]  #[0.2, 0.2, 0] #[10, 3, 0]  # position of object
        r = 0.05

        # Initialize empty geometry using the build in kernel in GMSH
        geometry = pygmsh.geo.Geometry()
        # Fetch model we would like to add data to
        model = geometry.__enter__()
        # Add circle
        sin = 0.5 # sin(30°)
        cos = np.sqrt(3)/2 # cos(30°)
        pc0 = model.add_point(c)
        pc2 = model.add_point((0.24898979485, 0.21, 0), mesh_size=0.1*resolution)
        pc3 = model.add_point((0.24898979485, 0.19,0), mesh_size=0.1*resolution)
        circle3 = model.add_circle_arc(pc2, pc0, pc3)


        # Add design interface
        mb = meshes.boundaries["obstacle_fluid"]
        coords = mb.coordinates()
        cells = mb.cells()
        # find first coordinate
        coords_row = np.asarray(coords[:, 0] == 0.24898979485) * np.asarray(coords[:, 1] == 0.19)
        id = np.where(coords_row==True)[0][0]
        pco = pc3
        lines = []
        for i in range(len(cells)):
            cell_id = np.where([cells==id][0][:, 0] + [cells==id][0][:, 1] == True)[0][0]
            id = (set(cells[cell_id]) - set([id])).pop()
            if len(cells) > 1:
                pcn = model.add_point(coords[id])
            else:
                pcn = pc2
            cells = np.delete(cells, (cell_id), axis=0)
            lines.append(model.add_line(pco, pcn))
            pco = pcn

        # Add elastic flag
        pf1 = model.add_point((0.6, 0.21, 0), mesh_size=0.1*resolution)
        pf2 = model.add_point((0.6, 0.19, 0), mesh_size=0.1*resolution)
        fl1 = model.add_line(pf2, pc3)
        fl2 = model.add_line(pf1, pf2)
        fl3 = model.add_line(pc2, pf1)

        # obstacle
        obstacle = model.add_curve_loop(lines + [fl3, fl2, fl1])
        flag = model.add_curve_loop([fl3, fl2, fl1, -circle3])

        # Add points with finer resolution on left side
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
            channel_loop, holes=[obstacle])
        plane_surface2 = model.add_plane_surface(
            flag)

        # Call gmsh kernel before add physical entities
        model.synchronize()

        volume_marker = 6
        model.add_physical([channel_lines[0]], "inflow") # mark inflow boundary with 1
        model.add_physical([channel_lines[2]], "outflow") # mark outflow boundary with 2
        model.add_physical([channel_lines[1], channel_lines[3]], "walls") # mark walls with 3
        model.add_physical([circle3], "noslip_obstacle")
        model.add_physical(lines, "obstacle") # mark obstacle with 4
        model.add_physical([fl1, fl2, fl3], "interface") # mark interface with 5
        model.add_physical([plane_surface], "fluid") # mark fluid domain with 6
        model.add_physical([plane_surface2], "solid") # mark solid domain with 7

        geometry.generate_mesh(dim=2)

        print('geometry generated')

        gmsh.write( pathmesh + "/mesh.msh")
        gmsh.clear()
        geometry.__exit__()

        print('write complete', flush=True)

        mesh_from_file = meshio.read(pathmesh + "/mesh.msh")

        def create_mesh(mesh, cell_type, prune_z=False):
            cells = mesh.get_cells_type(cell_type)
            cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
            points = mesh.points[:, :2] if prune_z else mesh.points
            out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
            return out_mesh

        facet_mesh_path = mw_facet
        triangle_mesh_path = mw_boundary

        for i in [facet_mesh_path, triangle_mesh_path]:
            if os.path.isfile(i):
                os.remove(i)
                os.remove(i[:-4] + "h5")

        line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
        meshio.write(facet_mesh_path, line_mesh)

        triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
        meshio.write(triangle_mesh_path, triangle_mesh)

    print('remeshing complete')
    pass