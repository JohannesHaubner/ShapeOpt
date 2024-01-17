"""
Code snippets from Jørgen Riseth and Simon Funke
"""

from typing import Dict, List, Tuple

from dolfin import Mesh, MeshFunction, MeshView, interpolate, cells, Function
import numpy as np
from dolfin.cpp.mesh import MeshFunctionSizet
import mpi4py as MPI
comm = MPI.MPI.COMM_WORLD
id = comm.Get_rank()

def transfer_to_subfunc(f, Vbf):
    # Extract meshes
    V_full = f.function_space()
    f_f = Function(Vbf)

    mesh = V_full.mesh()
    submesh = Vbf.mesh()

    # Build cell mapping between sub and parent meshes
    cell_map = submesh.topology().mapping()[mesh.id()].cell_map()

    # Get cell dofmaps
    dofmap = Vbf.dofmap()
    dofmap_full = V_full.dofmap()


    d_full = []
    d = []
    imin, imax = dofmap_full.ownership_range()
    iminl, imaxl = dofmap.ownership_range()

    # Transfer dofs
    for c in cells(submesh):
        d_full.append([dofmap_full.local_to_global_index(i) for i in dofmap_full.cell_dofs(cell_map[c.index()])])
        d.append([dofmap.local_to_global_index(i) for i in dofmap.cell_dofs(c.index())])  
        #f_f.vector().set_local(dofmap_full.cell_dofs(cell_map[c.index()])) = f.vector()[dofmap.cell_dofs(c.index())]


    d_f_a = np.asarray(d_full).flatten()
    d_a = np.asarray(d).flatten()
    d = np.column_stack((d_f_a, d_a))
    reduced = np.asarray(list(set([tuple(i) for i in d.tolist()])), dtype='int')

    data = reduced

    #len data , data2

    f_vec = f.vector().gather(range(f.vector().size()))
    f_f_vec = f_f.vector().gather(range(f_f.vector().size()))
    f_f_vec[data[:,1]] = f_vec[data[:,0]]


    f_f.vector().set_local(f_f_vec[iminl:imaxl])
    f_f.vector().apply("")
    return f_f


class FacetView(Mesh):
    def __init__(self, boundaries: MeshFunctionSizet, name: str, value: int):
        super().__init__(MeshView.create(boundaries, value))
        self.rename(name, "")
        self.value = value
        
    def mark_facets(self, subdomain: Mesh, subdomainbdry: MeshFunctionSizet):
        """Label a meshfunction defined on subdomain with the current value."""
        self.build_mapping(subdomain)
        facetmap = self.topology().mapping()[subdomain.id()].cell_map()
        for facet in facetmap: 
            subdomainbdry[facet] = self.value
        return facetmap
            

class SubdomainView(Mesh):
    def __init__(self, subdomains: MeshFunctionSizet, name: str, value: int):
        super().__init__(MeshView.create(subdomains, value))
        self.rename(name, "")
        self.value = value
        self.mesh = MeshView.create(subdomains, value)
        self.boundaries = MeshFunction('size_t', self, self.topology().dim()-1, 0)
        
    def mark_boundaries(self, boundarymeshes: List[FacetView]):
        for bdry in boundarymeshes:
            bdry.mark_facets(self, self.boundaries)
        return self.boundaries
            

class SubMeshCollection:
    def __init__(self, subdomains: MeshFunctionSizet, boundaries: MeshFunctionSizet,
                 subdomain_labels: Dict[str, int], boundary_labels: Dict[str, int],
                 subdomain_boundaries: Dict[str, Tuple[str]]):
        self.subdomains = {
            name: SubdomainView(subdomains, name, value) for name, value in subdomain_labels.items()
        }
        self.boundaries = {
            name: FacetView(boundaries, name, value) for name, value in boundary_labels.items()
        }
        self._create_boundary_maps(subdomain_boundaries)
        
    def _create_boundary_maps(self, subdomain_boundaries):
        for subdomain in self.subdomains.values():
            relevant_boundaries = subdomain_boundaries[subdomain.name()]
            subdomain.mark_boundaries([
                self.boundaries[bdry_name] for bdry_name in self.boundaries if bdry_name in relevant_boundaries
            ])
