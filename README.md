# ShapeOpt

ShapeOpt is based on FEniCS, dolfin-adjoint and IPOPT.

```
mkdir Output 
cd Output
mkdir MeshGeneration
mkdir ReducedObjective
mkdir Tests
cd Tests
mkdir ForwardEquation

python3 ShapeOpt/MeshGeneraton/create_mesh_.py
python3 main.py
```


## Structure

### Mesh_Generation
create_mesh_.py provides an example for generating a mesh with labels for the different boundary parts using pygmsh 

### Control_to_Trafo
We apply a method of mappings approach and solve everything on a reference domain. In order not to have too much redundancy in the representation of the transformations, we choose a scalar valued function on the design boundary as control for the optimization problem and define the deformation field using an extension operator from the design boundary to the whole domain. This can be performed in several ways, one example is given here.

### Reduced_Objective
After having defined the transformation, the reduced objective and its gradient (performed with dolfin-adjoint) can be evaluated. An example for a simple Stokes flow is given.

### Constraints
Collection of constraint for the optimization problem: volume, barycenter, determinant.

### Ipopt
Solving the constraint optimization problem is performed using Ipopt.

### Tools
Here, different tools are collect, e.g. initialization of function spaces, save and load objects, etc.
