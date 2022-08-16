[![GNU GPLv3 License](https://img.shields.io/badge/license-GNU_GPLv3-green?style=plastic)](https://choosealicense.com/licenses/gpl-3.0/)

# ShapeOpt

ShapeOpt is based on FEniCS, dolfin-adjoint, IPOPT, cyipopt, pygmsh.

Code repository for the manuscript

>J. Haubner, M. Ulbrich: Advanced Numerical Methods for Shape Optimal Design of Fluid-Structure Interaction Problems. 


## Usage/Examples

Requires a recent master version of dolfin with MeshView support. It might require the changes propsed in https://bitbucket.org/fenics-project/dolfin/issues/1123/assemble-on-mixed-meshview-forms-returns.
Moreover, it also requires a dolfin-adjoint version which supports the changes of git+https://github.com/dolfin-adjoint/pyadjoint.git@constant-adjfloat.

The Dockerfile (preliminary version) can be used by running:
```
docker build -t shapeopt .
docker run -it shapeopt
```


## Running Tests

To run tests, run the following command

```bash
pytest
```
## License

[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)

## Structure

### Control_to_Trafo
We apply a method of mappings approach and solve everything on a reference domain. In order not to have too much 
redundancy in the representation of the transformations, we choose a scalar valued function on the design boundary 
as control for the optimization problem and define the deformation field using an extension operator from the design 
boundary to the whole domain. In the example code, we define the Control_to_Trafo-mapping as a composition of boundary
operators and extension operators.

### Reduced_Objective
After having defined the transformation, the reduced objective and its gradient (computed using dolfin-adjoint) 
can be evaluated. An example for a simple Stokes flow is given.

### Constraints
Collection of constraint for the optimization problem: volume, barycenter, determinant.

### Ipopt
Solving the constraint optimization problem is performed using Ipopt.

### Tools
Here, different tools are collect, e.g. initialization of function spaces, save and load objects, etc.
