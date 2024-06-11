[![GNU GPLv3 License](https://img.shields.io/badge/license-GNU_GPLv3-green?style=plastic)](https://choosealicense.com/licenses/gpl-3.0/)
[![Test ShapeOpt](https://github.com/JohannesHaubner/ShapeOpt/actions/workflows/test_shapeopt.yml/badge.svg?style=plastic)](https://github.com/JohannesHaubner/ShapeOpt/actions/workflows/test_shapeopt.yml)

# ShapeOpt

<p align="center">
    <img src="example/FSI/mesh/init_fsi.gif"/>
</p>
<p align="center">
    <img src="example/FSI/mesh/opt_fsi.gif"/>
</p>
<p align="center">
    <img src="example/FSI/mesh3/opt_fsi_interface.gif"/>
</p>

ShapeOpt is based on FEniCS, dolfin-adjoint, IPOPT, cyipopt, pygmsh.

Code repository for the manuscript

>J. Haubner, M. Ulbrich: Advanced Numerical Methods for Shape Optimal Design of Fluid-Structure Interaction Problems. 

based on the implementation for the PhD thesis 

>J. Haubner: Shape Optimization for Fluid-Structure Interaction, Doctoral Dissertation, Technische Universität München, 2020

## Usage/Examples

### Using Docker image
The Dockerfile can be used by running:
```
docker build -t shapeopt .
docker run -it shapeopt
```
or
```
docker pull ghcr.io/johanneshaubner/shapeopt:latest
docker run -ti -v ${PWD}:/root/shared -w /root/shared --entrypoint=/bin/bash --rm ghcr.io/johanneshaubner/shapeopt:latest
```

### IPOPT with HSL
For runs in parallel, IPOPT needs to be installed with HSL. To do so, the Coin-HSL sources archive needs to be added such that it becomes shapeopt/hsl/coinhsl. Afterwards we build the docker image via Dockerfile_hsl.

If not ran from Docker image:
Requires a recent master version of dolfin with MeshView support. Requires the changes propsed in https://bitbucket.org/fenics-project/dolfin/issues/1123/assemble-on-mixed-meshview-forms-returns.

### Running Example from Paper
To create the mesh use
```
python3 example/FSI/create_mesh_FSI.py
```

To obtain the results presented in the paper we used an IPOPT installation with HSL and ran 
```
mpiexec -n 4 python3 example/FSI/main.py >> example/FSI/mesh/Output/terminal.txt
```
(FFC Compiler seems to get stuck in the beginning, cancelling when this happens and restarting the simulation seems to fix this.)

To obtain the tables and figures we ran (with setting the corresponding options to True in the script)
```
mpiexec -n 4 python3 example/FSI/visualize.py
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
can be evaluated. An example for a simple Stokes flow and Fluid-Structure Interaction is given.

### Constraints
Collection of constraint for the optimization problem: volume, barycenter, determinant.

### Mesh_Postprocessing
Postprocessing step for improving the mesh quality after each optimization problem solve.

### Ipopt
Solving the constraint optimization problem is performed using Ipopt.

### Tools
Here, different tools are collect, e.g. initialization of function spaces, save and load objects, etc.

## Acknowledgement
We would like to acknowledge [Jørgen S. Dokken](http://jsdokken.com/), [Henrik N. Finsberg](https://finsberg.github.io/) and [Simon W. Funke](https://github.com/funsim) for the help, support and discussions on reproducibility.

