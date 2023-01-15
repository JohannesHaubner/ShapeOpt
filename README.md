[![GNU GPLv3 License](https://img.shields.io/badge/license-GNU_GPLv3-green?style=plastic)](https://choosealicense.com/licenses/gpl-3.0/)
[![Test ShapeOpt](https://github.com/JohannesHaubner/ShapeOpt/actions/workflows/test_shapeopt.yml/badge.svg?style=plastic)](https://github.com/JohannesHaubner/ShapeOpt/actions/workflows/test_shapeopt.yml)

# ShapeOpt

ShapeOpt is based on FEniCS, dolfin-adjoint, IPOPT, cyipopt, pygmsh.

Code repository for the manuscript

>J. Haubner, M. Ulbrich: Advanced Numerical Methods for Shape Optimal Design of Fluid-Structure Interaction Problems. 

and the PhD thesis

>J. Haubner: Shape Optimization for Fluid-Structure Interaction, Doctoral Dissertation, Technische Universität München, 2020

## Usage/Examples

Requires a recent master version of dolfin with MeshView support. It might require the changes propsed in https://bitbucket.org/fenics-project/dolfin/issues/1123/assemble-on-mixed-meshview-forms-returns.

The Dockerfile (preliminary version) can be used by running:
```
docker build -t shapeopt .
docker run -it shapeopt
```
or
```
docker pull ghcr.io/johanneshaubner/shapeopt:latest
docker run -it ghcr.io/johanneshaubner/shapeopt:latest
```

### Running IPOPT with HSL

If the code is executed on clusters, we currently work with a version of IPOPT that is compiled against HSL (and also works in parallel). To do so, the Coin-HSL sources archive needs to be added such that it becomes shapeopt/hsl/coinhsl. Afterwards we build the docker image via Dockerfile_hsl and save it to a tar.gz file via:
```
docker save myimage:latest | gzip > shapeopt_hsl.tar.gz
``` 
Then we copy the shapeopt_hsl.tar.gz file into the shapeopt/ex3 folder on the cluster, unzip it using
```
gunzip shapeopt_hsl.tar.gz
```
and run the code using
```
sbatch run.sh
```
If shapeopt/ex3/shapeopt.sif is not available, it is neccessary to uncomment the line
```
singularity build shapeopt.sif docker-arxiv://shapopt_hsl.tar
```
in the run.sh script.

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

### Mesh_Postprocessing
Postprocessing step for improving the mesh quality after each optimization problem solve.

### Ipopt
Solving the constraint optimization problem is performed using Ipopt.

### Tools
Here, different tools are collect, e.g. initialization of function spaces, save and load objects, etc.

## Acknowledgement
We would like to acknowledge [Jørgen Dokken](http://jsdokken.com/) and [Henrik Finsberg](https://finsberg.github.io/) for the help, support and discussions on reproducibility.

