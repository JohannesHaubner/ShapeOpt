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
