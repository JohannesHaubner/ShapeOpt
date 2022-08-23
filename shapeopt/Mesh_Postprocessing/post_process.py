from dolfin import *

def harmonic(defo):
    V = defo.function_space()
    mesh = defo.function_space().mesh()

    v = TestFunction(V)
    u = TrialFunction(V)

    a = inner(grad(u),grad(v))*dx(mesh)
    L = inner(defo, v)*dx(mesh)

    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, Constant(("0.0","0.0")), boundary)

    u = Function(V)

    solve(a == L, u, bc)

    return u

def biharmonic(defo):
    V = defo.function_space()
    mesh = defo.function_space().mesh()

    breakpoint()
