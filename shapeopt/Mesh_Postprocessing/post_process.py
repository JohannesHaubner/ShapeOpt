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

    # mixed approach for biharmonic extension
    VV = FunctionSpace(mesh, MixedElement(V.ufl_element(), V.ufl_element()))

    uw = TrialFunction(VV)
    psi = TestFunction(VV)
    (u, w) = split(uw)
    (psiu, psiw) = split(psi)

    a = (inner(w, psiw) - inner(grad(u), grad(psiw)) - inner(grad(w), grad(psiu)))*dx
    L = Constant(0.0)*(psiw[0] + psiu[0])*dx

    bc = DirichletBC(VV.sub(0), defo, "on_boundary")

    uw = Function(VV)

    solve(a == L, uw, bc)

    u = project(uw.sub(0), V)
    u.vector().axpy(-1.0, defo.vector())

    return u

