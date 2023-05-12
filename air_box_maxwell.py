import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.mesh import Mesh, create_unit_cube, CellType, locate_entities, locate_entities_boundary
from dolfinx.fem import FunctionSpace, Function, Constant, form, locate_dofs_topological, Expression, dirichletbc, petsc, VectorFunctionSpace
from dolfinx.cpp.fem.petsc import (discrete_gradient, interpolation_matrix)
from dolfinx.io import XDMFFile, VTXWriter
from ufl import TrialFunction, TestFunction,  curl, dx, inner, cos, as_vector, pi, SpatialCoordinate, VectorElement 


# Unit cube mesh
n = 50
mesh = create_unit_cube(MPI.COMM_WORLD, n, n, n, cell_type=CellType.tetrahedron)
tdim =  mesh.topology.dim
x = SpatialCoordinate(mesh)


# Define alpha and beta functions
S = FunctionSpace(mesh, ("DG", 0))
beta = Function(S)
beta.x.array[:] = 0
alpha = Constant(mesh, 1.0)

def conductive_marker(x):
    inside = np.zeros(x.shape[1])
    for i in range(tdim):
        inside += np.logical_and(x[i] > 0.25, x[i] < 0.75)
    marked = inside == tdim
    return marked

cells = locate_entities(mesh, tdim, conductive_marker)
beta.x.array[cells] = 1.0

with XDMFFile(MPI.COMM_WORLD, "mesh_out.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(beta)


# Define exact solution and source term
u_e = as_vector((cos(pi * x[1]), cos(pi * x[2]), cos(pi * x[0])))
f = curl(curl(u_e)) + u_e


# Define variational problem
degree = 1
V = FunctionSpace(mesh, ("N1curl", degree))
ndofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

u = TrialFunction(V)
v = TestFunction(V)

a = form(inner(alpha * curl(u), curl(v)) * dx + inner(beta * u, v) * dx)
L = form(inner(f, v) * dx)

u = Function(V)

def boundary_marker(x):
    boundaries = [np.logical_or(np.isclose(x[i], 0.0), np.isclose(x[i], 1.0)) for i in range(3)]
    return np.logical_or(np.logical_or(boundaries[0], boundaries[1]), boundaries[2])

u_bc_ufl = u_e
boundary_facets = locate_entities_boundary(mesh, dim=tdim - 1, marker=boundary_marker)
boundary_dofs = locate_dofs_topological(V, entity_dim=tdim - 1, entities=boundary_facets)
u_bc_expr = Expression(u_bc_ufl, V.element.interpolation_points())

u_bc = Function(V)
u_bc.interpolate(u_bc_expr)
bc = dirichletbc(u_bc, boundary_dofs)

A = petsc.assemble_matrix(a, bcs=[bc])
A.assemble()
b = petsc.assemble_vector(L)
petsc.apply_lifting(b, [a], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                mode=PETSc.ScatterMode.REVERSE)
petsc.set_bc(b, [bc])


# Setup Solver
ksp = PETSc.KSP().create(mesh.comm)
ksp.setOptionsPrefix(f"ksp_{id(ksp)}")
ksp.setOperators(A)

ams_options = {"pc_hypre_ams_cycle_type": 1,
                 "pc_hypre_ams_tol": 1e-8,
                 "solver_atol": 1e-8, "solver_rtol": 1e-8,
                 "solver_initial_guess_nonzero": False,
                 "solver_type": "gmres"}
petsc_options = {}

pc = ksp.getPC()
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts.prefixPush(option_prefix)

for option, value in petsc_options.items():
    opts[option] = value
opts.prefixPop()
pc.setType("hypre")
pc.setHYPREType("ams")

option_prefix = ksp.getOptionsPrefix()
opts.prefixPush(option_prefix)
for option, value in ams_options.items():
    opts[option] = value
opts.prefixPop()

V_CG = FunctionSpace(mesh, ("CG", degree))._cpp_object
G = discrete_gradient(V_CG, V._cpp_object)
G.assemble()
pc.setHYPREDiscreteGradient(G)

W = FunctionSpace(mesh, ("CG", degree))
G = discrete_gradient(W._cpp_object, V._cpp_object)
G.assemble()

qel = VectorElement("CG", mesh.ufl_cell(), degree)
Q3 = FunctionSpace(mesh, qel)
Pi = interpolation_matrix(Q3._cpp_object, V._cpp_object)
Pi.assemble()

ksp.pc.setType("hypre")
ksp.pc.setHYPREType("ams")
ksp.pc.setHYPREDiscreteGradient(G)
ksp.pc.setHYPRESetInterpolations(dim=mesh.geometry.dim, ND_Pi_Full=Pi)


# Set Nodes Interior to Non-Conductive Region
def non_conductive_marker(x):
    inside = np.zeros(x.shape[1], dtype=np.int32)
    for i in range(tdim):
        inside += np.logical_and(x[i] > 0.25, x[i] < 0.75).astype(np.int32)
    marked = inside == tdim
    return ~marked

interior_nodes = np.zeros(ndofs)
cells = locate_entities(mesh, tdim, non_conductive_marker)
dof_ind = locate_dofs_topological(V, entity_dim=tdim, entities=cells)
interior_nodes[dof_ind] = 1.0
interior_nodes = PETSc.Vec().createWithArray(interior_nodes)
pc.setHYPREAMSSetInteriorNodes(interior_nodes)


# Solve
def monitor(ksp, its, rnorm):
    if mesh.comm.rank == 0:
        print(f"Iteration: {its}, rel. residual: {rnorm}")

ksp.setMonitor(monitor)
ksp.setFromOptions()
pc.setUp()
ksp.setUp()
ksp.view()

ksp.solve(b, u.vector)
u.x.scatter_forward()

print(f"Convergence reason: {ksp.getConvergedReason()}")

# from https://github.com/jpdean/maxwell.git :
def save_function(v, filename):
    """Save a function v to file. The function is interpolated into a
    discontinuous Lagrange space so that functions in Nedelec and
    Raviart-Thomas spaces can be visualised exactly"""
    mesh = v.function_space.mesh
    k = v.function_space.ufl_element().degree()
    # NOTE: Alternatively could pass this into function so it doesn't need
    # to be created each time
    W = VectorFunctionSpace(mesh, ("Discontinuous Lagrange", k))
    w = Function(W)
    w.name = v.name
    w.interpolate(v)
    with VTXWriter(mesh.comm, filename, [w]) as file:
        file.write(0.0)

save_function(u, "output.bp")
