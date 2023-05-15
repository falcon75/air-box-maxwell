import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.mesh import Mesh, create_unit_cube, CellType, locate_entities, locate_entities_boundary
from dolfinx.fem import FunctionSpace, Function, Constant, form, locate_dofs_topological, Expression, dirichletbc, petsc, VectorFunctionSpace
from dolfinx.cpp.fem.petsc import (discrete_gradient, interpolation_matrix)
from dolfinx.io import XDMFFile, VTXWriter
from ufl import TrialFunction, TestFunction,  curl, dx, inner, cos, as_vector, pi, SpatialCoordinate, VectorElement


# save_function and inspiration from https://github.com/jpdean/maxwell.git :
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


# Unit cube mesh
n = 10
mesh = create_unit_cube(MPI.COMM_WORLD, n, n, n,
                        cell_type=CellType.hexahedron)
tdim = mesh.topology.dim
x = SpatialCoordinate(mesh)

# Define alpha and beta functions
S = FunctionSpace(mesh, ("DG", 0))
beta = Function(S)
beta.x.array[:] = 0  # set beta in non conductive region to 0
alpha = Constant(mesh, 1.0)


def conductive_marker(x):
    inside = np.zeros(x.shape[1])
    for i in range(tdim):
        inside += np.logical_and(x[i] >= 0.3, x[i] <= 0.7)
    marked = inside == tdim
    return marked


conductive_cells = locate_entities(mesh, tdim, conductive_marker)
beta.x.array[conductive_cells] = 1.0  # set beta in conductive region to 1

with XDMFFile(MPI.COMM_WORLD, "mesh_out.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(beta)

# Define boundary condition (this is the exact solution from the constant beta cube)
u_e = as_vector((cos(pi * x[1]), cos(pi * x[2]), cos(pi * x[0])))
f = curl(curl(u_e)) + u_e

# Formulation
degree = 1
V = FunctionSpace(mesh, ("N1E", degree))
ndofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

u = TrialFunction(V)
v = TestFunction(V)

a = form(inner(alpha * curl(u), curl(v)) * dx + inner(beta * u, v) * dx)
a_beta = form(inner(beta * curl(u), curl(v)) * dx)

L = form(inner(f, v) * dx)

u = Function(V)


def boundary_marker(x):
    boundaries = [np.logical_or(np.isclose(
        x[i], 0.0), np.isclose(x[i], 1.0)) for i in range(3)]
    return np.logical_or(np.logical_or(boundaries[0], boundaries[1]), boundaries[2])


u_bc_ufl = u_e
boundary_facets = locate_entities_boundary(
    mesh, dim=tdim - 1, marker=boundary_marker)
boundary_dofs = locate_dofs_topological(
    V, entity_dim=tdim - 1, entities=boundary_facets)
u_bc_expr = Expression(u_bc_ufl, V.element.interpolation_points())

u_bc = Function(V)
u_bc.interpolate(u_bc_expr)
bc = dirichletbc(u_bc, boundary_dofs)

A = petsc.assemble_matrix(a, bcs=[bc])
A.assemble()

Abeta = petsc.assemble_matrix(a_beta, bcs=[bc])
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

ams_options = {"pc_type": "hypre",
               "pc_hypre_type": "ams",
               "pc_hypre_ams_cycle_type": 3,
               "pc_hypre_ams_tol": 1e-3,
               "pc_hypre_ams_relax_times": 1,
               "pc_hypre_ams_relax_type": 2,
               "pc_hypre_ams_projection_frequency": 1,
               "pc_hypre_boomeramg_strong_threshold": 0.7,
               "pc_hypre_boomeramg_agg_num_paths": 2,
               "pc_hypre_boomeramg_agg_nl": 4,
               "pc_hypre_boomeramg_relax_type_all": False,
               }

petsc_options = {"ksp_type": "gmres",
                 "ksp_atol": 1e-8, "ksp_rtol": 1e-8,
                 "ksp_residual_type": "unpreconditioned",
                #  "ksp_gmres_restart": 100,
                #  "ksp_gmres_haptol": 1e-5,
                 "ksp_initial_guess_nonzero": False, }

opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts.prefixPush(option_prefix)

for option, value in petsc_options.items():
    opts[option] = value
opts.prefixPop()

option_prefix = ksp.getOptionsPrefix()
opts.prefixPush(option_prefix)
for option, value in ams_options.items():
    opts[option] = value
opts.prefixPop()

W = FunctionSpace(mesh, ("CG", degree))
G = discrete_gradient(W._cpp_object, V._cpp_object)
G.assemble()

X = VectorElement("CG", mesh.ufl_cell(), degree)
Q = FunctionSpace(mesh, X)
Pi = interpolation_matrix(Q._cpp_object, V._cpp_object)
Pi.assemble()


i_nodes = Function(V)
dofs = locate_dofs_topological(V, entity_dim=tdim, entities=conductive_cells)
i_nodes.x.array[:] = 1.
i_nodes.x.array[dofs] = 0.
i_nodes.x.array[boundary_dofs] = 0.

def monitor(ksp, its, rnorm):
    if mesh.comm.rank == 0:
        print(f"Iteration: {its}, rel. residual: {rnorm}")

ksp.setMonitor(monitor)
ksp.setFromOptions()

pc = ksp.getPC()
pc.setHYPREDiscreteGradient(G)
pc.setHYPRESetInterpolations(dim=3, ND_Pi_Full=Pi)
pc.setHYPREAMSSetInteriorNodes(i_nodes.vector)
# one_zero_zero = Function(V)
# zero_one_zero = Function(V)
# zero_zero_one = Function(V)

# one_zero_zero.interpolate(lambda x: np.vstack((np.ones_like(x[0]),
#                                                np.zeros_like(x[0]),
#                                                np.zeros_like(x[0]))))
# zero_one_zero.interpolate(lambda x: np.vstack((np.zeros_like(x[0]),
#                                                np.ones_like(x[0]),
#                                                np.zeros_like(x[0]))))
# zero_zero_one.interpolate(lambda x: np.vstack((np.zeros_like(x[0]),
#                                                np.zeros_like(x[0]),
#                                                np.ones_like(x[0]))))

# pc.setHYPRESetEdgeConstantVectors(one_zero_zero.vector,
#                                   zero_one_zero.vector,
#                                   zero_zero_one.vector)
# pc.setHYPRESetBetaPoissonMatrix(None)
pc.setUp()
ksp.setUp()
ksp.view()

ksp.solve(b, u.vector)
# u.x.scatter_forward()

print(f"Convergence reason: {ksp.getConvergedReason()}")
# save_function(u, "output.bp")
