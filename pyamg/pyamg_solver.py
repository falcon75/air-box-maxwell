import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.mesh import Mesh, create_unit_cube, CellType, locate_entities, locate_entities_boundary
from dolfinx.fem import FunctionSpace, Function, Constant, form, locate_dofs_topological, Expression, dirichletbc, petsc, VectorFunctionSpace
from dolfinx.fem import assemble_matrix
from dolfinx.cpp.fem.petsc import (discrete_gradient, interpolation_matrix)
from dolfinx.io import XDMFFile, VTXWriter
from ufl import TrialFunction, TestFunction,  curl, dx, grad, inner, cos, as_vector, pi, SpatialCoordinate, VectorElement
import scipy.sparse

import scipy
import matplotlib.pyplot as plt
import pyamg


from edgeAMG import edgeAMG

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
                        cell_type=CellType.tetrahedron)
tdim = mesh.topology.dim
x = SpatialCoordinate(mesh)

# Define alpha and beta functions
S = FunctionSpace(mesh, ("DG", 0))
beta = Function(S)
alpha = Constant(mesh, 1.0)


def conductive_marker(x):
    inside = np.zeros(x.shape[1])
    for i in range(tdim):
        inside += np.logical_and(x[i] >= 0.3, x[i] <= 0.7)
    marked = inside == tdim
    return marked


conductive_cells = locate_entities(mesh, tdim, conductive_marker)
beta.x.array[conductive_cells] = 1e6    # set beta in conductive region to 1
# set beta in non conductive region to 0
beta.x.array[:] = 1

# Define boundary condition (this is the exact solution from the constant beta cube)
u_e = as_vector((cos(pi * x[1]), cos(pi * x[2]), cos(pi * x[0])))
f = curl(curl(u_e)) + u_e

# Formulation
degree = 1
V = FunctionSpace(mesh, ("N1curl", degree))
ndofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

u = TrialFunction(V)
v = TestFunction(V)

a = form(inner(alpha * curl(u), curl(v)) * dx + inner(beta * u, v) * dx)
L = form(inner(f, v) * dx)


V_H1 = FunctionSpace(mesh, ("Lagrange", degree))
u1 = TrialFunction(V_H1)
v1 = TestFunction(V_H1)
a_h1 = form(inner(alpha * grad(u1), grad(v1)) * dx + inner(beta * u1, v1) * dx)


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

A = assemble_matrix(a, bcs=[bc])
A.finalize()


A_node = assemble_matrix(a_h1, bcs=[bc])
A_node.finalize()

Anode = scipy.sparse.csr_matrix((A_node.data, A_node.indices, A_node.indptr))

b = petsc.assemble_vector(L)
petsc.apply_lifting(b, [a], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD,
              mode=PETSc.ScatterMode.REVERSE)
petsc.set_bc(b, [bc])


# Create a scipy sparse matrix that shares data with A
As = scipy.sparse.csr_matrix((A.data, A.indices, A.indptr))

x0 = np.ones_like(b.array)
r_None = []
r_SA = []
r_edgeAMG = []


# SA solver
ml_SA = pyamg.smoothed_aggregation_solver(As)
ML_SAOP = ml_SA.aspreconditioner()
x_prec, info = pyamg.krylov.cg(As, b, x0, maxiter=200, M=ML_SAOP, tol=1e-8, residuals=r_SA)


# edgeAMG
W = FunctionSpace(mesh, ("CG", degree))
G = discrete_gradient(W._cpp_object, V._cpp_object)
G.assemble()
ai, aj, av = G.getValuesCSR()
D = scipy.sparse.csr_matrix((av, aj, ai))
ml = edgeAMG(Anode, As, D)
MLOp = ml.aspreconditioner()
x_prec, info = pyamg.krylov.cg(
    As, b, x0, maxiter=200, M=MLOp, tol=1e-8, residuals=r_edgeAMG)


# No preconditioner
x_prec, info = pyamg.krylov.cg(
    As, b, x0, maxiter=200, M=None, tol=1e-8, residuals=r_None)


fig, ax = plt.subplots()
ax.semilogy(np.arange(0, len(r_edgeAMG)), r_edgeAMG, label='edge AMG')
ax.semilogy(np.arange(0, len(r_None)), r_None, label='CG')
ax.semilogy(np.arange(0, len(r_SA)), r_SA, label='CG + AMG')
ax.grid(True)
plt.legend()

plt.show()
