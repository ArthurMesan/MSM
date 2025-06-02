# Basic Libraries
import numpy as np # For vector and matrix manipulation
import gmsh # Interface with the software GMSH
from scipy.integrate import solve_ivp

# Visualization 
import pyvista
from mpi4py import MPI
import matplotlib.pyplot as plt

# FEniCS
from ufl import (SpatialCoordinate, TestFunction, TrialFunction, as_vector,
                 dx, dot, grad, inner)
from petsc4py import PETSc

from dolfinx import default_scalar_type
from dolfinx.mesh import create_mesh, compute_midpoints, locate_entities_boundary, exterior_facet_indices
from dolfinx.io import gmshio, XDMFFile, VTKFile
from dolfinx.io.gmshio import model_to_mesh
from dolfinx.fem import (Constant, dirichletbc, Function, Expression, functionspace, assemble_scalar,
                         form, locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.plot import vtk_mesh
from dolfinx import fem, geometry

R = 0.12e-3 / 2 # Raio do capilar
d = 6e-3       # Distância entre o capilar e o eletrodo
Ra = 2e-3      # Raio do eletrodo interno
Rb = 9e-3      # Raio do eletrodo externo
L = Rb - Ra    # Comprimento do eletrodo
h = 0.05*L       # altura do eletrodo
y_capilar = R*25

x_air = Rb
y_air = (y_capilar+d)*1.2

rank = MPI.COMM_WORLD
gmsh.initialize()

gdim = 2  # Geometric dimension of the mesh
model_rank = 0
mesh_comm = MPI.COMM_WORLD

if mesh_comm.rank == model_rank:
    # Cria ar
    rectangle_air = gmsh.model.occ.addRectangle(0, 0, 0, x_air, y_air, tag=1)
    gmsh.model.occ.synchronize()
    
    # Cria o tubo capilar
    rectangle =  gmsh.model.occ.addRectangle(0, 0, 0, R, y_capilar, tag=2)
    gmsh.model.occ.synchronize()

    # Criar circulo
    circle = gmsh.model.occ.addDisk(0, y_capilar, 0, R, R)
    gmsh.model.occ.synchronize()

    # Criando o capilar unindo o retangulo e o circulo
    nozzle = gmsh.model.occ.fuse([(gdim, rectangle)], [(gdim, circle)])
    gmsh.model.occ.synchronize()

    # Cria o eletrodo
    eletrodo =  gmsh.model.occ.addRectangle(Ra, d+y_capilar+R, 0, L, h, tag=4)
    gmsh.model.occ.synchronize()
   
    ov, map_to_input = gmsh.model.occ.fragment([(gdim, rectangle_air)], [(gdim, rectangle), (gdim, eletrodo)])
    gmsh.model.occ.synchronize()

    names = ["air", "capilar", "eletrodo"]
    for i in range(3):
        gmsh.model.addPhysicalGroup(gdim, tags=[ov[i][1]], tag = i+1, name = names[i]) # external elements
        gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.0001)
    gmsh.model.mesh.generate(gdim)


# Converts the mesh created in GMSH to the format used by Dolfinx.
mesh, cell_tags, facet_tags = gmshio.model_to_mesh(
    gmsh.model, comm=mesh_comm, rank=model_rank, gdim=gdim
)
mesh.topology.create_connectivity(gdim, gdim)
# gmsh.write("malha.msh")
# Finaliza o GMSH
gmsh.finalize()


pyvista.start_xvfb()

# Extrai a topologia da malha para criar uma visualização em PyVista
topology, cell_types, x = vtk_mesh(mesh)
grid = pyvista.UnstructuredGrid(topology, cell_types, x)

# Configura a visualização com PyVista
plotter = pyvista.Plotter()

num_local_cells = mesh.topology.index_map(gdim).size_local
grid.cell_data["Marker"] = cell_tags.values[cell_tags.indices < num_local_cells]
grid.set_active_scalars("Marker")

# Remova o color e use cmap para coloração baseada no "Marker"
actor = plotter.add_mesh(grid, show_edges=True, cmap="viridis", edge_color="black")
plotter.view_xy()
plotter.add_axes()
plotter.show()


## CONTANTES
epsilon_0 = 8.854e-12  # Permissividade do vácuo
epsilon_r = 84         # Permissividade relativa da Formamida
epsilon = epsilon_r * epsilon_0
gamma = 0.05
voltage = np.sqrt((gamma*R)/epsilon_0)*np.log((4*d)/R)
print("Tensão Necessária para gerar o campo elétrico crítico", voltage)

e0 = Constant(mesh, PETSc.ScalarType(1))
er = Constant(mesh, PETSc.ScalarType(epsilon_r))

# SIMULANDO
# Lista de elementos de cada dominio
air_elements = cell_tags.find(1)
nozzle_elements = cell_tags.find(2)
eletrodo_elements = cell_tags.find(3)

# Constante eletrica
Q = functionspace(mesh, ("DG", 0))
epsilon = Function(Q) #relative epsilon

epsilon.x.array[air_elements] = np.full_like(air_elements, e0, dtype=default_scalar_type)
epsilon.x.array[nozzle_elements] = np.full_like(nozzle_elements, er, dtype=default_scalar_type)
epsilon.x.array[eletrodo_elements] = np.full_like(eletrodo_elements, e0, dtype=default_scalar_type)


# Condicoes de contorno
V = functionspace(mesh, ("Lagrange", 2))
tdim = mesh.topology.dim

# Definir a coordenada radial r
# Definir a coordenada radial r a partir de x[0]
x = SpatialCoordinate(mesh)
r = x[0]

# Definir graus de liberdade para as fronteiras
def points_capilar(x):
    cond_x = (x[0] <= R)
    cond_y = x[1] <= y_capilar+R
    return cond_x & cond_y

def points_eletrodo(x):
    cond_x = (x[0] > Ra)
    cond_y = x[1] >= d+y_capilar+R
    return cond_x & cond_y

facets_capilar = locate_entities_boundary(mesh, tdim - 1, points_capilar)
facets_eletrodo = locate_entities_boundary(mesh, tdim - 1, points_eletrodo)

dofs_capilar = locate_dofs_topological(V, tdim - 1, facets_capilar)
dofs_eletrodo = locate_dofs_topological(V, tdim - 1, facets_eletrodo)

# Condição de contorno para o capilar (potencial 0)
bc_capilar = dirichletbc(PETSc.ScalarType(voltage), dofs_capilar, V)
bc_eletrodo = dirichletbc(PETSc.ScalarType(0), dofs_eletrodo, V)

# Aplicar as condições de contorno
bcs = [bc_capilar, bc_eletrodo]

# Implementando a forma fraca
# Definir as funções teste e trial
u = TrialFunction(V)
v = TestFunction(V)

# Equação variacional com coordenadas cilíndricas
a = epsilon * r * dot(grad(u), grad(v)) * dx

# Segundo termo da forma fraca axissimétrica
L = e0 * v * dx
phi_ = Function(V)
problem = LinearProblem(a, L, u=phi_, bcs=bcs)
u = problem.solve()

# Cria o plotter
plotter = pyvista.Plotter()

# Converte a malha para um UnstructuredGrid
phi_grid = pyvista.UnstructuredGrid(*vtk_mesh(V))

# Adiciona os dados do campo A_z
phi_grid.point_data["phi_"] = u.x.array
phi_grid.set_active_scalars("phi_")

# Adiciona a malha deformada no plotter
actor = plotter.add_mesh(phi_grid, show_edges=False, n_colors=32)
plotter.view_xy()
plotter.add_axes()
plotter.show()