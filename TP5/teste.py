import gmsh
from mpi4py import MPI
from dolfinx.plot import vtk_mesh
import pyvista
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import default_scalar_type
from dolfinx.io import gmshio 
import matplotlib.pyplot as plt 
from dolfinx import geometry
from dolfinx.mesh import locate_entities_boundary
from dolfinx.fem.petsc import LinearProblem
from scipy.constants import pi, epsilon_0
from dolfinx.fem import (Constant, dirichletbc, Function, Expression, functionspace, assemble_scalar,
                         form, locate_dofs_geometrical, locate_dofs_topological)
from ufl import (SpatialCoordinate, TestFunction, TrialFunction, as_vector,
                 dx, dot, grad, inner)
import ufl


gdim = 2  # Densidade da malha 2D
model_rank = 0
mesh_comm = MPI.COMM_WORLD

# Parâmetros do problema
rc = 0.0075
zc = 0.06
x = 0.005
rs = 0.01
m = 0.022
zb = 0.04
awg = 18
N = 200 

dimensao_x = dimensao_y = 0.2
buracos = []


# Wire
class AWG:
    def __init__(self, type):
        self.AWG = type
        self.diameter = 0.127e-3*92**((36-type)/39) #(m)
        self.area = np.pi/4*self.diameter**2
        self.sigma = 58.0e6 # (S/m)
awg = AWG(18) # (default 18)

mu0_constant = 4e-7 * np.pi  # Permeabilidade magnética do vácuo (H/m)
muf_constant = 1200 * mu0_constant    # Permeabilidade do núcleo
I = 1               # Corrente inicial (A)
R = 0.05            # Raio da esfera (m)
coil_radius = 0.06  # Raio externo da bobina (m)
z_length = 0.1      # Altura da bobina (m)
coil_area = np.pi * (coil_radius**2 - R**2)
J_density = I / coil_area
print(f"Densidade de corrente J = {J_density:.2e} A/m²")


# Inicializando o GMSH
gmsh.initialize()

ar = gmsh.model.occ.addRectangle(0, 0, 0, dimensao_x, dimensao_y, tag=4)
gmsh.model.occ.synchronize()

esfera = gmsh.model.occ.addDisk(0, rs + 0.01, 0, rs, rs,tag=5)
gmsh.model.occ.synchronize()
buracos.append((2,5))

nucleo = gmsh.model.occ.addRectangle(0, 2*rs + 0.02,0, rc, zc, tag=2)
gmsh.model.occ.synchronize()
buracos.append((2,2))

fio =  gmsh.model.occ.addRectangle(rc, 2*rs + 0.04 + rc/2,0, awg.diameter, zc/2, tag=3)
gmsh.model.occ.synchronize()
buracos.append((2,3))

# Criando o dominio
whole_domain, map_to_input = gmsh.model.occ.fragment([(2, ar)], buracos)
gmsh.model.occ.synchronize()



# Adicionado Fisica ao modelo
gmsh.model.add_physical_group(dim=2, tags=[ar],tag=4,name="ar")
gmsh.model.add_physical_group(dim=2, tags=[nucleo],tag=2,name="nucleo")
gmsh.model.add_physical_group(dim=2, tags=[fio],tag=3,name="fio")
gmsh.model.add_physical_group(dim=2, tags=[esfera],tag=5,name="esfera")
gmsh.model.occ.synchronize()



# Tamanho da malha
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1/(2**9))

# Gerando a malha
gmsh.model.mesh.generate(2)


# COnvertendo o modelo GMSH para um mesh do dolfinx
mesh,cell_tags,facet_tags = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2)
# Finaliza o GMSH
gmsh.write("mesh.msh")
gmsh.finalize()

pyvista.start_xvfb()


# Configura a visualização com PyVista
#plotter = pyvista.Plotter()
#mesh.topology.create_connectivity(2, 2)
#grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh, 2))
#num_local_cells = mesh.topology.index_map(2).size_local
#grid.cell_data["Marker"] = cell_tags.values[cell_tags.indices < num_local_cells]
#grid.set_active_scalars("Marker")

#actor = plotter.add_mesh(grid, show_edges=True, cmap="viridis", edge_color="black")

#plotter.view_xy()
#plotter.show()


V = functionspace(mesh, ("Lagrange", 2))
tdim = mesh.topology.dim

# Definido as cordenadas espaciais
x = SpatialCoordinate(mesh)
r = x[0]

# Encontrando boundaries e aplicando condições de contorno
facets = locate_entities_boundary(mesh, tdim - 1, lambda x: np.full(x.shape[1], True))
dofs = locate_dofs_topological(V, tdim - 1, facets)
bc = dirichletbc(default_scalar_type(0), dofs, V)


Q = functionspace(mesh, ("DG", 0))
mu = Function(Q) #

# Permeability
mu0 = Constant(mesh, PETSc.ScalarType(4e-7*np.pi))
muf = Constant(mesh, PETSc.ScalarType(1200*4e-7*np.pi))

# List of elements in each domain
core_elements = cell_tags.find(2) # nucleo = 3
coil_elements = cell_tags.find(3) # fio = 4
cyl_elements = cell_tags.find(5) # esfera = 2 
air_elements = cell_tags.find(4) # ar = 1

# Defining the permeability values
mu.x.array[core_elements] = np.full_like(core_elements, muf, dtype=default_scalar_type)
mu.x.array[coil_elements] = np.full_like(coil_elements, mu0, dtype=default_scalar_type)
mu.x.array[cyl_elements] = np.full_like(cyl_elements, muf, dtype=default_scalar_type)
mu.x.array[air_elements] = np.full_like(air_elements, mu0, dtype=default_scalar_type)



# Densidade de corrente
j = I/awg.area
print('current density, J =',j)

# Area da seção transversal do fio
j = N*I/(zc * rc)
print('current density, J =',j)

# Creates a function J in the space Q that is non-zero only inside the coil
J = Function(Q)
Js = Constant(mesh, PETSc.ScalarType(j))
J.x.array[coil_elements] = np.full_like(coil_elements, Js, dtype=default_scalar_type)

# Define Trial and Test functions
u = TrialFunction(V)
v = TestFunction(V)

print("Teste-------1")

dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags);

# Variational equation with cylindrical coordinates
a = (1 / mu) * (1 / r) * dot(grad(u), grad(v)) * dx

# Source term of the axially symmetric weak form
L = J * v * dx

# Solving the linear problem
A_ = Function(V) # A_ = r*A_alpha
problem = LinearProblem(a, L, u=A_, bcs=[bc])
print("Teste-------2")
problem.solve()
print("Teste-------3")

# -----------------------------------------------

# Creates the plotter
plotter = pyvista.Plotter(off_screen=True)

# Converts the mesh to an UnstructuredGrid
A_grid = pyvista.UnstructuredGrid(*vtk_mesh(V))

# Adds the A_z field data to the grid
A_grid.point_data["A_"] = A_.x.array
A_grid.set_active_scalars("A_")

# Applies the deformation based on the A_z field
warp = A_grid.warp_by_scalar("A_")

# Adds the deformed mesh to the plotter
actor = plotter.add_mesh(warp, show_edges=False)
actor = plotter.add_title(
    'Fig. 4 Potencial vetor magnético', font='courier', color='k', font_size=7)

plotter.view_xy()
plotter.screenshot("malha.png")

