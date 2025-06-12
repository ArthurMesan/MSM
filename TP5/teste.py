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
r_cilindro = 0.01
x_offset = 0.005
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
gmsh.model.add_physical_group(dim=2, tags=[ar],tag=1,name="ar")
gmsh.model.add_physical_group(dim=2, tags=[esfera],tag=2,name="esfera")
gmsh.model.add_physical_group(dim=2, tags=[nucleo],tag=3,name="nucleo")
gmsh.model.add_physical_group(dim=2, tags=[fio],tag=4,name="fio")
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

# Configura a visualização com PyVista
pyvista.start_xvfb()

# Configura a visualização com PyVista
plotter = pyvista.Plotter(off_screen=True)
mesh.topology.create_connectivity(2, 2)
grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh, 2))
num_local_cells = mesh.topology.index_map(2).size_local
grid.cell_data["Marker"] = cell_tags.values[cell_tags.indices < num_local_cells]
grid.set_active_scalars("Marker")

# Remova o color e use cmap para coloração baseada no "Marker"
actor = plotter.add_mesh(grid, show_edges=True, cmap="viridis", edge_color="black")

plotter.view_xy()
plotter.screenshot("malha.png")

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
mu = Function(Q)

# Permeability
mu0 = Constant(mesh, PETSc.ScalarType(mu0_constant))
muf = Constant(mesh, PETSc.ScalarType(muf_constant))

# List of elements in each domain
core_elements = cell_tags.find(3) 
coil_elements = cell_tags.find(4) 
cyl_elements = cell_tags.find(2)  
air_elements = cell_tags.find(1) 

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


dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)

# Variational equation with cylindrical coordinates
a = (1 / mu) * (1 / r) * dot(grad(u), grad(v)) * dx

# Source term of the axially symmetric weak form
L = J * v * dx

# Solving the linear problem
A_ = Function(V) # A_ = r*A_alpha
problem = LinearProblem(a, L, u=A_, bcs=[bc])
problem.solve()


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
actor = plotter.add_title('Potencial vetor magnético', font='courier', color='k', font_size=7)

plotter.view_xy()
plotter.screenshot("campo.png")


#Densidade de FLuxo Magnético B e Campo Magnético H
# Compute Magnetic flux density (B = curl A)
W = functionspace(mesh, ("DG", 0, (mesh.geometry.dim, )))
B = Function(W)
B_expr = Expression(as_vector((-(1/r)*A_.dx(1), (1/r)*A_.dx(0))), W.element.interpolation_points())
B.interpolate(B_expr)

# Compute Magnetic Field (H = B/mu)
H = Function(W)
H_expr = Expression(as_vector((-(1/r/mu)*A_.dx(1), (1/r/mu)*A_.dx(0))), W.element.interpolation_points())
H.interpolate(H_expr)

print("Teste-------1")


#Gráfico de B
# Iterpolate B again to mach vtk_mesh DoF.
Wl = functionspace(mesh, ("Lagrange", 2, (mesh.geometry.dim, )))
Bl = Function(Wl)
Bl.interpolate(B)

pyvista.start_xvfb()
topology, cell_types, geo = vtk_mesh(V)
values = np.zeros((geo.shape[0], 3), dtype=np.float64)
values[:, :len(Bl)] = Bl.x.array.real.reshape((geo.shape[0], len(Bl)))

# Create a point cloud of glyphs
function_grid = pyvista.UnstructuredGrid(topology, cell_types, geo)
function_grid["Bl"] = values
glyphs = function_grid.glyph(orient="Bl", factor=10,scale=True)

# Create a pyvista-grid for the mesh
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh, mesh.topology.dim))

# Create plotter
plotter = pyvista.Plotter(off_screen=True)
#plotter.add_mesh(grid, style="wireframe", color="k")
plotter.add_mesh(glyphs)
plotter.view_xy()
actor = plotter.add_title(
    'Densidade de fluxo magnético', font='courier', color='k', font_size=7)
#plotter.window_size = [1000, 250]
#plotter.camera.zoom(3)
plotter.screenshot("malha_campo_B.png")

print(geo.min(axis=0), geo.max(axis=0))



# Gráfico de H
# Iterpolate H again to mach vtk_mesh DoF.
Wl = functionspace(mesh, ("Lagrange", 2, (mesh.geometry.dim, )))
Hl = Function(Wl)
Hl.interpolate(H)

pyvista.start_xvfb()
topology, cell_types, geo = vtk_mesh(V)
values = np.zeros((geo.shape[0], 3), dtype=np.float64)
values[:, :len(Hl)] = Hl.x.array.real.reshape((geo.shape[0], len(Hl)))

# Create a point cloud of glyphs
function_grid = pyvista.UnstructuredGrid(topology, cell_types, geo)
function_grid["Hl"] = values
glyphs = function_grid.glyph(orient="Hl", factor=1.5e-5)

# Create a pyvista-grid for the mesh
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh, mesh.topology.dim))

# Create plotter
plotter = pyvista.Plotter(off_screen=True)
#plotter.add_mesh(grid, style="wireframe", color="k")
plotter.add_mesh(glyphs)
plotter.view_xy()
actor = plotter.add_title(
    'Fig. 6 Campo magnético', font='courier', color='k', font_size=7)
#plotter.window_size = [1000, 250];
#plotter.camera.zoom(3)
plotter.screenshot("malha_campo_H.png")


# Inedutancia - fluxo de corrente pela entrada de energia total
#Creating a balanced tree with the elements of the mesh
bb_tree = geometry.bb_tree(mesh, mesh.topology.dim) 

# sample points
tol = 1e-6
z_points = np.linspace(0,dimensao_y, 101)
r_points = np.full_like(dimensao_y, 0)
points = np.zeros((3, 101))
points[0] = r_points
points[1] = z_points

B_values = []
cells = []
points_on_proc = []

# Find cells whose bounding-box collide with the the points
cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
# Choose one of the cells that contains the point
colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, points.T)
for i, point in enumerate(points.T):
    if len(colliding_cells.links(i)) > 0:
        points_on_proc.append(point)
        cells.append(colliding_cells.links(i)[0])

points_on_proc = np.array(points_on_proc, dtype=np.float64)
B_values = B.eval(points_on_proc, cells)

# plot horizontal line
fig = plt.figure(figsize=(8,5))
plt.plot(z_points, B_values[:,0], 'y', linewidth=2); # magnify w
plt.plot(z_points, B_values[:,1], 'g', linewidth=2); # magnify w
plt.grid(True);


# Core and Cylinder boundaries
plt.plot(np.array([2*rs + 0.01 + x_offset + zc,2*rs + 0.01 + x_offset + zc]),np.array([0,np.max(B_values[:,1])]),'-.b', linewidth=1.5)
plt.plot(np.array([2*rs + 0.01 + x_offset,2*rs + 0.01 + x_offset]),np.array([0,np.max(B_values[:,1])]),'-.b', linewidth=1.5)
plt.plot(np.array([0.01,0.01]),np.array([0,np.max(B_values[:,1])]),'-.g', linewidth=1.5)
plt.plot(np.array([2*rs+r_cilindro,2*rs+r_cilindro]),np.array([0,np.max(B_values[:,1])]),'-.g',linewidth=1.5)


plt.legend(['Br (T)','Bz (T)', 'Core boudaries', 'Cylinder boundaries'], loc='upper left');
fig.suptitle("Fig. 7 Indução magnética");
plt.grid(True)
plt.xlabel('$z$ (m), r = 0')
plt.ylabel('$|B|$')

plt.savefig('inducao_magnetica.png')

# Campo nas superfícies
z_core = 5.0e-2 # (m) (default 5.0e-2)
r_air = 5*r_cilindro
z_air = 2.5*z_core
# Plot B
r_line = np.linspace(tol , r_air, 101)
z_line = z_core/2*np.ones(np.shape(r_line))

points = np.zeros((3, 101))
points[0] = r_line
points[1] = z_line

B_values = []
cells = []
points_on_proc = []

# Find cells whose bounding-box collide with the the points
cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
# Choose one of the cells that contains the point
colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, points.T)
for i, point in enumerate(points.T):
    if len(colliding_cells.links(i)) > 0:
        points_on_proc.append(point)
        cells.append(colliding_cells.links(i)[0])

points_on_proc = np.array(points_on_proc, dtype=np.float64)
B_values = B.eval(points_on_proc, cells)

# plot horizontal line
fig = plt.figure(figsize=(8,5))
plt.plot(r_line, B_values[:,0], 'y', linewidth=2); # magnify w
plt.plot(r_line, B_values[:,1], 'g', linewidth=2); # magnify w
plt.grid(True);


plt.legend(['Br (T)','Bz (T)'], loc='upper right');
fig.suptitle("Fig. 7 Indução magnética");
plt.grid(True)
plt.xlabel('$r$ (m), z = r_core/2')
plt.ylabel('$B$')

plt.savefig('Campos_na_superficie.png')