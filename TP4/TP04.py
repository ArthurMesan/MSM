import gmsh
from mpi4py import MPI
from dolfinx.plot import vtk_mesh
import numpy as np
from petsc4py import PETSc
from dolfinx import default_scalar_type
from dolfinx.io import gmshio 
from scipy.constants import pi, epsilon_0
from dolfinx.fem import (Constant, dirichletbc, Function, Expression, functionspace, assemble_scalar,
                         form, locate_dofs_geometrical, locate_dofs_topological)
from ufl import (SpatialCoordinate, TestFunction, TrialFunction, as_vector,
                 dx, dot, grad, inner)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities_boundary
from dolfinx.fem import locate_dofs_topological, dirichletbc
from petsc4py import PETSc
import pyvista
from dolfinx.plot import vtk_mesh
import dolfinx.plot
from scipy.constants import pi, epsilon_0
from scipy.integrate import solve_ivp
import sympy as sp



# GMSH
gdim = 2  # Dimensão geométrica
model_rank = 0
mesh_comm = MPI.COMM_WORLD


# Parametros do problema   
gamma = 0.05  # Tensão superficial (N/m)
epsilon_r = 84  # Permissividade relativa
d = 6e-3  # Distância capilar-barrier (m)
R = 0.12e-3  # Raio do bocal (m capilar) 
Rc = R  # Raio de curvatura igual ao bocal (radio do capilar)
rho = 1.13e3  # Densidade (kg/m^3)
Rg = 0.002e-3  # Raio da gotícula (m)

epsilon_0 = 8.854e-12
epsilon_r = 84
epsilon = 8.854e-12 * epsilon_r


#e0 = 8.854187817e-10 # Permeabilidade eletrica do vácuo

# Campo eletrico critico
E_critico = np.sqrt(pi * gamma / (2 * epsilon_0 * R))
# Tensão critica
V_critico = np.sqrt((gamma * Rc) / epsilon_0) * np.log(4 * d / Rc)

print("Tensão critica: ")
print(V_critico)


# Creação da malha com GMSH
gmsh.initialize()

# Creação do objeto
buraco = []

ar = gmsh.model.occ.addRectangle(0, 0, 0, 12e-3, 12e-3, tag=3)
gmsh.model.occ.synchronize()

#Tubo capilar                                raio do eletrodo externo
barreira = gmsh.model.occ.addRectangle(5e-3, 9e-3, 0, 7e-3, R*2, tag=1)
gmsh.model.occ.synchronize()
buraco.append((2,1))


bocal =  gmsh.model.occ.addRectangle(0, 0, 0, R, 3e-3, tag=2)
gmsh.model.occ.synchronize()
circulo = gmsh.model.occ.addDisk(0, 3e-3, 0, R, R)
gmsh.model.occ.synchronize()
entrada_bocal = gmsh.model.occ.fuse([(2, bocal)], [(2, circulo)])
gmsh.model.occ.synchronize()
buraco.append((2,2))

# Creando dominio
whole_domain, map_to_input = gmsh.model.occ.fragment([(2, ar)], buraco)
gmsh.model.occ.synchronize()


# Marcação fisica das regioes
# adição de grupos físicos
gmsh.model.add_physical_group(dim=2, tags=[ar],tag=1,name="ar")
gmsh.model.add_physical_group(dim=2, tags=[bocal],tag=2,name="bocal")
gmsh.model.add_physical_group(dim=2, tags=[barreira],tag=3,name="barreira")

# Função de sincronização que garente a sincronia entre o
# modelo geométrico e a malha
gmsh.model.occ.synchronize()

# Tamanha do malha
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1/(2**13))

# Gerando a malha 2D
gmsh.model.mesh.generate(2)

# Elementos Finitos foram criados pelo Gmsh,
# para a discretizzar a geometria
#gmsh.fltk.run()
#gmsh.finalize()

# Inicio da integração entre Gmsh e DOLFINX para a simulação do método
# dos elementos finitos.
# Conversão da malha para o DOLFINx
mesh,cell_tags,facet_tags = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2)
# Finaliza o GMSH
gmsh.write("mesh.msh")
gmsh.finalize()

#-------------------------------------------------------------------


from dolfinx.plot import vtk_mesh
import pyvista
pyvista.start_xvfb()

# Configura a visualização com PyVista
plotter = pyvista.Plotter(off_screen=True)  # off_screen garante que não abre a janela
mesh.topology.create_connectivity(2, 2)
grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh, 2))
num_local_cells = mesh.topology.index_map(2).size_local
grid.cell_data["Marker"] = cell_tags.values[cell_tags.indices < num_local_cells]
grid.set_active_scalars("Marker")

# Adiciona a malha com coloração
actor = plotter.add_mesh(grid, show_edges=True, cmap="viridis", edge_color="black")

plotter.view_xy()
plotter.screenshot("malha_com_marcadores.png")


#--------------------------------------------------------------------

#Espaço de funções para permissividade ralativa
# Define um espaço de funções Q no domínio da malha, usando elemtentos
# Disconti uos de Galerkin (DG) de GRAU 0, OU SEJA, funções constantes
# por celula.

Q = functionspace(mesh, ("DG", 0))
epsilon = Function(Q) # epsilon relativo

#Permissividade do vácuo
e0 = Constant(mesh, PETSc.ScalarType(1))
# Permissividade relativa de um material
er = Constant(mesh, PETSc.ScalarType(epsilon_r))

# Elementos do subconjunto na malha
# Os valores contantes são indicadores para cada região do domínio 
elementos_ar = cell_tags.find(1)
elementos_bocal = cell_tags.find(2)
# Definição do que seria um bloco
elementos_bloco = cell_tags.find(3)

# Atribuição dos valores de permissividade de cada grupo de células
epsilon.x.array[elementos_bloco] = np.full_like(elementos_bloco, e0, dtype=default_scalar_type)
epsilon.x.array[elementos_bocal] = np.full_like(elementos_bocal, er, dtype=default_scalar_type)
epsilon.x.array[elementos_ar] = np.full_like(elementos_ar, e0, dtype=default_scalar_type)


# Definição do espaço de funções continuas de grau
# 2 para o problema.
# TDIM pega a dimensão topológica da malha
#  para a variável de solução V.
V = functionspace(mesh, ("Lagrange", 2))
tdim = mesh.topology.dim

#Definição de cordenada especial e variavel radial e
x = SpatialCoordinate(mesh)
r = x[0]    # Eixo radial (x)

# Funções para encontrar fronteiras do domínio
def fronteira_bocal(x):
    return np.logical_and(np.isclose(x[0], R), x[1] <= 5e-3) #Proximo ao raio de 5mm
    
def froteira_barreira(x):
    return np.logical_and(np.isclose(x[1], 9e-3), x[0] > 2e-3) #Proximo ao raio de 2mm


# Estrutura de fronteira do bocal e barreira
#facets_capilar = locate_entities_boundary(mesh, tdim - 1, fronteira_bocal)
#facets_barreira = locate_entities_boundary(mesh, tdim - 1, froteira_barreira)

# Topologia dos pontos que foram triangulados
dofs_capilar = locate_dofs_geometrical(V, fronteira_bocal)
dofs_barreira = locate_dofs_geometrical(V, froteira_barreira)

# Condições de controrno para capilar (potencial 0)
bc_capilar = dirichletbc(PETSc.ScalarType(V_critico), dofs_capilar, V)
bc_eletrodo = dirichletbc(PETSc.ScalarType(0), dofs_barreira, V)

# Aplicando as condições
bcs = [bc_capilar, bc_eletrodo]


u = TrialFunction(V)
v = TestFunction(V)

# Equação em cordenadas cilindricas
a = epsilon * r * dot(grad(u), grad(v)) * dx

# Segundo termo da forma fraca????
L = e0 * v * dx 

phi = Function(V)

problema = LinearProblem(a, L, u = phi, bcs=bcs)
problema.solve()

# Cria o plotter
plotter = pyvista.Plotter(off_screen=True) 


# Converte a malha para um UnstructuredGrid
E_grid = pyvista.UnstructuredGrid(*vtk_mesh(V))

# Adiciona os dados do campo A_z
E_grid.point_data["phi"] = phi.x.array
E_grid.set_active_scalars("phi")

# Adiciona a malha deformada no plotter
actor = plotter.add_mesh(E_grid, show_edges=False,n_colors=32)
plotter.view_xy()
plotter.show(screenshot="linhas_de_campo.png")



#Amortecimento do camopo eletrico ao longo da linha horizontal da malha
#Calculo do campo eletrico
#from ufl import VectorElement
#element = VectorElement("CG", mesh.ufl_cell(), 1)
#W = functionspace(mesh, element)

#W = functionspace(mesh, ("CG", 0, (mesh.geometry.dim,)))
W = functionspace(mesh, ("DG", 0, (mesh.geometry.dim, )))
E = Function(W)
E_expr = Expression(as_vector((phi.dx(0), phi.dx(1))), W.element.interpolation_points())
E.interpolate(E_expr)

# Criando arvore de caixas delimitadoras
import matplotlib.pyplot as plt 
from dolfinx import geometry

bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)

# Gerando pontos de amostras atraves de uma linha
x = np.linspace(3e-3+2*Rc, 6e-3, 101)
points = np.zeros((3, 101))
points[1] = x

E_values = []
cells = []
points_on_proc = []

# Econtra pontos que colidem com as celulas da malha
cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
# Pegando as celulas que colidem com os pontos
colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, points.T)
for i, point in enumerate(points.T):
    if len(colliding_cells.links(i)) > 0:
        points_on_proc.append(point)
        cells.append(colliding_cells.links(i)[0])

points_on_proc = np.array(points_on_proc, dtype=np.float64)
E_values = E.eval(points_on_proc, cells)

fig = plt.figure(figsize=(8,5))
plt.plot(x, -E_values[:,1], 'k', linewidth=2); 
plt.grid(True);
plt.xlabel('$x$')
plt.legend(['E(V/s)'], loc='upper right')
fig.suptitle("$E(x)$")

plt.savefig("campo_eletrico_horizontal.png", dpi=300, bbox_inches='tight')


# Cálculo do movimento da gotícula sob o efeito do campo elétrico

m = (4 / 3) * pi * Rg**3 * rho  # Massa da gotícula
Q = 8 * pi * np.sqrt(epsilon_0 * gamma) * Rg**2 / (3 / 2)  # Carga da gotícula
x_eixo = x
E = -E_values[:, 1] - 3e-3  # Campo elétrico ao longo do eixo x

def campo_eletrico(t, y):
    x , v = y
    proximo = np.argmin(np.abs(x_eixo - x))
    Fe = Q * E[proximo] if x < d else 0 # Força elétrica
    a = Fe / m  # Aceleração
    return [v, a]

tempo = (0, 0.2e-3)  # Intervalo de tempo
y0 = [0, 0]  # Condições iniciais: posição inicial e velocidade inicial
t_eval = np.linspace(*tempo, 500)
solucao = solve_ivp(campo_eletrico, tempo, y0, t_eval=t_eval)

x_pos = solucao.y[0]
v_vel = solucao.y[1]


# Gráficos de posição e velocidade
plt.figure().set_figwidth(5)
plt.plot(solucao.t*1e3, x_pos*1000)
plt.xlabel("Tempo (ms)")
plt.ylabel("Posição (mm)")
plt.title("Posição da Gotícula")

plt.tight_layout()
plt.savefig("posição.png")

plt.figure().set_figwidth(5)
plt.plot(solucao.t*1e3, v_vel)
plt.xlabel("Tempo (ms)")
plt.ylabel("Velocidade (m/s)")
plt.title("Velocidade da Gotícula")

plt.tight_layout()
plt.savefig("velocidade.png")