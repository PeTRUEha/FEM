import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict
from multiprocessing.pool import Pool

from geometry_configuration import configure_geometry
from utils import *
from mesh import Node, Curve, Element, local_stiffness, Mesh
from plots import plot_over_line

class GlobalStiffness(sparse.lil_matrix):
    def update(self, element: Element):
        K_local = local_stiffness(element)
        for i, j in np.ndindex((3, 3)):
            self[2 * element.nodes[i].ID, 2 * element.nodes[j].ID] += K_local[2 * i, 2 * j]
            self[2 * element.nodes[i].ID + 1, 2 * element.nodes[j].ID] += K_local[2 * i + 1, 2 * j]
            self[2 * element.nodes[i].ID, 2 * element.nodes[j].ID + 1] += K_local[2 * i, 2 * j + 1]
            self[2 * element.nodes[i].ID + 1, 2 * element.nodes[j].ID + 1] += K_local[2 * i + 1, 2 * j + 1]


def global_stiffness(mesh: Mesh):
    N = len(mesh.nodes)
    K = GlobalStiffness((2 * N, 2 * N))
    for element in mesh.elements.values():
        K.update(element)
    return K


def parallel_global_stiffness(colors: Dict[int, int]) -> GlobalStiffness:
    N = len(mesh.nodes)
    K = GlobalStiffness((2 * N, 2 * N))
    for one_color_elements in invert_dict(colors).values():
        with Pool(1) as pool:
            pool.map(GlobalStiffness.update, [(K, Element.get[element_n]) for element_n in one_color_elements])
            pool.apply_async(GlobalStiffness.update, [(K, node) for node in one_color_elements])
        # for element_number in one_color_elements:
        #     K.update(Element.get[element_number])
    return K


def invert_dict(dictionary: Dict) -> Dict:
    inv_map = {}
    for k, v in dictionary.items():
        inv_map[v] = inv_map.get(v, [])
        inv_map[v].append(k)
    return inv_map


def rhs(mesh: Mesh):
    N = len(mesh.nodes)
    R = np.zeros((2 * N, 1))
    for edge in mesh.edges.values():
        if edge.is_border():
            n = edge.get_inner_normal()
            boundary_condition = edge.get_boundary_condition()
            for node in edge.nodes:
                R[2 * node.ID] += boundary_condition * n[0] * edge.length
                R[2 * node.ID + 1] += boundary_condition * n[1] * edge.length

    return R / 2


@print_execution_time('Equation system assembly')
def assemble_equation_system(mesh: Mesh):
    # Заполнение матрицы жёсткости
    # K = parallel_global_stiffness()
    K = global_stiffness(mesh)
    # Заполнение правой части
    R = rhs(mesh)
    # Применяем фиксирующие граничные условия
    for edge in mesh.curves[1].edges:
        fix_in_place(K, R, edge.nodes[0], 'x')
        fix_in_place(K, R, edge.nodes[1], 'x')

    for edge in mesh.curves[3].edges:
        fix_in_place(K, R, edge.nodes[0], 'y')
        fix_in_place(K, R, edge.nodes[1], 'y')
    return K, R


@print_execution_time('Writing arrays into elements and nodes')
def calculate_array_values(U, mesh):
    for i in range(len(mesh.nodes)):
        mesh.nodes[i].values['displacement'] = np.array([U[2 * i], U[2 * i + 1]])

    for el in mesh.elements.values():
        el.get_strain()
        el.get_stress()


def fix_in_place(K, R, node, how='x'):
    for i in range(len(R)):
        if how == 'x':
            K[2 * node.ID, i] = 0
            K[i, 2 * node.ID] = 0
        if how == 'y':
            K[2 * node.ID + 1, i] = 0
            K[i, 2 * node.ID + 1] = 0
    if how == 'x':
        K[2 * node.ID, 2 * node.ID] = 1
        R[2 * node.ID] = 0

    if how == 'y':
        K[2 * node.ID + 1, 2 * node.ID + 1] = 1
        R[2 * node.ID + 1] = 0

if __name__ == "__main__":
    mesh = configure_geometry()
    #    graph = build_graph()
    # colors = nx.greedy_color(graph)
    # nx.draw(graph, node_size=100, labels=colors, font_color='black')
    # plt.show()
    # exit(0)
    N = len(mesh.nodes)
    K, R = assemble_equation_system(mesh)
    U = print_execution_time("System solution with spsolve")(spsolve)(K, R)
    calculate_array_values(U, mesh)
    plot_over_line(mesh)