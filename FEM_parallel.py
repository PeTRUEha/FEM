import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict
from multiprocessing.pool import Pool

from geometry_configuration import fix_in_place, configure_geometry
from utils import *
from mesh import Node, Curve, Element, local_stiffness, build_graph
from plots import plot_over_line


class GlobalStiffness(sparse.lil_matrix):
    def update(self, element: Element):
        K_local = local_stiffness(element)
        for i, j in np.ndindex((3, 3)):
            self[2 * element.nodes[i].ID, 2 * element.nodes[j].ID] += K_local[2 * i, 2 * j]
            self[2 * element.nodes[i].ID + 1, 2 * element.nodes[j].ID] += K_local[2 * i + 1, 2 * j]
            self[2 * element.nodes[i].ID, 2 * element.nodes[j].ID + 1] += K_local[2 * i, 2 * j + 1]
            self[2 * element.nodes[i].ID + 1, 2 * element.nodes[j].ID + 1] += K_local[2 * i + 1, 2 * j + 1]


def global_stiffness():
    N = len(Node.get)
    K = GlobalStiffness((2 * N, 2 * N))
    for element in Element.get.values():
        K.update(element)
    return K


def parallel_global_stiffness(colors: Dict[int, int]) -> GlobalStiffness:
    N = len(Node.get)
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


def rhs():
    N = len(Node.get)
    R = np.zeros((2 * N, 1))
    for element in Element.get.values():
        for edge in element.edges:
            if edge.is_border():
                n = edge.get_inner_normal()
                boundary_condition = edge.get_boundary_condition()
                for node in edge.nodes:
                    R[2 * node.ID] += boundary_condition * n[0] * edge.length
                    R[2 * node.ID + 1] += boundary_condition * n[1] * edge.length

    return R / 2


@print_execution_time('Equation system assembly')
def assemble_equation_system():
    # Заполнение матрицы жёсткости
    # K = parallel_global_stiffness()
    K = global_stiffness()
    # Заполнение правой части
    R = rhs()
    # Применяем фиксирующие граничные условия
    for edge in Curve.get[1].edges:
        fix_in_place(K, R, edge.nodes[0], 'x')
        fix_in_place(K, R, edge.nodes[1], 'x')

    for edge in Curve.get[3].edges:
        fix_in_place(K, R, edge.nodes[0], 'y')
        fix_in_place(K, R, edge.nodes[1], 'y')
    return K, R


@print_execution_time('Writing arrays into elements and nodes')
def calculate_array_values(U):
    for i in range(len(Node.get)):
        Node.get[i].values['displacement'] = np.array([U[2 * i], U[2 * i + 1]])

    for el in Element.get.values():
        el.get_strain()
        el.get_stress()

if __name__ == "__main__":
    configure_geometry()
#    graph = build_graph()
    # colors = nx.greedy_color(graph)
    # nx.draw(graph, node_size=100, labels=colors, font_color='black')
    # plt.show()
    # exit(0)
    N = len(Node.get)
    K, R = assemble_equation_system()
    U = print_execution_time("System solution with spsolve")(spsolve)(K, R)
    calculate_array_values(U)
    plot_over_line()