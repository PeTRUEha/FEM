import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from geometry_configuration import fix_in_place, configure_geometry
from utils import *
from mesh import Node, Curve, Element, local_stiffness
from graphs import plot_over_line

def global_stiffness():
    N = len(Node.get)
    K = sparse.lil_matrix((2 * N, 2 * N))
    for element in Element.get.values():
        K_local = local_stiffness(element)
        for i, j in np.ndindex((3, 3)):
            K[2 * element.nodes[i].ID, 2 * element.nodes[j].ID] += K_local[2 * i, 2 * j]
            K[2 * element.nodes[i].ID + 1, 2 * element.nodes[j].ID] += K_local[2 * i + 1, 2 * j]
            K[2 * element.nodes[i].ID, 2 * element.nodes[j].ID + 1] += K_local[2 * i, 2 * j + 1]
            K[2 * element.nodes[i].ID + 1, 2 * element.nodes[j].ID + 1] += K_local[2 * i + 1, 2 * j + 1]
    return K


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
    K = global_stiffness()
    # Заполнение правой части
    R = rhs()
    # Применяем фиксирующие граничные условия
    for edge in Curve.get[2].edges:
        fix_in_place(K, R, edge.nodes[0], 'x')
        fix_in_place(K, R, edge.nodes[1], 'x')

    for edge in Curve.get[4].edges:
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


configure_geometry()
N = len(Node.get)
K, R = assemble_equation_system()
U = print_execution_time("System solution with spsolve")(spsolve)(K, R)
calculate_array_values(U)
plot_over_line()