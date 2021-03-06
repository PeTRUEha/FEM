import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from multiprocessing.pool import Pool
from itertools import repeat
from typing import List, Tuple

from geometry_configuration import configure_geometry
from utils import *
from mesh import Element, local_stiffness, Mesh
from plots import plot_over_line, visualize
from constants import N_PROCESSES
from utils import split_list

class GlobalStiffnessLil(sparse.lil_matrix):
    def update(self, element: Element):
        K_local = local_stiffness(element)
        for i, j in np.ndindex((3, 3)):
            self[2 * element.nodes[i].ID, 2 * element.nodes[j].ID] += K_local[2 * i, 2 * j]
            self[2 * element.nodes[i].ID + 1, 2 * element.nodes[j].ID] += K_local[2 * i + 1, 2 * j]
            self[2 * element.nodes[i].ID, 2 * element.nodes[j].ID + 1] += K_local[2 * i, 2 * j + 1]
            self[2 * element.nodes[i].ID + 1, 2 * element.nodes[j].ID + 1] += K_local[2 * i + 1, 2 * j + 1]

    @staticmethod
    def from_elements(elements: List[Element], shape):
        K = GlobalStiffnessLil(shape)
        for element in elements:
            K.update(element)
        K_coo = K.tocoo()
        return K_coo

class GlobalStiffnessCoo(sparse.coo_matrix):
    def __init__(self, elem_count, shape: Tuple[int, int]):
        sparse.coo_matrix.__init__(self, shape)
        array_len = elem_count * 9 * 4
        self.row = np.zeros(array_len)
        self.col = np.zeros(array_len)
        self.data = np.zeros(array_len)
        self.last_array_index = 0

    def update(self, element: Element):
        K_local = local_stiffness(element)
        for i, j in np.ndindex((3, 3)):
            self[2 * element.nodes[i].ID, 2 * element.nodes[j].ID] = K_local[2 * i, 2 * j]
            self[2 * element.nodes[i].ID + 1, 2 * element.nodes[j].ID] = K_local[2 * i + 1, 2 * j]
            self[2 * element.nodes[i].ID, 2 * element.nodes[j].ID + 1] = K_local[2 * i, 2 * j + 1]
            self[2 * element.nodes[i].ID + 1, 2 * element.nodes[j].ID + 1] = K_local[2 * i + 1, 2 * j + 1]

    def __setitem__(self, key: Tuple[int, int], value):
        i, j = key
        self.row[self.last_array_index] = i
        self.col[self.last_array_index] = j
        self.data[self.last_array_index] = value
        self.last_array_index += 1

    @staticmethod
    def from_elements(elements: List[Element], shape):
        K = GlobalStiffnessCoo(len(elements), shape)
        for element in elements:
            K.update(element)
        return K

@print_execution_time('Serial global stiffness construction')
def global_stiffness(mesh: Mesh):
    N = len(mesh.nodes)
    K = GlobalStiffnessCoo(len(mesh.elements), (2 * N, 2 * N))
    for element in mesh.elements.values():
        K.update(element)
    K = K.tolil()
    return K

@print_execution_time('Parallel global stiffness construction')
def parallel_global_stiffness(mesh) -> GlobalStiffnessLil:
    N = len(mesh.nodes)
    shape = (2 * N, 2 * N)
    with Pool(N_PROCESSES) as pool:
        all_args = zip(split_list(list(mesh.elements.values()), N_PROCESSES), repeat(shape))
        results = pool.starmap(GlobalStiffnessCoo.from_elements, all_args)
    K = sparse.csr_matrix(shape)
    for result in results:
        K += result
    K = K.tolil()
#    print('conversion done')
    return K


@print_execution_time('Right hand side assembly')
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


class EquationSystem():
    def __init__(self, matrix, rhs):
        self.matrix = matrix
        self.rhs = rhs


    @print_execution_time("Fixating conditions application")
    def apply_fixating_conditions(self, mesh):
        """Fixing in place all nodes in curve 1 by x and curve 3 by y"""
        K = self.matrix
        R = self.rhs

        indices_to_fix = get_indices_to_fix(mesh)
        for index in indices_to_fix:
            K[index, :] = 0

        K = K.T

        for index in indices_to_fix:
            K[index, :] = 0
            K[index, index] = 1
            R[index] = 0

        self.matrix = K.tocsc().T
        self.rhs = R



def get_indices_to_fix(mesh):
    indices_to_fix = []
    for edge in mesh.curves[1].edges:
        for node in edge.nodes:
            indices_to_fix.append(2 * node.ID)

    for edge in mesh.curves[3].edges:
        for node in edge.nodes:
            indices_to_fix.append(2 * node.ID + 1)
    return indices_to_fix

def assemble_equation_system(mesh: Mesh, parallel: bool):
    # Заполнение матрицы жёсткости
    if parallel:
        K = parallel_global_stiffness(mesh)
    else:
        K = global_stiffness(mesh)
    R = rhs(mesh)
    eq_system = EquationSystem(K, R)
    eq_system.apply_fixating_conditions(mesh)
    return eq_system.matrix, eq_system.rhs




@print_execution_time('Total parallel')
def main_parallel():
    mesh = configure_geometry()
    K, R = assemble_equation_system(mesh, parallel=True)
    U = print_execution_time("System solution with spsolve")(spsolve)(K, R)
    mesh.parallel_calculate_array_values(U)
    return mesh


@print_execution_time('Total serial')
def main_serial():
    mesh = configure_geometry()
    K, R = assemble_equation_system(mesh, parallel=False)
    U = print_execution_time("System solution with spsolve")(spsolve)(K, R)
    mesh.calculate_array_values(U)
    return mesh

if __name__ == "__main__":
    mesh = main_parallel()
    visualize(mesh)
    print()

