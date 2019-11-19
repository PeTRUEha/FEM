import re
import math

import numpy as np
from numpy import linalg, matrix, linspace
from scipy import integrate, sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from utils import *
from constants import *
from meshing import Node, Curve, Edge, Element


FILENAME = 'meshes\\donut5.inp'

def read_2d_mesh(filename):
    """Считывает данные о сетке из файла .inp, перенумеровывая
    узлы от нуля"""
    with open(filename, 'r') as file:

        nodes_found = 0
        elements_found = 0

        line = next(file)

        while line:
            if 'N O D E S' in line:
                nodes_found = 1
                break
            line = next(file, "")

        if not nodes_found:
            print('Nodes not found')
            quit()

        line = next(file)  # пропускаем строку
        line = next(file)

        nodes_count = 0
        renumbering = dict()

        while line:
            match = re.search(r'\s+(\S+),\s+(\S+),\s+(\S+\d),*\s+', line)
            if match:
                n_old = int(match.group(1))
                n_new = nodes_count
                nodes_count += 1
                renumbering.update({n_old: n_new})

                x, y = map(float, [match.group(2),
                                   match.group(3)])
                Node(n_new, x, y)
            else:
                break

            line = next(file)

        while line:
            if 'E L E M E N T S' in line:
                elements_found = 1
                break
            line = next(file, "")

        if not elements_found:
            print('Elements not found')
            quit()

        line = next(file)  # пропускаем строку
        line = next(file)

        while line:
            items = line.split(',')
            if len(items) == 1:
                break
            else:
                n = int(items[0])
                its_nodes = map(int, items[1:])
                new_nodes = [renumbering[node] for node in its_nodes]
                Element(n, new_nodes)
                # print(new_element)
            line = next(file)


def probe_location_from_nodes(kind, x, y):
    for e in Element.get.values():
        if e.covers(x, y):
            return e.probe_location_from_nodes(kind, x, y)
    return np.nan


def probe_location_from_element(kind, x, y):
    for e in Element.get.values():
        if e.covers(x, y):
            return e.probe_location_from_element(kind, x, y)
    return np.nan


def local_stiffness(element):
    """ Возвращает локальную матрицу жёсткости элемента,
    iй координате в матрице соответствует iй узел в списке узлов элемента"""
    n = element.nodes
    x1, x2, x3 = n[0].x, n[1].x, n[2].x
    y1, y2, y3 = n[0].y, n[1].y, n[2].y
    B = 0.5 / element.area * matrix([[y2 - y3, 0, y3 - y1, 0, y1 - y2, 0],
                                     [0, x3 - x2, 0, x1 - x3, 0, x2 - x1],
                                     [x3 - x2, y2 - y3, x1 - x3, y3 - y1, x2 - x1, y1 - y2]])

    D = matrix([[LAMBDA + 2 * MU, LAMBDA, 0],
                [LAMBDA, LAMBDA + 2 * MU, 0],
                [0, 0, MU]])
    K_local = element.area * B.T * D * B
    return K_local


def plot_over_line(line=linspace(0, 2, 50)):
    values = [probe_location_from_nodes('displacement', x, 0) for x in line]
    eps = [probe_location_from_element('strain', x, 0) for x in line]
    sigma = [probe_location_from_element('stress', x, 0) for x in line]

    x_values = [value[0] if not type(value) == float else math.nan
                for value in values]
    y_values = [value[1] if not type(value) == float else math.nan
                for value in values]
    eps11_values = [value[0] if not type(value) == float else math.nan
                    for value in eps]
    sigma11_values = [value[0] if not type(value) == float else math.nan
                      for value in sigma]
    sigma22_values = [value[1] if not type(value) == float else math.nan
                      for value in sigma]

    plot_graph(line, x_values, 'displacement x')
    plot_graph(line, y_values, 'displacement y')
    plot_graph(line, eps11_values, 'eps11')
    plot_graph(line, sigma11_values, 'sigma11')
    plot_graph(line, sigma22_values, 'sigma22')


def plot_graph(line, values, title):
    plt.plot(line, values)
    plt.title(title)
    plt.show()


def visualize():
    X = linspace(0, 2, 50)
    Y = linspace(0, 2, 50)

    values = [[probe_location_from_nodes('displacement', x, y) for x in X] for y in Y]
    eps = [[probe_location_from_element('strain', x, y) for x in X] for y in Y]

    x_values = [[value[0] if not type(value) == float else math.nan for value in values[i]] for i in range(50)]
    y_values = [[value[1] if not type(value) == float else math.nan for value in values[i]] for i in range(50)]
    abs_values = [[np.sqrt(x_values[i][j] ** 2 + y_values[i][j] ** 2) for j in range(50)] for i in range(50)]
    eps11 = [[value[0] if not type(value) == float else math.nan for value in eps[i]] for i in range(50)]
    eps22 = [[value[1] if not type(value) == float else math.nan for value in eps[i]] for i in range(50)]

    visualize_array(x_values, 'x_displacement', X=X, Y=Y)
    visualize_array(y_values, 'y_displacement', X=X, Y=Y)
    visualize_array(abs_values, 'abs_displacement', X=X, Y=Y)
    visualize_array(eps11, 'eps11', X=X, Y=Y)
    visualize_array(eps22, 'eps22', X=X, Y=Y)


def visualize_array(values, title, X=linspace(0, 2, 50), Y=linspace(0, 2, 50)):
    sp = plt.subplot()
    plt.title(title)
    c_x = sp.pcolor(X, Y, values, cmap='plasma', vmin=np.nanmin(values), vmax=np.nanmax(values))
    plt.colorbar(c_x)
    plt.show()


def plot_stiffness(K):
    fig = plt.figure(figsize=(14, 12))
    stiffness = fig.add_subplot(111)
    stiffness.set_title('stiffness')
    array = K.toarray()
    c_y = stiffness.pcolor(range(2 * N), range(2 * N), array, cmap='plasma', vmin=np.nanmin(array),
                           vmax=np.nanmax(array))
    plt.colorbar(c_y)
    plt.show()


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

    return R / 2  # не знаю, почему на 2


def fix_in_place(K, R, node, how='x'):
    for i in range(2 * N):
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


def set_curves():
    Curve(1)  # внутренняя поверхность
    Curve(2)  # поверхность слева-сверху
    Curve(3)  # внешняя поверхность
    Curve(4)  # поверхность справа-снизу

    for edge in Edge.get.values():
        if edge.is_border():
            c = edge.get_centre()
            if abs(c[0]) < 0.001:
                Curve.get[2].add(edge)
            elif abs(c[1]) < 0.001:
                Curve.get[4].add(edge)
            elif c[0] ** 2 + c[1] ** 2 < RSplit ** 2:
                Curve.get[1].add(edge)
            else:
                Curve.get[3].add(edge)


@print_execution_time('Mesh configuration')
def prepare_mesh():
    read_2d_mesh(FILENAME)
    set_curves()


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


def calculate_array_values(U):
    for i in range(len(Node.get)):
        Node.get[i].values['displacement'] = np.array([U[2 * i], U[2 * i + 1]])

    for el in Element.get.values():
        el.get_strain()
        el.get_stress()


prepare_mesh()
N = len(Node.get)
K, R = assemble_equation_system()
U = print_execution_time("System solution with spsolve")(spsolve)(K, R)
calculate_array_values(U)