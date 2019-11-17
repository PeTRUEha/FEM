import re
import math

import numpy as np
from numpy import linalg, matrix, linspace
from scipy import integrate, sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from utils import *


FILENAME = 'meshes\\donut5.inp'


class Node:
    get = dict()

    def __init__(self, ID, x, y, z=0):
        self.ID = ID
        self.x = x
        self.y = y
        self.z = z
        self.elements = []
        self.values = {}  # заполняется при инициализации элементов
        Node.get.update({self.ID: self})

    def __str__(self):
        return 'node {}:\n({}, {}, {})\n'.format(self.ID, self.x, self.y, self.z)


class Edge:
    get = dict()

    def __init__(self, node1, node2, element):
        """ID узлов на вход подаются упорядоченными по возрастанию,
        вместо конструктора рекомендуется использовать add_edge_info"""

        self.ID = (node1.ID, node2.ID)
        self.nodes = [node1, node2]
        self.elements = [element]
        self.length = self.get_len()

        Edge.get.update({self.ID: self})
        element.edges.append(self)

    def add_edge_info(node1, node2, element):
        ID = (min(node1.ID, node2.ID), max(node1.ID, node2.ID))
        if ID in Edge.get:
            Edge.get[ID].elements.append(element)
            element.edges.append(Edge.get[ID])
        else:
            Edge(Node.get[ID[0]],
                 Node.get[ID[1]],
                 element)

    def get_centre(self):
        a = self.nodes[0]
        b = self.nodes[1]
        centre = ((a.x + b.x) / 2, (a.y + b.y) / 2)
        return centre

    def get_inner_normal(self):
        '''находит для граничных элементов нормаль к границе, смотрящую внутрь'''
        a = self.nodes[0]
        b = self.nodes[1]
        # узел элемента, не находящийся на этом ребре
        other = (set(self.elements[0].nodes) - {a, b}).pop()

        v = np.array([-b.y + a.y, b.x - a.x])
        n = v / np.linalg.norm(v)

        v1 = np.array([other.x - a.x, other.y - a.y])

        if np.dot(n, v1) > 0:
            # print('a = {}\nb = {}\nb - a = {}\nother = {}\nn = {}'.format((a.x, a.y), (b.x, b.y), ((b.x - a.x) , (b.y-a.y)), (other.x, other.y), (n[0], n[1])))
            return n
        else:
            # print('a = {}\nb = {}\nb - a = {}\nother = {}\nn = {}'.format((a.x, a.y), (b.x, b.y), ((b.x - a.x) , (b.y-a.y)), (other.x, other.y), (n[0], n[1])))
            # print('reversing')
            return -1 * n

    def is_border(self):
        if len(self.elements) == 1:
            return True
        else:
            return False

    def get_len(self):
        a = self.nodes[0]
        b = self.nodes[1]
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    def get_boundary_condition(self):
        c = self.get_centre()
        if self.curve.name == 1:
            return P1(c[0], c[1])
        elif self.curve.name == 3:
            return P2(c[0], c[1])
        else:
            return 0


class Curve:
    get = dict()

    def __init__(self, name):
        self.name = name
        Curve.get.update({name: self})
        self.edges = []

    def add(self, edge):
        edge.curve = self
        self.edges.append(edge)


class Element:
    """Конечный элемент первого порядка. Предполагается, что
    к моменту инициализации словарь с узлами уже есть и
    назначен в статическую переменную get"""

    get = dict()

    def __init__(self, ID, node_ids):
        self.ID = ID
        self.nodes = [Node.get[node_id] for node_id in list(node_ids)]
        self.edges = list()
        Element.get.update({self.ID: self})
        self.values = {}

        self.area = self.calculate_area()

        for i in range(-1, len(self.nodes) - 1, 1):
            Edge.add_edge_info(self.nodes[i], self.nodes[i + 1], self)

        for node in self.nodes:
            node.elements.append(self)

    def calculate_area(self):
        nodes = self.nodes
        if len(nodes) == 3:
            return triangle_area_2d(*nodes)
        else:
            raise WrongElementTypeError('All elements have to be triangular')

    def __str__(self):
        return 'element {}:\nnodes: {}\n'.format(self.ID, [node.ID for node in self.nodes])

    def inverse_jacobian(self, ksi=0, eta=0):
        """ksi = Ax + b, это A"""
        if len(self.nodes) == 3:
            n = self.nodes
            x0, x1, x2, y0, y1, y2 = n[0].x, n[1].x, n[2].x, n[0].y, n[1].y, n[2].y
            denominator = (x1 * y0 - x2 * y0 - x0 * y1 + x2 * y1 + x0 * y2 - x1 * y2)
            a11 = -(y1 - y2) / denominator
            a12 = (x1 - x2) / denominator
            a21 = -(-y0 + y2) / denominator
            a22 = -(x0 - x2) / denominator
            return matrix([[a11, a12], [a21, a22]])

    def intercept(self):
        """ksi = Ax + b, это b"""
        if len(self.nodes) == 3:
            n = self.nodes
            x0, x1, x2, y0, y1, y2 = n[0].x, n[1].x, n[2].x, n[0].y, n[1].y, n[2].y
            denominator = (x1 * y0 - x2 * y0 - x0 * y1 + x2 * y1 + x0 * y2 - x1 * y2)
            b1 = (x2 * y1 - x1 * y2) / denominator
            b2 = -(x2 * y0 - x0 * y2) / denominator

            return matrix([[b1], [b2]])

    def get_ksi_eta(self, x, y):
        A = self.inverse_jacobian()
        b = self.intercept()
        ksiEta = (A * [[x], [y]] + b)
        ksi, eta = float(ksiEta[0][0]), float(ksiEta[1][0])
        return ksi, eta

    def get_form_function(self, node, x, y):
        (ksi, eta) = self.get_ksi_eta(x, y)
        index = self.nodes.index(node)
        res = Ntr[index](ksi, eta)
        return res

    def covers(self, x, y):
        [n1, n2, n3] = self.nodes
        p = (x, y)
        p1, p2, p3 = (n1.x, n1.y), (n2.x, n2.y), (n3.x, n3.y)
        if (is_to_the_left(p, p1, p2) * is_to_the_left(p3, p1, p2) >= 0
                and is_to_the_left(p, p1, p3) * is_to_the_left(p2, p1, p3) >= 0
                and is_to_the_left(p, p2, p3) * is_to_the_left(p1, p2, p3) >= 0):
            return True
        else:
            return False
    #  TODO: leave just one get_value
    def get_value(self, kind, x, y):
        return sum([self.get_form_function(node, x, y) * node.values[kind]
                    for node in self.nodes])

    def getValue1(self, kind, x, y):
        return self.values[kind]

    def get_strain(self):
        n = self.nodes
        x1, x2, x3 = n[0].x, n[1].x, n[2].x
        y1, y2, y3 = n[0].y, n[1].y, n[2].y
        B = 0.5 / self.area * matrix([[y2 - y3, 0, y3 - y1, 0, y1 - y2, 0],
                                      [0, x3 - x2, 0, x1 - x3, 0, x2 - x1],
                                      [x3 - x2, y2 - y3, x1 - x3, y3 - y1, x2 - x1, y1 - y2]])
        u1, u2, u3 = [n[i].values['displacement'][0] for i in range(3)]
        v1, v2, v3 = [n[i].values['displacement'][1] for i in range(3)]

        U = matrix([[u1], [v1], [u2], [v2], [u3], [v3]])
        Epsilon = B * U
        self.values['strain'] = Epsilon.T.reshape(-1, ).tolist()[0]

    def get_stress(self):
        D = matrix([[LAMBDA + 2 * MU, LAMBDA, 0],
                    [LAMBDA, LAMBDA + 2 * MU, 0],
                    [0, 0, MU]])
        eps = matrix(np.array(self.values['strain'])).T
        sigma = D * eps
        self.values['stress'] = sigma.T.reshape(-1, ).tolist()[0]


def triangle_area_2d(v1, v2, v3):
    M = matrix([[v1.x - v3.x, v1.y - v3.y],
                [v2.x - v3.x, v2.y - v3.y]])
    area = abs(linalg.det(M) / 2)
    return area

def is_to_the_left(p, p1, p2):
    """ if positive is to the left, if zero is on, if negative is to the right"""
    x, y = p
    x1, y1 = p1
    x2, y2 = p2
    D = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    return -D


def read_2d_mesh(filename):
    """Считывает данные о сетке из файла .inp, перенумеровывая
    узлы от нуля"""
    with open(filename, 'r') as file:

        nodes_found = 0
        elements_found = 0

        line = next(file)

        while line:
            if 'N O D E S' in line:
                print('found N O D E S !')
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
                print('finished reading nodes')
                break

            line = next(file)

        while line:
            if 'E L E M E N T S' in line:
                print('found E L E M E N T S!')
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
                print('finished reading elements')
                break
            else:
                n = int(items[0])
                its_nodes = map(int, items[1:])
                new_nodes = [renumbering[node] for node in its_nodes]
                Element(n, new_nodes)
                # print(new_element)
            line = next(file)


def probe_location(kind, x, y):
    for e in Element.get.values():
        if e.covers(x, y):
            return e.get_value(kind, x, y)
    return np.nan


def probe_location1(kind, x, y):
    for e in Element.get.values():
        if e.covers(x, y):
            return e.getValue1(kind, x, y)
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


def visualization():
    line = linspace(0, 2, 50)
    values = [probe_location('displacement', x, 0) for x in line]
    eps = [probe_location1('strain', x, 0) for x in line]
    sigma = [probe_location1('stress', x, 0) for x in line]

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

    plt.plot(line, x_values)
    plt.title('displacement x')
    plt.show()
    plt.plot(line, y_values)
    plt.title('displacement y')
    plt.show()
    plt.plot(line, eps11_values)
    plt.title('eps11')
    plt.show()
    plt.plot(line, sigma11_values)
    plt.title('sigma11')
    plt.show()
    plt.plot(line, sigma22_values)
    plt.title('sigma22')
    plt.show()

    X = linspace(0, 2, 50)
    Y = linspace(0, 2, 50)

    values = [[probe_location('displacement', x, y) for x in X] for y in Y]
    eps = [[probe_location1('strain', x, y) for x in X] for y in Y]

    x_values = [[value[0] if not type(value) == float else math.nan for value in values[i]] for i in range(50)]
    y_values = [[value[1] if not type(value) == float else math.nan for value in values[i]] for i in range(50)]
    abs_values = [[np.sqrt(x_values[i][j] ** 2 + y_values[i][j] ** 2) for j in range(50)] for i in range(50)]
    eps11 = [[value[0] if not type(value) == float else math.nan for value in eps[i]] for i in range(50)]
    eps22 = [[value[1] if not type(value) == float else math.nan for value in eps[i]] for i in range(50)]

    sp2 = plt.subplot()
    plt.title("x_displacement")
    c_x = sp2.pcolor(X, Y, x_values, cmap='plasma', vmin=np.nanmin(x_values), vmax=np.nanmax(x_values))
    plt.colorbar(c_x)
    plt.show()

    sp1 = plt.subplot()
    plt.title("y_displacement")
    c_y = sp1.pcolor(X, Y, y_values, cmap='plasma', vmin=np.nanmin(y_values), vmax=np.nanmax(y_values))
    plt.colorbar(c_y)
    plt.show()

    sp3 = plt.subplot()
    plt.title("abs_displacement")
    c_abs = sp3.pcolor(X, Y, abs_values, cmap='plasma', vmin=np.nanmin(abs_values), vmax=np.nanmax(abs_values))
    plt.colorbar(c_abs)
    plt.show()

    sp4 = plt.subplot()
    plt.title("eps11")
    c_eps11 = sp4.pcolor(X, Y, eps11, cmap='plasma', vmin=np.nanmin(eps11), vmax=np.nanmax(eps11))
    plt.colorbar(c_eps11)
    plt.show()

    sp5 = plt.subplot()
    plt.title("eps22")
    c_eps22 = sp5.pcolor(X, Y, eps22, cmap='plasma', vmin=np.nanmin(eps22), vmax=np.nanmax(eps22))
    plt.colorbar(c_eps22)
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
                boundaryCondition = edge.get_boundary_condition()
                for node in edge.nodes:
                    R[2 * node.ID] += boundaryCondition * n[0] * edge.length
                    R[2 * node.ID + 1] += boundaryCondition * n[1] * edge.length

    return R / 2  # не знаю, почему на 2


def fix(K, R, node, how='x'):
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


""" cписок функций формы для равностороннего треугольника с высотой один и координатами
- высотами до первой и второй стороны"""

Ntr = [lambda ksi, eta: ksi,
       lambda ksi, eta: eta,
       lambda ksi, eta: 1 - ksi - eta]

dNtr = [[1, 0],
        [0, 1],
        [-1, -1]]

RSplit = 1.5  # радиус для отличения внешней и внутренней границы
F = 0  # массовая сила

LAMBDA = 100
MU = 80

# Граничные условия (давление):
P1 = lambda x, y: -1  # внутри
P2 = lambda x, y: 2  # снаружи

# Считывание сетки из файла
read_2d_mesh(FILENAME)
set_curves()

N = len(Node.get)
# Заполнение матрицы жёсткости
K = global_stiffness()

# Заполнение правой части
R = rhs()

# Применяем фиксирующие граничные условия
for edge in Curve.get[2].edges:
    fix(K, R, edge.nodes[0], 'x')
    fix(K, R, edge.nodes[1], 'x')

for edge in Curve.get[4].edges:
    fix(K, R, edge.nodes[0], 'y')
    fix(K, R, edge.nodes[1], 'y')

# plot_stiffness(K)
# Решение системы KU = R
U = spsolve(K, R)

for i in range(len(Node.get)):
    Node.get[i].values['displacement'] = np.array([U[2 * i], U[2 * i + 1]])

for el in Element.get.values():
    el.get_strain()
    el.get_stress()

# Визуализация результатов
visualization()

