import math

import numpy as np
from numpy import matrix
import networkx as nx
import itertools
import matplotlib.pyplot as plt

from utils import triangle_area_2d, is_to_the_left, WrongElementTypeError, print_execution_time
from constants import LAMBDA, MU, Ntr


class Mesh:
    def __init__(self, nodes, elements):
        self.nodes = nodes
        self.elements = elements


class Node:
    get = dict()

    def __init__(self, ID, x, y):
        self.ID = ID
        self.x = x
        self.y = y
        # self.elements = []
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
        self.curve = None

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
        return self.curve.boundary_condition(c[0], c[1])


class Curve:
    get = dict()

    def __init__(self, name):
        self.name = name
        Curve.get.update({name: self})
        self.edges = []
        # normal pressure on the curve
        self.boundary_condition = lambda x, y: 0

    def add(self, edge):
        edge.curve = self
        self.edges.append(edge)


class Element:
    """Конечный элемент первого порядка. Предполагается, что
    к моменту инициализации словарь с узлами уже есть и
    назначен в статическую переменную get"""

    get = dict()
    graph = nx.Graph()

    def __init__(self, ID, node_ids):
        self.ID = ID
        self.nodes = [Node.get[node_id] for node_id in list(node_ids)]
        self.edges = list()
        Element.get.update({self.ID: self})
        self.values = {}

        self.area = self.calculate_area()

        for i in range(-1, len(self.nodes) - 1, 1):
            Edge.add_edge_info(self.nodes[i], self.nodes[i + 1], self)

        # for node in self.nodes:
        #     node.elements.append(self)

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


    def probe_location_from_nodes(self, kind, x, y):
        return sum([self.get_form_function(node, x, y) * node.values[kind]
                    for node in self.nodes])

    def probe_location_from_element(self, kind, x, y):
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


@print_execution_time("Graph construction")
def build_graph():
    graph = nx.Graph()
    for node in Node.get.values():
        for element1, element2 in set(itertools.combinations(set(node.elements), 2)):
            graph.add_edge(element1.ID, element2.ID)
    return graph