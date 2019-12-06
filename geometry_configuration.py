import re

from constants import RSplit, FILENAME, P1, P2
from mesh import Node, Element, Curve, Edge, Mesh
from utils import print_execution_time
import networkx as nx


class MeshReader:
    def __init__(self, file):
        self.file = file
        self.renumbering = {}
        self.nodes = []
        self.elements = []

    def skip_lines(self, n_lines):
        for i in range(n_lines):
            next(self.file)

    def read_2d_mesh(self):
        """Считывает данные о сетке из файла .inp, перенумеровывая
        узлы от нуля"""
        self.skip_to("N O D E S")
        self.skip_lines(1)  # пропускаем строку
        self.read_nodes()
        self.skip_to('E L E M E N T S')
        self.skip_lines(1)  # пропускаем строку
        self.read_elements()
        return Mesh(self.nodes, self.elements)

    def skip_to(self, string):
        line = next(self.file)
        while line:
            if string in line:
                break
            line = next(self.file, "")

    def read_nodes(self):
        nodes_count = 0
        line = next(self.file)
        while line:
            match = re.search(r'\s+(\S+),\s+(\S+),\s+(\S+\d),*\s+', line)
            if match:
                n_old = int(match.group(1))
                n_new = nodes_count
                nodes_count += 1
                self.renumbering.update({n_old: n_new})

                x, y = map(float, [match.group(2),
                                   match.group(3)])
                self.nodes.append(Node(n_new, x, y))
                line = next(self.file)
            else:
                break

    def read_elements(self):
        line = next(self.file)
        while line:
            items = line.split(',')
            if len(items) == 1:
                break
            else:
                n = int(items[0])
                its_nodes = map(int, items[1:])
                new_nodes = [self.renumbering[node] for node in its_nodes]
                self.elements.append(Element(n, new_nodes))
                # print(new_element)
            line = next(self.file)

def fix_in_place(K, R, node, how='x'):
    for i in range(2 * len(Node.get)):
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


def create_curves():
    #0 - внутренняя поверхность
    #1 - поверхность слева-сверху
    #2 - внешняя поверхность
    #3 - поверхность справа-снизу
    curves = {i: Curve(i) for i in range(4)}

    for edge in Edge.get.values():
        if edge.is_border():
            c = edge.get_centre()
            if abs(c[0]) < 0.001:
                curves[1].add(edge)
            elif abs(c[1]) < 0.001:
                curves[3].add(edge)
            elif c[0] ** 2 + c[1] ** 2 < RSplit ** 2:
                curves[0].add(edge)
            else:
                curves[2].add(edge)

    curves[0].boundary_condition = P1
    curves[2].boundary_condition = P2
    return curves


@print_execution_time('Geometry configuration')
def configure_geometry():
    with open(FILENAME, 'r') as file:
        mesh = MeshReader(file).read_2d_mesh()
    curves = create_curves()
    mesh.add_curves(curves)
