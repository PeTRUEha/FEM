import re

from constants import RSplit, FILENAME, P1, P2
from mesh import Node, Element, Curve, Edge
from utils import print_execution_time
import networkx as nx


class MeshReader:
    def __init__(self, file):
        self.file = file
        self.renumbering = {}

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
                Node(n_new, x, y)
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
                Element(n, new_nodes)
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


@print_execution_time('Geometry configuration')
def configure_geometry():
    with open(FILENAME, 'r') as file:
        MeshReader(file).read_2d_mesh()
    create_curves()
    Curve.get[1].boundary_condition = P1
    Curve.get[3].boundary_condition = P2