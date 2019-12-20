import re
import scipy.sparse as sp
from constants import RSplit, FILENAME, P1, P2
from mesh import Node, Element, Curve, Edge, Mesh
from utils import print_execution_time



class MeshReader:
    def __init__(self, file):
        self.file = file
        self.nodes = {}
        self.elements = []
        self.edges = {}

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
        self.create_edges()
        self.leave_only_border_edges()
        return Mesh(self.nodes.values(), self.elements, self.edges)

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

                x, y = map(float, [match.group(2),
                                   match.group(3)])
                self.nodes[n_old] = (Node(n_new, x, y))
                line = next(self.file)
            else:
                break

    def read_elements(self):
        line = next(self.file)
        node_count = 0
        while line:
            items = line.split(',')
            if len(items) == 1:
                break
            else:
                n = node_count
                node_count += 1
                nodes_old_ids = map(int, items[1:])
                element_nodes = [self.nodes[id] for id in nodes_old_ids]
                self.elements.append(Element(n, element_nodes))
                # print(new_element)
            line = next(self.file)

    def create_edges(self):
        for element in self.elements:
            for i in range(-1, len(element.nodes) - 1, 1):
                self.add_edge_info(element.nodes[i], element.nodes[i + 1], element)

    def add_edge_info(self, node1, node2, element):
        min_node, max_node = (node1, node2) if node1.ID < node2.ID else (node2, node1)
        ID = (min_node.ID, max_node.ID)
        if ID in self.edges:
            self.edges[ID].elements.append(element)
        else:
            self.edges[ID] = Edge(min_node, max_node, element)

    def leave_only_border_edges(self):
        self.edges = {ID: edge for ID, edge in self.edges.items() if edge.is_border()}


#0 - внутренняя поверхность
    #1 - поверхность слева-сверху
    #2 - внешняя поверхность
    #3 - поверхность справа-снизу
def create_curves(edges):
    curves = {i: Curve(i) for i in range(4)}

    for edge in edges.values():
        if edge.is_border():
            c = edge.get_centre()
            if abs(c[0]) < 0.0001:
                curves[1].add(edge)
            elif abs(c[1]) < 0.0001:
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
    curves = create_curves(mesh.edges)
    mesh.add_curves(curves)
    return mesh
