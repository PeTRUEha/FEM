


# Граничные условия (давление):
P1 = lambda x, y: -1  # внутри
P2 = lambda x, y: 2  # снаружи

"""Cписок функций формы для равностороннего треугольника с высотой один и координатами
- высотами до первой и второй стороны"""
Ntr = [lambda ksi, eta: ksi,
       lambda ksi, eta: eta,
       lambda ksi, eta: 1 - ksi - eta]

dNtr = [[1, 0],
        [0, 1],
        [-1, -1]]

RSplit = 1.5  # радиус для отличения внешней и внутренней границы
F = 0  # массовая сила
N_PROCESSES = 4
LAMBDA = 100
MU = 80
FILENAME = 'meshes\\donut4.inp'

