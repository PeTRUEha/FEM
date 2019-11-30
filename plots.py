import math

import numpy as np
from matplotlib import pyplot as plt
from numpy import linspace

from mesh import probe_location_from_nodes, probe_location_from_element


def plot_over_line(line=linspace(0, 2, 50)):
    values = [probe_location_from_nodes('displacement', x, 0) for x in line]
    eps = [probe_location_from_element('strain', x, 0) for x in line]
    sigma = [probe_location_from_element('stress', x, 0) for x in line]

    x_values = [value[0] if not type(value) == float else math.nan for value in values]
    y_values = [value[1] if not type(value) == float else math.nan for value in values]
    eps11_values = [value[0] if not type(value) == float else math.nan for value in eps]
    sigma11_values = [value[0] if not type(value) == float else math.nan for value in sigma]
    sigma22_values = [value[1] if not type(value) == float else math.nan for value in sigma]

    # plot_graph(line, x_values, 'displacement x')
    # plot_graph(line, y_values, 'displacement y')
    # plot_graph(line, eps11_values, 'eps11')
    plot_graph(line, sigma11_values, 'sigma11')
    # plot_graph(line, sigma22_values, 'sigma22')


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
    N = array.shape[0]
    c_y = stiffness.pcolor(range(2 * N), range(2 * N), array, cmap='plasma', vmin=np.nanmin(array),
                           vmax=np.nanmax(array))
    plt.colorbar(c_y)
    plt.show()