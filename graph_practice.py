# import networkx as nx
# import matplotlib.pyplot as plt
#
# graph = nx.Graph()
# graph.add_edge(4, 2)
# graph.add_edge(1, 2)
# nx.draw(graph, labels = {4:4, 2:2, 1:1}, color={4:4, 2:2, 1:1})
# plt.show()
# # plt.plot([0, 1, 2], [0, 1, 2])
# print(graph.nodes)



# import itertools
#
# print(set(itertools.combinations({2, 3, 4, 6, 7}, 2)))

import threading
import time
import random


# def worker(number):
#     sleep = random.randrange(1, 10)
#     time.sleep(sleep)
#     print("I am Worker {}, I slept for {} seconds".format(number, sleep))
#
#
# for i in range(5):
#     t = threading.Thread(target=worker, args=(i,))
#     t.start()
#
# print("All Threads are queued, let's see when they finish!")
import numpy as np
from scipy.sparse import coo_matrix

row  = np.array([0, 0, 1, 3, 1, 0, 0])
col  = np.array([0, 2, 1, 3, 1, 0, 0])
data = np.array([1, 1, 1, 1, 1, 1, 1])
coo = coo_matrix((data, (row, col)), shape=(4, 4))

lil = coo.tolil()
print(type(lil.T))