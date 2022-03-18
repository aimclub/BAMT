
from matplotlib.pyplot import axes, axis
import networkx as nx


# try:
#     G = nx.DiGraph([(0, 1), (0, 2), (1, 2)])
#     nx.find_cycle(G, orientation="original")
# except nx.exception.NetworkXNoCycle as error:
#     print(error)
#     print('Нет цикла')


G = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2,3)])

def cycle_check(graph, edge, delete=False):
    new_graph=graph.copy()
    if delete:
        new_graph.remove_edge(*edge)
    else:
        new_graph.add_edge(*edge)
    try:
        nx.find_cycle(new_graph, orientation="original")
    except nx.exception.NetworkXNoCycle:
        return(new_graph.edges)
    else:
        return(graph.edges)


edge=(3,0)
result=cycle_check(G, edge)
print(result)

