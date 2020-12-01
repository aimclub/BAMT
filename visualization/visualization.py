import networkx as nx
from pyvis.network import Network
from matplotlib.colors import CSS4_COLORS, TABLEAU_COLORS
from matplotlib.patches import Rectangle
from typing import Dict, List
import matplotlib.pyplot as plt

"""
>>visualizer
Function for visualizing bayesian network

Input:

Skelet - vertices and edges
bn1={'V': ['age',
  'sex',
  'has_high_education',
  'relation_status',
  'number_of_relatives'],
 'E': [['number_of_relatives', 'sex'],
  ['age', 'has_high_education'],
  ['sex', 'has_high_education'],
  ['age', 'relation_status'],
  ['sex', 'relation_status']]}

Node_type - types of vertices
node_type2={'age': 'cont',
 'sex': 'disc',
 'has_high_education': 'disc',
 'relation_status': 'disc',
 'number_of_relatives': 'disc'}
 

Output:
-graph

"""


def visualizer(bn1,node_type):
    
    G = nx.DiGraph()
    G.add_nodes_from(bn1['V'])
    G.add_edges_from(bn1['E'])
    nodes = list(G.nodes)
    
    network = Network(height="400px", width="90%", notebook=True, directed=nx.is_directed(G), layout='hierarchical')

    added_nodes_levels  = dict()

    for node in nodes:
        level = 0
        if len(G.in_edges(node)) == 0:
            level = 0
        if len(G.in_edges(node)) > 0:
            level = 1
        if len(G.out_edges(node)) == 0:
            level = 2
        added_nodes_levels[node] = level
    
    is_ordered = False
    
    while not is_ordered:
        is_ordered = True
        for node in nodes:
            cur_level = added_nodes_levels[node]
            if any([cur_level <= added_nodes_levels.get(parent, -1) for parent in G.predecessors(node)]):
                added_nodes_levels[node] = cur_level + 1
                is_ordered = False
                
    color2hex = list(TABLEAU_COLORS.values())
    color2hex = {f'C{i}': value for i, value in enumerate(color2hex)}

    for i, color in enumerate(['blue', 'gold', 'lime', 'red', 'magenta', 'peru',
                           'dodgerblue', 'orangered', 'mediumspringgreen',
                           'indianred', 'mediumslateblue', 'coral'], start=10):

        color2hex[f'C{i}'] = CSS4_COLORS[color]
    
    color_types = ['По типу']
    color_type='По типу'
    
    classes1=G.nodes
    
    classes_for_legend = classes1
    classes_for_legend_short = classes1
        
    if color_types.index(color_type)==0:
        classes_for_legend = []
        for class_item in classes1:
            classes_for_legend.append(node_type[class_item])
        classes_for_legend_short =  ['1 - непрерывные', '2 - дискретные']
        
    classes2color = {node_class: color2hex[f'C{i}'] for i, node_class in enumerate(classes_for_legend)}

    for node in nodes:
        node_class = node
        level = added_nodes_levels[node]
        network.add_node(node, label=node, 
                         color=str(classes2color[node_type[node]]), 
                         size=45, level = level,
                         font={'size': 36},
                         title=f'Узел байесовской сети {node}')
        added_nodes_levels[node] = level
    
    for edge in G.edges:
        network.add_edge(edge[0], edge[1])
    

    
    network.hrepulsion(node_distance=300, central_gravity = 0.5)
    
    handles = []
    labels = []
   

    for geotag, color in zip(classes_for_legend_short, classes2color.values()):
        handles.append(Rectangle([0, 0], 1, 0.5, color=color))
        labels.append(geotag)
                  
    plt.figure(figsize=(13.5, 1.5), dpi=150)
    plt.legend(handles, labels, loc='center', ncol=5)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()
    
    return network.show(f'bayesian_network.html')