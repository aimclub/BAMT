import networkx as nx
from pyvis.network import Network
from matplotlib.colors import CSS4_COLORS, TABLEAU_COLORS
from matplotlib.patches import Rectangle
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from block_learning.sampling import get_probability


def draw_BN(bn1: dict, node_type: dict, name: str):
   
    """Function for drawing the graph of BN

    Args:
        bn1 (dict): input BN structure as dict
        node_type (dict): dictionary with node types (descrete or continuous)
        name (str): name of output html page
    """
    
    G = nx.DiGraph()
    G.add_nodes_from(bn1['V'])
    G.add_edges_from(bn1['E'])
    nodes = list(G.nodes)
    
    network = Network(height="800px", width="100%", notebook=True, directed=nx.is_directed(G), layout='hierarchical')

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
                           'indianred', 'mediumslateblue', 'coral','darkseagreen',
                           'mediumseagreen', 'darkslategrey', 'pink', 'darkgoldenrod',
                           'lightgoldenrodyellow', 'magenta', 'indigo','lightcoral', 
                           'lightslategrey', 'honeydew', 'maroon',], start=10):

        color2hex[f'C{i}'] = CSS4_COLORS[color]
    
    color_types = ['По типу']
    color_type='По типу'
    
    if color_types.index(color_type)==0:
        classes_for_legend = []
        for class_item in G.nodes:
            classes_for_legend.append(node_type[class_item])
        
    classes2color = {node_class: color2hex[f'C{i}'] for i, node_class in enumerate(classes_for_legend)}
    classes_for_legend_short = {node_class for i, node_class in enumerate(classes_for_legend)}

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
    
    for geotag, color in classes2color.items():
        handles.append(Rectangle([0, 0], 1, 0.5, color=color))
        labels.append(geotag)
                
    plt.figure(figsize=(13.5, 1.5), dpi=150)
    plt.legend(handles, labels, loc='center', ncol=5)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()
    
    network.show_buttons(filter_=['physics'])
    return network.show(f'visualization_result/'+ name + '.html')





def grouped_barplot(df: pd.DataFrame, cat: str, subcat: str, val: str, err: str):
    """Helper function for drawing hists with error bar

        Args:
            df (pd.DataFrame): source dataset
            cat (str): name of parameter
            subcat (str): name of column with data
            val (str): name of column with probability
            err (str): name of columns with error
        """        
    u = df[cat].unique()
    x = np.arange(len(u))
    subx = df[subcat].unique()
    offsets = (np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.)
    width= np.diff(offsets).mean()
    for i, gr in enumerate(subx):
        dfg = df[df[subcat] == gr]
        plt.bar(x+offsets[i], dfg[val].values, width=width, 
                label="{}".format(gr), yerr=dfg[err].values)
    plt.xlabel(cat)
    plt.ylabel(val)
    if isinstance(u[0], float):
        u = [round(_, 1) for _ in u]
            
    plt.xticks(x, u, rotation=90)
    plt.legend()
    plt.show()
        



def draw_comparative_hist(parameter: str, original_data: pd.DataFrame, sample: pd.DataFrame):
    """Function for drawing comparative hist for discrete synthetic and real data

    Args:
        parameter (str): name of parameter
        original_data (pd.DataFrame): real dataset
        sample (pd.DataFrame): synthetic dataset
    """    
    df1 = pd.DataFrame()
    probs = get_probability(original_data, original_data, parameter)
            
    df1[parameter] = probs.keys()
        
    
        
    df1['Probability'] = [p[1] for p in probs.values()]
    df1['Error'] = [p[2]-p[1] for p in probs.values()]
    df1['Data'] = 'Исходные данные'

        
    df2 = pd.DataFrame()
    probs = get_probability(sample, original_data, parameter)
    df2[parameter] =probs.keys()
    df2['Probability']  = [p[1] for p in probs.values()]
    df2['Error'] = [p[2]-p[1] for p in probs.values()]
    df2['Data'] = 'Синтетические данные'


        
    final_df = pd.concat([df1, df2])
    
        
    
    grouped_barplot(final_df, parameter, 'Data', 'Probability', 'Error')
    plt.show()
        