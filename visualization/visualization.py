import os.path
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
import numpy as np
from pyvis.network import Network
from matplotlib.colors import CSS4_COLORS, TABLEAU_COLORS
from matplotlib.patches import Rectangle
import random


def draw_comparative_hist(parameter: str, original_data: pd.DataFrame, synthetic_data: pd.DataFrame, node_type: dict):
    """Function for drawing comparative distribution

    Args:
        parameter (str): name of parameter
        original_data (pd.DataFrame): original dataset
        data_without_restore (pd.DataFrame): sample of bn with node
        data_with_restore (pd.DataFrame): sample of bn without node
        node_type (dict): dictionary with types of nodes
    """
    if (node_type[parameter] == 'disc'):
        df1 = pd.DataFrame()
        df1[parameter] = original_data[parameter]
        df1['Data'] = 'Исходные данные'
        df1['Probability'] = df1[parameter].apply(
            lambda x: (df1.groupby(parameter)[parameter].count()[x]) / original_data.shape[0])
        df2 = pd.DataFrame()
        df2[parameter] = synthetic_data[parameter]
        df2['Data'] = 'Синтетические данные'
        df2['Probability'] = df2[parameter].apply(
            lambda x: (df2.groupby(parameter)[parameter].count()[x]) / synthetic_data.shape[0])
        # df3 = pd.DataFrame()
        # df3[parameter] = data_with_restore[parameter]
        # df3['Data'] = 'Данные из сети без изучаемого узла'
        # df3['Probability'] = df3[parameter].apply(
        #     lambda x: (df3.groupby(parameter)[parameter].count()[x]) / data_with_restore.shape[0])
        final_df = pd.concat([df1, df2])
        ax = sns.barplot(x=parameter, y="Probability", hue="Data", data=final_df)
        ax.xaxis.set_tick_params(rotation=45)
    else:
        ax = sns.distplot(original_data[parameter], hist=False, label='Исходные данные')
        ax = sns.distplot(synthetic_data[parameter], hist=False, label='Синтетические данные')
        #ax = sns.distplot(data_with_restore[parameter], hist=False, label='Данные из сети без изучаемого узла')
        ax.xaxis.set_tick_params(rotation=45)
        ax.legend()

    # ax.set_xticks(range(0, original_data[parameter].nunique(), 5))

    plt.show()


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
    color2hex = {f'C{i+2}': value for i, value in enumerate(color2hex)}
    color2hex['C0'] = '#eb3d3d'
    color2hex['C1'] = '#3d46eb'
       

    for i, color in enumerate(['blue', 'red', 'lime', 'gold', 'magenta', 'peru',
                           'dodgerblue', 'orangered', 'mediumspringgreen',
                           'indianred', 'mediumslateblue', 'coral','darkseagreen',
                           'mediumseagreen', 'darkslategrey', 'pink', 'darkgoldenrod',
                           'lightgoldenrodyellow', 'indigo','lightcoral', 
                           'lightslategrey', 'honeydew', 'maroon',], start=12):

        color2hex[f'C{i}'] = CSS4_COLORS[color]
    
    color_types = ['По типу']
    color_type='По типу'
    
    if color_types.index(color_type)==0:
        classes_for_legend = []
        for class_item in G.nodes:
            classes_for_legend.append(node_type[class_item])
    
    classes_for_legend_uniq = set(classes_for_legend)
    number_of_classes=len(classes_for_legend_uniq)
    
    r = lambda: random.randint(0,255)
    if (number_of_classes>35):
        for i in range(35,35+(number_of_classes-34)):
            color2hex[f'C{i}'] = '#%02X%02X%02X' % (r(),r(),r())
          
    classes2color = {node_class: color2hex[f'C{i}'] for i, node_class in enumerate(classes_for_legend_uniq)}

    for node in nodes:
        level = added_nodes_levels[node]
        network.add_node(node, label=node, 
                         color=classes2color[node_type[node]],
                         size=45, level = level,
                         font={'size': 36},
                         title=f'Узел байесовской сети {node}')
        added_nodes_levels[node] = level
    
    for edge in G.edges:
        network.add_edge(edge[0], edge[1])
       
    network.hrepulsion(node_distance=300, central_gravity = 0.5)
    
    
    #handles = []
    #labels = []
    #for geotag, color in classes2color.items():
    #    handles.append(Rectangle([0, 0], 1, 0.5, color=color))
    #    labels.append(geotag)           
    #plt.figure(figsize=(13.5, 1.5), dpi=150)
    #plt.legend(handles, labels, loc='center', ncol=5)
    #plt.axis('off')
    #plt.tight_layout()
    #plt.show()
    #plt.close()
    
    #network.show_buttons(filter_=['physics'])
    if not (os.path.exists('../visualization_result')):
        os.mkdir("../visualization_result")
    
    #G_legend = nx.DiGraph()
    #G_legend.add_nodes_from(classes_for_legend_uniq)
    
    #network_legend = Network(height="800px", width="100%", notebook=True)
    #for node in classes_for_legend_uniq:
    #    network_legend.add_node(node, label=node, 
    #                     color=classes2color[node],
    #                     size=45, level = level,
    #                     font={'size': 36},
    #                     title=f'{node}')
    #network_legend.show(f'../visualization_result/'+ name + '_legend.html')
    
    return network.show(f'../visualization_result/'+ name + '.html')
