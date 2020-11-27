from libpgm.nodedata import NodeData
from libpgm.graphskeleton import GraphSkeleton
import json


def save_structure(bn: dict, name: str = None):
    """Function for saving bn_structure as a json dictionary in txt file.

    Args:
        bn (dict): dictionary with structure 
        name (str, optional): Name of file. Defaults to None.
    """   
    if name == None:
        name = 'BN_structure'
    json.dump(bn, open("models/structure/"+name+".txt",'w'))
    

def save_params(bn_param: dict, name: str = None):
    """Function for saving bn_parameters as a json dictionary in txt file

    Args:
        bn_param (dict): dictionary with parameters structure.
        name (str, optional): Name of file. Defaults to None.
    """    
    if name == None:
        name = 'BN_params'
    json.dump(bn_param, open("models/parameters/"+name+".txt",'w'))
    

    

    

    


    