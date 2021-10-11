import json
import os
from external.libpgm.graphskeleton import GraphSkeleton
from external.libpgm.nodedata import NodeData
from core.core_utils import project_root


def save_structure(bn: dict, name: str = 'BN_structure'):
    """Function for saving bn_structure as a json dictionary in txt file.

    Args:
        bn (dict): dictionary with structure 
        name (str, optional): Name of file. Defaults to None.
    """

    structure_location = r'models/structure_bn'
    structure_bn_path = os.path.join(project_root(), structure_location)
    if not os.path.exists(structure_bn_path):
        os.makedirs(structure_bn_path)
    json.dump(bn, open(f'{structure_bn_path}/{name}.txt', 'w'))


def save_params(bn_param: dict, name: str = 'BN_params'):
    """Function for saving bn_parameters as a json dictionary in txt file

    Args:
        bn_param (dict): dictionary with parameters structure.
        name (str, optional): Name of file. Defaults to None.
    """
    params_location = r'models/parameter_bn'
    params_bn_path = os.path.join(project_root(), params_location)
    if not os.path.exists(params_bn_path):
        os.makedirs(params_bn_path)

    json.dump(bn_param, open(f'{params_bn_path}/{name}.txt', 'w'))


def read_structure(name: str) -> GraphSkeleton:
    """Function for reading json structure of BN 

    Args:
        name (str): Name of file with structure

    Returns:
        GraphSkeleton: object of BN structure
    """
    skel = GraphSkeleton()
    skel_loc = os.path.join(project_root(), r'models/structure_bn', f"{name}.txt")
    skel.load(skel_loc)
    skel.toporder()
    return skel


def read_params(name: str) -> NodeData:
    """Function for reading parameters of BN

    Args:
        name (str): Name of file with parameters

    Returns:
        NodeData: object of BN parameters
    """
    nd = NodeData()
    params_location = r'models/parameter_bn'
    params_bn_path = os.path.join(project_root(), params_location, f"{name}.txt")
    nd.load(params_bn_path)
    nd.entriestoinstances()
    return nd
