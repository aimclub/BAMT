import json
import os
from external.libpgm.graphskeleton import GraphSkeleton
from external.libpgm.nodedata import NodeData
from core.core_utils import project_root


def save_structure(bn: dict, name: str = None):
    """Function for saving bn_structure as a json dictionary in txt file.

    Args:
        bn (dict): dictionary with structure 
        name (str, optional): Name of file. Defaults to None.
    """
    if name == None:
        name = 'BN_structure'

    structure_bn_path = f'{project_root()}/models/structure_bn'
    if not os.path.exists(structure_bn_path):
        os.mkdir(structure_bn_path)
    json.dump(bn, open(f'{structure_bn_path}/{name}.txt', 'w'))


def save_params(bn_param: dict, name: str = None):
    """Function for saving bn_parameters as a json dictionary in txt file

    Args:
        bn_param (dict): dictionary with parameters structure.
        name (str, optional): Name of file. Defaults to None.
    """
    if name == None:
        name = 'BN_params'
    params_bn_path = f'{project_root()}/models/parameter_bn'
    if not os.path.exists(params_bn_path):
        os.mkdir(params_bn_path)

    json.dump(bn_param, open(f'{params_bn_path}/{name}.txt', 'w'))


def read_structure(name: str) -> GraphSkeleton:
    """Function for reading json structure of BN 

    Args:
        name (str): Name of file with structure

    Returns:
        GraphSkeleton: object of BN structure
    """
    skel = GraphSkeleton()
    skel.load(f'{project_root()}/models/structure_bn/{name}.txt')
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
    nd.load(f'{project_root()}/models/parameter_bn/{name}.txt')
    nd.entriestoinstances()
    return nd
