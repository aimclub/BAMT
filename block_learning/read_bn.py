from libpgm.nodedata import NodeData
from libpgm.graphskeleton import GraphSkeleton
import json



def read_structure(name: str) -> GraphSkeleton:
    """Function for reading json structure of BN 

    Args:
        name (str): Name of file with structure

    Returns:
        GraphSkeleton: object of BN structure
    """    
    skel = GraphSkeleton()
    skel.load("models/structure/"+name+".txt")
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
    nd.load("models/parameters/"+name+".txt")
    nd.entriestoinstances()
    return nd