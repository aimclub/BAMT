from libpgm.nodedata import NodeData
from libpgm.graphskeleton import GraphSkeleton
import json



def read_structure(name: str) -> GraphSkeleton:
    skel = GraphSkeleton()
    skel.load("models/structure/"+name+".txt")
    skel.toporder()
    return skel
    

def read_params(name: str) -> NodeData:
    nd = NodeData()
    nd.load("models/parameters/"+name+".txt")
    nd.entriestoinstances()
    return nd