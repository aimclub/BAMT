from pomegranate import BayesianNetwork
import json

"""
Function for reading a BN model
from json file


Input:
-name
The name of json file with 
BN model

Output:
BayesianNetwork object
"""



def read_model(name: str) -> BayesianNetwork:
    
    string_data = ""
    with open('models/'+ name +'.json') as f:
        string_data = json.load(f)
    bn = BayesianNetwork.from_json(string_data)
    return(bn)
    