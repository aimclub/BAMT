from pomegranate import BayesianNetwork
import json

"""
Function for saving BN model to json file


Input:
-bn
BayesianNetwork object

-name
Name of the model

Output:
Saving trained BN in a json file


"""

def save_model(bn: BayesianNetwork, name: str = None):
    
    if name == None:
        name = 'BN'
    with open('models/'+name+'.json', 'w+') as f:
        json.dump(bn.to_json(), f)
    

    

    


    