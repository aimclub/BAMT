import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)


from copy import copy
import math
from typing import List
import numpy as np
import pandas as pd
from external.pyBN.utils.independence_tests import mutual_information, entropy
from preprocess.discretization import get_nodes_type
from preprocess.numpy_pandas import loc_to_DataFrame
from preprocess.graph import edges_to_dict

def child_dict(net: list):
    res_dict = dict()
    for e0, e1 in net:
        if e1 in res_dict:
            res_dict[e1].append(e0)
        else:
            res_dict[e1] = [e0]
    return res_dict

def precision_recall(pred_net: list, true_net: list, decimal = 2):
    pred_dict = child_dict(pred_net)
    true_dict = child_dict(true_net)
    corr_undir = 0
    corr_dir = 0
    for e0, e1 in pred_net:
        flag = True
        if e1 in true_dict:
            if e0 in true_dict[e1]:
                corr_undir += 1
                corr_dir += 1
                flag = False
        if (e0 in true_dict) and flag:
            if e1 in true_dict[e0]:
                corr_undir += 1
    pred_len = len(pred_net)
    true_len = len(true_net)
    shd = pred_len + true_len - corr_undir - corr_dir
    return {'AP': round(corr_undir/pred_len, decimal), 'AR': round(corr_undir/true_len, decimal), 'AHP': round(corr_dir/pred_len, decimal), 'AHR': round(corr_dir/true_len, decimal), 'SHD': shd}

