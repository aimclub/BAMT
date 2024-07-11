
import pandas as pd
from math import log10, floor
from statistics import mean


def data_tranformation(data):
    if 'Unnamed: 0' in data.columns:
        data.drop(['Unnamed: 0'], axis=1, inplace=True)
    if 'class' in data.columns:
        data.drop(['class'], axis=1, inplace=True)    
    
    for i in data.columns:
        if data[i].dtype == 'int64':
            data[i] = data[i].astype(float)

    return data

def group(structure):
    grouped_data = {}

    for tuple_value in structure:
        last_value = tuple_value[-1]
        if last_value in grouped_data:
            grouped_data[last_value].append(tuple_value)
        else:
            grouped_data[last_value] = [tuple_value]
    
    return(grouped_data)

files = ['Iris','Yeast', 'QSAR fish toxicity', 'Liver', 'Glass', 'Ecoli', 'Balance', 'Breast_cancer', 'Parkinsons', 'Vowel', 'QSAR Aquatic', 'Wine','Block', 'Breast_tissue', 'CPU', 'Ionosphere', 'Sonar', 'Vehicle', 'Wdbc', 'Wpbc'] # ['Iris','Yeast', 'QSAR fish toxicity', 'Liver', 'Glass', 'Ecoli', 'Balance', 'Breast_cancer', 'Parkinsons', 'Vowel', 'QSAR Aquatic', 'Wine']
rule_parent_count = {}
pybnesian_parent_count = {}
composite_parent_count = {}
composite_restr_parent_count = {}

for file in files:
    data_train_composite = pd.read_csv('examples/data/pybnesian_MoTBFs/' + file + '/' + file + '_train_fold_' + str(1) + '.csv') 
    N = len(data_train_composite)
    # data_train_composite = data_tranformation(data_train_composite)
    # vertices = list(data_train_composite.columns)
    rule_parent_count[file] = floor(log10(N))
    len_list_pybnesian = []
    len_list_comp = []
    len_list_comp_restr = []
    for number in range(0,10):

        structure_init = pd.read_csv('examples/data/pybnesian_MoTBFs/' + file + '/' + file + '_edges_fold_' + str(number) + '.csv')
        structure = list(zip(structure_init['edges'][0::2], structure_init['edges'][1::2]))
        lengths = [len(value) for key, value in group(structure).items()]
        max_length = max(lengths) if lengths else 0 
        len_list_pybnesian.append(max_length)


        textfile = open('examples/results/results_for_pybnesian_MoTBFs/' + 'CBN_for_pybnesian_MoTBFs_without_structure_limit_' + file + '_run_' + str(number) + ".txt", "r")
        rows = textfile.read().split("\n")[:-1]
        structure  = eval(rows[0].split(' = ')[1])
        lengths = [len(value) for key, value in group(structure).items()]
        max_length = max(lengths) if lengths else 0 
        len_list_comp.append(max_length)


        textfile = open('examples/results/results_for_pybnesian_MoTBFs/changes_in_mutation_90/' + 'CBN_for_pybnesian_MoTBFs_without_structure_limit_add_rule_' + file + '_run_' + str(number) + ".txt", "r")
        rows = textfile.read().split("\n")[:-1]
        structure  = eval(rows[0].split(' = ')[1])
        lengths = [len(value) for key, value in group(structure).items()]
        max_length = max(lengths) if lengths else 0 
        len_list_comp_restr.append(max_length)

    pybnesian_parent_count[file] = mean(len_list_pybnesian)
    composite_parent_count[file] = mean(len_list_comp)
    composite_restr_parent_count[file] = mean(len_list_comp_restr)


print(rule_parent_count)
print(pybnesian_parent_count)
print(composite_parent_count)
print(composite_restr_parent_count)