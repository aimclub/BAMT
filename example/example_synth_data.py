import time

start = time.time()

from bayesian.sampling import generate_synthetics
from preprocess.discretization import discretization, get_nodes_type, code_categories, get_nodes_sign
from bayesian.train_bn import structure_learning, parameter_learning
import pandas as pd
from bayesian.save_bn import save_structure, save_params, read_structure, read_params
from external.libpgm.hybayesiannetwork import HyBayesianNetwork
from visualization.visualization import draw_BN
from block_learning.partial_bn_train import direct_connect, direct_train

p1 = time.time()
print(f"Time elapsed for importing: {p1 - start}")

vk_data = pd.read_csv("../data/vk_data.csv")

p2 = time.time()
print(f"Time elapsed for uploading data: {p2 - p1}")

node_types = get_nodes_type(vk_data)
nodes_sign = get_nodes_sign(vk_data)

# ROWS = 5000
# vk_data = vk_data.iloc[:ROWS, :]

data = vk_data[
    ['age', 'sex', 'has_high_education', 'relation', 'num_of_relatives', 'about', 'activities', 'books', 'interests',
     'movies', 'top1_interes', 'top2_interes', 'top3_interes', 'top4_interes', 'top5_interes', 'max_tr', 'mean_tr',
     'is_parent', 'is_driver', 'has_pets', 'cash_usage']]

coded_data, coder = code_categories(data, 'label',
                                    ['top1_interes', 'top2_interes', 'top3_interes', 'top4_interes', 'top5_interes'])
discrete, est = discretization(coded_data, 'equal_frequency', bins=5, columns=['age', 'max_tr', 'mean_tr'])

columns_to_disc = [['age', 'sex', 'has_high_education', 'relation', 'num_of_relatives'],
                   ['about', 'activities', 'books', 'interests', 'movies'],
                   ['top1_interes', 'top2_interes', 'top3_interes', 'top4_interes', 'top5_interes'],
                   ['max_tr', 'mean_tr', 'is_parent', 'is_driver', 'has_pets', 'cash_usage']]

modules = [data[X] for X in columns_to_disc]
modules_d = [discrete[X] for X in columns_to_disc]
nodes_types = [get_nodes_type(data) for data in modules]

# bn = structure_learning(discrete, 'HC', node_types, 'K2')
# param = parameter_learning(data, node_types, bn, 'simple')
#
# save_structure(bn, 'full_net')
# skel = read_structure('full_net')
# save_params(param, 'full_net_param')
# params = read_params('full_net_param')
# full = HyBayesianNetwork(skel, params)

n = 1
hybns = []
learning_loop_start = time.time()
for module, module_d, node_type in zip(modules, modules_d, nodes_types):
    bn = structure_learning(module_d, 'HC', node_types, 'K2', max_iter=1e5)
    param = parameter_learning(module, node_types, bn, 'simple')
    l1 = time.time()
    print(f"STAGE {n}: elapsed {l1 - learning_loop_start}")


    save_structure(bn, 'full_net')
    skel = read_structure('full_net')
    save_params(param, 'full_net_param')
    params = read_params('full_net_param')
    full = HyBayesianNetwork(skel, params)

    draw_BN(bn, node_type, 'full_net' + str(n))
    n += 1

    hybns.append(bn)
print('FOR LOOP FINISHED')

hybn_connect = direct_connect(hybns, discrete, node_types)
print('CONNECT FINISHED')

direct_train_start = time.time()
hybn_final = direct_train(hybns, coded_data, hybn_connect)
dt = time.time()
print(f'DIRECT TRAIN FINISHED ({dt - direct_train_start})')

final_struct = dict()
final_struct['V'] = hybn_final.V
final_struct['E'] = hybn_final.E

sample = generate_synthetics(hybn_final, nodes_sign, 'simple')
nd = get_nodes_type(sample)
draw_BN(final_struct, nd, 'final_structure')


# Time elapsed for importing: 5.485136270523071
# Time elapsed for uploading data: 0.5628087520599365
# STAGE 1: elapsed 0.9903216361999512
# STAGE 2: elapsed 1.9904534816741943
# STAGE 3: elapsed 3.5354082584381104
# STAGE 4: elapsed 4.835458040237427
# FOR LOOP FINISHED
#   0%|          | 33/1000000 [00:16<138:40:40,  2.00it/s]
# ['0 1', '0 2', '0 3']
# CONNECT FINISHED
# 0 1
# 0 2
# 0 3
# DIRECT TRAIN FINISHED (477.1661093235016)

