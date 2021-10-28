import time

start = time.time()

from Preprocessors import Preprocessor
import pandas as pd
from sklearn import preprocessing as pp
import Builders
from pgmpy.estimators import K2Score

p1 = time.time()
print(f"Time elapsed for importing: {p1 - start}")

vk_data = pd.read_csv("../Data/vk_data.csv")
ROWS = 50
vk_data = vk_data.iloc[:ROWS, :]

p2 = time.time()
print(f"Time elapsed for uploading data: {p2 - p1}")

encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

p = Preprocessor([('encoder', encoder), ('discretizer', discretizer)])

nodes_type_mixed = p.get_nodes_types(vk_data)

columns = [col for col in vk_data.columns.to_list() if (nodes_type_mixed[col] == 'disc') or (nodes_type_mixed[col] == 'disc_num')]
discrete_data = vk_data[columns]

discretized_data, est = p.apply(discrete_data)
info = p.get_info()

# columns = [col for col in discretized_data.columns.to_list() if (info['types'][col] == 'disc') or (info['types'][col] == 'disc_num')]

# discretized_data = discretized_data[columns]

info = {"types": p.get_nodes_types(discretized_data),
        "signs": info['signs']}
print(discretized_data.columns)
print(info['types'])

# Под капот!
bn = Builders.HCStructureBuilder(data=discretized_data, scoring_function=('K2', K2Score),
                                 descriptor=info)
# Вместо: bn = D BN(); dn.add_nodes(data); bn.build_edges(data)

params = {'init_nodes': None,
          'bl_add': None,
          'cont_disc': None}

bn.build(data=discretized_data, **params)

print(bn.skeleton)


# modules = [data[X] for X in columns_to_disc]
# modules_d = [discrete[X] for X in columns_to_disc]
# nodes_types = [get_nodes_type(data) for data in modules]

# bn = structure_learning(discrete, 'HC', node_types, 'K2')
# param = parameter_learning(data, node_types, bn, 'simple')
#
# save_structure(bn, 'full_net')
# skel = read_structure('full_net')
# save_params(param, 'full_net_param')
# params = read_params('full_net_param')
# full = HyBayesianNetwork(skel, params)

# n = 1
# hybns = []
# learning_loop_start = time.time()
# for module, module_d, node_type in zip(modules, modules_d, nodes_types):
#     bn = structure_learning(module_d, 'HC', node_types, 'K2', max_iter=1e5)
#     param = parameter_learning(module, node_types, bn, 'simple')
#     l1 = time.time()
#     print(f"STAGE {n}: elapsed {l1 - learning_loop_start}")
#
#
#     save_structure(bn, 'full_net')
#     skel = read_structure('full_net')
#     save_params(param, 'full_net_param')
#     params = read_params('full_net_param')
#     full = HyBayesianNetwork(skel, params)
#
#     draw_BN(bn, node_type, 'full_net' + str(n))
#     n += 1
#
#     hybns.append(bn)
# print('FOR LOOP FINISHED')

