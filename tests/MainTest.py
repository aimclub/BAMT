import pandas as pd
from pgmpy.estimators import K2Score
from sklearn import preprocessing as pp

import bamt.networks as Networks
from bamt.preprocessors import Preprocessor

"""
Optional:
You can also uncomment print() that you need.
"""

hack_data = pd.read_csv("../data/real data/hack_processed_with_rf.csv")
cont_data = hack_data[["Gross", "Netpay", "Porosity", "Permeability", "Depth"]].dropna()
disc_data = hack_data[
    ["Tectonic regime", "Period", "Lithology", "Structural setting"]
].dropna()
hybrid_data = hack_data[
    [
        "Tectonic regime",
        "Period",
        "Lithology",
        "Structural setting",
        "Gross",
        "Netpay",
        "Porosity",
        "Permeability",
        "Depth",
    ]
].dropna()

cont_test_data = cont_data[cont_data.columns[:-1]]
cont_target = cont_data[cont_data.columns[-1]]
disc_test_data = disc_data[disc_data.columns[:-1]]
disc_target = disc_data[disc_data.columns[-1]]
hybrid_test_data = hybrid_data[hybrid_data.columns[:-1]]
hybrid_target = hybrid_data[hybrid_data.columns[-1]]

encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")
p = Preprocessor([("encoder", encoder), ("discretizer", discretizer)])

# Discrete pipeline
discretized_data, _ = p.apply(disc_data)
disc_bn = Networks.DiscreteBN()
info = p.info
disc_bn.add_nodes(info)
disc_bn.add_edges(data=discretized_data, scoring_function=("K2", K2Score))
disc_bn.fit_parameters(data=disc_data)
disc_bn.calculate_weights(discretized_data)
disc_predicted_values = disc_bn.predict(test=disc_test_data)
disc_predicted_values = pd.DataFrame.from_dict(disc_predicted_values, orient="columns")
synth_disc_data = disc_bn.sample(50)

disc_bn.save("./disc_bn.json")
disc_bn2 = Networks.DiscreteBN()
disc_bn2.load("./disc_bn.json")
synth_disc_data2 = disc_bn2.sample(50)
# print(disc_bn.weights)
# print(disc_bn2.weights)
# print(disc_bn.get_info())
# print(disc_bn2.get_info())
# print(synth_disc_data)
# print(synth_disc_data2)

# Continuous pipeline
discretized_data, _ = p.apply(cont_data)
cont_bn = Networks.ContinuousBN(use_mixture=True)
info = p.info
cont_bn.add_nodes(info)
cont_bn.add_edges(data=discretized_data, scoring_function=("K2", K2Score))
cont_bn.fit_parameters(data=cont_data)
cont_bn.calculate_weights(discretized_data)
cont_predicted_values = cont_bn.predict(test=cont_test_data)
cont_predicted_values = pd.DataFrame.from_dict(cont_predicted_values, orient="columns")
synth_cont_data = cont_bn.sample(50)

cont_bn.save("./cont_bn.json")
cont_bn2 = Networks.ContinuousBN(use_mixture=True)
cont_bn2.load("./cont_bn.json")
synth_cont_data2 = cont_bn2.sample(50)
# print(cont_bn.weights)
# print(cont_bn2.weights)
# print('RMSE on predicted values with continuous data: ' +
#       f'{mse(cont_target, cont_predicted_values, squared=False)}')
# print(cont_bn.get_info())
# print(cont_bn2.get_info())
# print(synth_cont_data)
# print(synth_cont_data2)

# Hybrid pipeline
discretized_data, _ = p.apply(hybrid_data)
hybrid_bn = Networks.HybridBN(use_mixture=True)
hybrid_bn2 = Networks.HybridBN(use_mixture=True, has_logit=True)
info = p.info
hybrid_bn.add_nodes(info)
hybrid_bn2.add_nodes(info)
hybrid_bn.add_edges(data=discretized_data, scoring_function=("K2", K2Score))
hybrid_bn2.add_edges(data=discretized_data, scoring_function=("K2", K2Score))
hybrid_bn.fit_parameters(data=hybrid_data)
hybrid_bn2.fit_parameters(data=hybrid_data)
hybrid_bn.calculate_weights(discretized_data)
hybrid_bn2.calculate_weights(discretized_data)
hybrid_predicted_values = hybrid_bn.predict(test=hybrid_test_data)
hybrid_predicted_values = pd.DataFrame.from_dict(
    hybrid_predicted_values, orient="columns"
)
synth_hybrid_data = hybrid_bn.sample(50)
synth_hybrid_data2 = hybrid_bn2.sample(50)

hybrid_bn.save("./hybrid_bn.json")
hybrid_bn3 = Networks.HybridBN(use_mixture=True)
hybrid_bn3.load("./hybrid_bn.json")
synth_hybrid_data3 = hybrid_bn3.sample(50)
# print(hybrid_bn.weights)
# print(hybrid_bn2.weights)
# print(hybrid_bn3.weights)
# print('RMSE on predicted values with hybrid data: ' +
#       f'{mse(hybrid_target, hybrid_predicted_values, squared=False)}')
# print(hybrid_bn.get_info())
# print(hybrid_bn2.get_info())
# print(hybrid_bn3.get_info())
# print(synth_hybrid_data)
# print(synth_hybrid_data2)
# print(synth_hybrid_data3)

# Save and load BN without weights
discretized_data, _ = p.apply(hybrid_data)
hybrid_bn = Networks.HybridBN(use_mixture=True)
info = p.info
hybrid_bn.add_nodes(info)
hybrid_bn.add_edges(data=discretized_data, scoring_function=("K2", K2Score))
hybrid_bn.fit_parameters(data=hybrid_data)
hybrid_bn.save("./hybrid_bn_without_weights.json")
hybrid_bn2 = Networks.HybridBN(use_mixture=True)
hybrid_bn2.load("./hybrid_bn_without_weights.json")
# print(hybrid_bn2.weights)
