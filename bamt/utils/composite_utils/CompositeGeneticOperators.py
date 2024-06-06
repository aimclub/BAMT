from math import log10
from random import choice

import pandas as pd
from golem.core.dag.graph_utils import ordered_subnodes_hierarchy
from numpy import std, mean, log
from scipy.stats import norm
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from .CompositeModel import CompositeModel
from .MLUtils import MlModels


def custom_crossover_all_model(
    graph_first: CompositeModel, graph_second: CompositeModel, max_depth
):
    num_cros = 100
    try:
        for _ in range(num_cros):
            selected_node1 = choice(graph_first.nodes)
            if selected_node1.nodes_from is None or selected_node1.nodes_from == []:
                continue

            selected_node2 = graph_second.get_nodes_by_name(str(selected_node1))[0]
            if selected_node2.nodes_from is None or selected_node2.nodes_from == []:
                continue

            model1 = selected_node1.content["parent_model"]
            model2 = selected_node2.content["parent_model"]

            selected_node1.content["parent_model"] = model2
            selected_node2.content["parent_model"] = model1

            break

    except Exception as ex:
        print(ex)
    return graph_first, graph_second


def custom_mutation_add_structure(graph: CompositeModel, **kwargs):
    num_mut = 100
    try:
        for _ in range(num_mut):
            rid = choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[choice(range(len(graph.nodes)))]
            nodes_not_cycling = random_node.descriptive_id not in [
                n.descriptive_id for n in ordered_subnodes_hierarchy(other_random_node)
            ] and other_random_node.descriptive_id not in [
                n.descriptive_id for n in ordered_subnodes_hierarchy(random_node)
            ]
            if nodes_not_cycling:
                other_random_node.nodes_from.append(random_node)
                ml_models = MlModels()
                other_random_node.content[
                    "parent_model"
                ] = ml_models.get_model_by_children_type(other_random_node)
                break

    except Exception as ex:
        graph.log.warn(f"Incorrect connection: {ex}")
    return graph


def custom_mutation_delete_structure(graph: CompositeModel, **kwargs):
    num_mut = 100
    try:
        for _ in range(num_mut):
            rid = choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[choice(range(len(graph.nodes)))]
            if (
                random_node.nodes_from is not None
                and other_random_node in random_node.nodes_from
            ):
                random_node.nodes_from.remove(other_random_node)
                if not random_node.nodes_from:
                    random_node.content["parent_model"] = None
                break
    except Exception as ex:
        print(ex)
    return graph


def custom_mutation_reverse_structure(graph: CompositeModel, **kwargs):
    num_mut = 100
    try:
        for _ in range(num_mut):
            rid = choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[choice(range(len(graph.nodes)))]
            if (
                random_node.nodes_from is not None
                and other_random_node in random_node.nodes_from
            ):
                random_node.nodes_from.remove(other_random_node)
                if not random_node.nodes_from:
                    random_node.content["parent_model"] = None
                other_random_node.nodes_from.append(random_node)
                ml_models = MlModels()
                other_random_node.content[
                    "parent_model"
                ] = ml_models.get_model_by_children_type(other_random_node)
                break
    except Exception as ex:
        print(ex)
    return graph


def custom_mutation_add_model(graph: CompositeModel, **kwargs):
    try:
        all_nodes = graph.nodes
        nodes_with_parents = [
            node
            for node in all_nodes
            if (node.nodes_from != [] and node.nodes_from is not None)
        ]
        if not nodes_with_parents:
            return graph
        node = choice(nodes_with_parents)
        ml_models = MlModels()
        node.content["parent_model"] = ml_models.get_model_by_children_type(node)
    except Exception as ex:
        print(ex)
    return graph


def composite_metric(graph: CompositeModel, data: pd.DataFrame, percent=0.02):
    data_all = data
    data_train, data_test = train_test_split(data_all, train_size=0.8, random_state=42)
    score, len_data = 0, len(data_train)
    for node in graph.nodes:
        data_of_node_train = data_train[node.content["name"]]
        data_of_node_test = data_test[node.content["name"]]
        if node.nodes_from is None or node.nodes_from == []:
            if node.content["type"] == "cont":
                mu, sigma = mean(data_of_node_train), std(data_of_node_train)
                score += norm.logpdf(
                    data_of_node_test.values, loc=mu, scale=sigma
                ).sum()
            else:
                count = data_of_node_train.value_counts()
                frequency = log(count / len_data)
                index = frequency.index.tolist()
                for value in data_of_node_test:
                    if value in index:
                        score += frequency[value]
        else:
            model, columns, target, idx = (
                MlModels().dict_models[node.content["parent_model"]](),
                [n.content["name"] for n in node.nodes_from],
                data_of_node_train.to_numpy(),
                data_train.index.to_numpy(),
            )
            setattr(model, "max_iter", 100000)
            features = data_train[columns].to_numpy()
            if len(set(target)) == 1:
                continue
            fitted_model = model.fit(features, target)

            features = data_test[columns].to_numpy()
            target = data_of_node_test.to_numpy()
            if node.content["type"] == "cont":
                predict = fitted_model.predict(features)
                rmse = root_mean_squared_error(target, predict) + 0.0000001
                a = norm.logpdf(target, loc=predict, scale=rmse)
                score += a.sum()
            else:
                predict_proba = fitted_model.predict_proba(features)
                idx = pd.array(list(range(len(target))))
                li = []

                for i in idx:
                    a = predict_proba[i]
                    try:
                        b = a[target[i]]
                    except BaseException:
                        b = 0.0000001
                    if b < 0.0000001:
                        b = 0.0000001
                    li.append(log(b))
                score += sum(li)

    edges_count = len(graph.get_edges())
    score -= (edges_count * percent) * log10(len_data) * edges_count

    return -score
