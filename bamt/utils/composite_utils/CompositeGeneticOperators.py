from CompositeModel import CompositeModel
from random import choice
from golem.core.dag.graph_utils import ordered_subnodes_hierarchy
from MLUtils import MlModels


def custom_crossover_all_model(graph_first: CompositeModel, graph_second: CompositeModel, max_depth):
    num_cros = 100
    try:
        for _ in range(num_cros):
            selected_node1=choice(graph_first.nodes)
            if selected_node1.nodes_from is None or selected_node1.nodes_from == []:
                continue

            selected_node2=graph_second.get_nodes_by_name(str(selected_node1))[0]
            if selected_node2.nodes_from is None or selected_node2.nodes_from == []:
                continue

            model1 = selected_node1.content['parent_model']
            model2 = selected_node2.content['parent_model']

            selected_node1.content['parent_model'] = model2
            selected_node2.content['parent_model'] = model1

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
            nodes_not_cycling = (random_node.descriptive_id not in
                                 [n.descriptive_id for n in ordered_subnodes_hierarchy(other_random_node)] and
                                 other_random_node.descriptive_id not in
                                 [n.descriptive_id for n in ordered_subnodes_hierarchy(random_node)])
            if nodes_not_cycling:
                other_random_node.nodes_from.append(random_node)
                ml_models = MlModels()
                other_random_node.content['parent_model'] = ml_models.get_model_by_children_type(other_random_node)
                break

    except Exception as ex:
        graph.log.warn(f'Incorrect connection: {ex}')
    return graph


def custom_mutation_delete_structure(graph: CompositeModel, **kwargs):
    num_mut = 100
    try:
        for _ in range(num_mut):
            rid = choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[choice(range(len(graph.nodes)))]
            if random_node.nodes_from is not None and other_random_node in random_node.nodes_from:
                random_node.nodes_from.remove(other_random_node)
                if not random_node.nodes_from:
                    random_node.content['parent_model'] = None
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
            if random_node.nodes_from is not None and other_random_node in random_node.nodes_from:
                random_node.nodes_from.remove(other_random_node)
                if not random_node.nodes_from:
                    random_node.content['parent_model'] = None
                other_random_node.nodes_from.append(random_node)
                ml_models = MlModels()
                other_random_node.content['parent_model'] = ml_models.get_model_by_children_type(other_random_node)
                break
    except Exception as ex:
        print(ex)
    return graph


def custom_mutation_add_model(graph: CompositeModel, **kwargs):
    try:
        all_nodes = graph.nodes
        nodes_with_parents = [node for node in all_nodes if (node.nodes_from != [] and node.nodes_from is not None)]
        if not nodes_with_parents:
            return graph
        node = choice(nodes_with_parents)
        ml_models = MlModels()
        node.content['parent_model'] = ml_models.get_model_by_children_type(node)
    except Exception as ex:
        print(ex)
    return graph
