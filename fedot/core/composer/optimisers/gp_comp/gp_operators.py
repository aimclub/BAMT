from copy import deepcopy
from random import choice, randint, uniform, random
from typing import (Any, Callable, List, Tuple)
from itertools import groupby
from uuid import uuid4
from fedot.core.chains.graph_node import PrimaryGraphNode
from fedot.core.chains.chain_convert import chain_as_nx_graph
from fedot.core.composer.constraint import constraint_function
from fedot.core.composer.optimisers.utils.multi_objective_fitness import MultiObjFitness

def _has_no_duplicates(graph):
    _, labels = chain_as_nx_graph(graph)
    list_of_nodes = []
    for node in labels.values():
        list_of_nodes.append(str(node))
    if len(list_of_nodes) != len(set(list_of_nodes)):
        return False
    return True

def node_depth(node: Any) -> int:
    if not node:
        return 0
    elif not node.nodes_from:
        return 0
    else:
        return 1 + max([node_depth(next_node) for next_node in node.nodes_from])

def nodes_from_depth(chain: Any, selected_height: int) -> List[Any]:
    nodes = []
    def get_nodes(chain: Any, node: Any, current_height):
        nonlocal nodes
        if current_height == selected_height:
                nodes.extend(node)
                return nodes
        elif isinstance(node, list):
            parents = []
            for simple_node in node:
                if chain.node_childs(simple_node):
                    parents.extend(chain.node_childs(simple_node))
            parents = [el for el, _ in groupby(parents)]
            nodes.extend(get_nodes(chain, parents, current_height + 1))
            return nodes
        else:
            if chain.node_childs(node):
                    nodes.extend(get_nodes(chain, chain.node_childs(node), current_height + 1))
            return nodes
    first = [node for node in chain.nodes if type(node) == PrimaryGraphNode]
    if not len(first):
        print("THERE ISN'T PRIMARY NODE")
    nodes = get_nodes(chain, first, current_height=0)
    nodes = [el for el, _ in groupby(nodes)]
    return nodes

def node_height(chain: Any, node: Any) -> int:
    def recursive_child_height(parent_node: Any) -> int:
        node_child = chain.node_childs(parent_node)
        if node_child:
            height = max(recursive_child_height(child) for child in node_child) + 1
            return height
        else:
            return 0

    height = recursive_child_height(node)
    return height


def nodes_from_height(chain: Any, selected_height: int) -> List[Any]:
    nodes = []
    def get_nodes(node: Any, current_height):
        nonlocal nodes
        if current_height == selected_height:
                nodes.extend(node)
                return nodes
        elif isinstance(node, list):
            parents = []
            for simple_node in node:
                if simple_node.nodes_from:
                    parents.extend(simple_node.nodes_from)
            parents = [el for el, _ in groupby(parents)]
            nodes.extend(get_nodes(parents, current_height + 1))
            return nodes
        else:
            if node.nodes_from:
                nodes.extend(get_nodes(node.nodes_from, current_height + 1))
            return nodes

    nodes = get_nodes(chain.root_node, current_height=0)
    nodes = [el for el, _ in groupby(nodes)]
    return nodes



def random_chain(chain_generation_params, requirements, max_depth=None) -> Any:
    secondary_node_func = chain_generation_params.secondary_node_func
    primary_node_func = chain_generation_params.primary_node_func
    chain_class = chain_generation_params.chain_class
    max_depth = max_depth if max_depth else requirements.max_depth

    def chain_growth(chain: Any, node_parent: Any):
        coin = uniform(0, 1)
        prim = set(requirements.primary).difference(set(map(str,chain.nodes)))
        if coin > 0.3:
            second = []
            offspring_size = {parent: randint(requirements.min_arity, requirements.max_arity) for parent in node_parent}
            while sum(list(offspring_size.values())) > 0:
                full = sum(list(offspring_size.values()))
                parents = []
                while not parents:
                    parents = [parent for parent in node_parent if offspring_size[parent]/full > random()]

                height = max([node_height(chain, parent) for parent in parents])
                is_max_depth_exceeded = height >= max_depth - 1
                is_primary_node_selected = height < max_depth - 1 and randint(0, 1)
                if is_max_depth_exceeded or is_primary_node_selected:
                    #primary_node = primary_node_func(operation_type=choice(requirements.primary))
                    if prim:
                        primary_node = primary_node_func(operation_type=choice(list(prim)))
                    else:
                        break
                    prim.difference_update(set(str(primary_node))) 
                    [parent.nodes_from.append(primary_node) for parent in parents]
                    chain.add_node(primary_node)
                    if not _has_no_duplicates(chain):
                        break
                else:
                    #secondary_node = secondary_node_func(operation_type=choice(requirements.secondary))
                    if prim:
                        secondary_node = secondary_node_func(operation_type=choice(list(prim)))
                    else:
                        break
                    prim.difference_update(set(str(secondary_node)))
                    chain.add_node(secondary_node)
                    if not _has_no_duplicates(chain):
                        break
                    [parent.nodes_from.append(secondary_node) for parent in parents]
                    second.append(secondary_node)
                for parent in parents:
                    offspring_size[parent] -= 1
            """for offspring_node in range(offspring_size):
                height = node_height(chain, parent)
                is_max_depth_exceeded = height >= max_depth - 1
                is_primary_node_selected = height < max_depth - 1 and randint(0, 1)
                if is_max_depth_exceeded or is_primary_node_selected:
                    primary_node = primary_node_func(operation_type=choice(requirements.primary))
                    parent.nodes_from.append(primary_node)
                    chain.add_node(primary_node)
                else:
                    secondary_node = secondary_node_func(operation_type=choice(requirements.secondary))
                    chain.add_node(secondary_node)
                    parent.nodes_from.append(secondary_node)"""
                    #node_parent = list(filter(lambda a: a != parent, node_parent))
                    #node_parent.append(secondary_node)
                    #chain_growth(chain, node_parent)
                    #second.append(secondary_node)
            if second:
                chain_growth(chain, second)
        else:
            #chain_root = secondary_node_func(operation_type=choice(requirements.secondary))
            if prim:
                chain_root = secondary_node_func(operation_type=choice(list(prim)))
                prim.difference_update(set(str(chain_root)))
                chain.add_node(chain_root)
                node_parent.append(chain_root)
                chain_growth(chain, node_parent)

    is_correct_chain = False
    chain = None
    while not is_correct_chain:
        chain = chain_class()
        chain_root = secondary_node_func(operation_type=choice(requirements.secondary))
        chain.add_node(chain_root)
        chain_growth(chain, [chain_root])
        is_correct_chain = constraint_function(chain,
                                               chain_generation_params.rules_for_constraint)
    return chain


def equivalent_subtree(chain_first: Any, chain_second: Any) -> List[Tuple[Any, Any]]:
    """Finds the similar subtree in two given trees"""

    def structural_equivalent_nodes(node_first, node_second):
        nodes = []
        is_same_type = type(node_first) == type(node_second)
        node_first_childs = node_first.nodes_from
        node_second_childs = node_second.nodes_from
        if is_same_type and ((not node_first.nodes_from) or len(node_first_childs) == len(node_second_childs)):
            nodes.append((node_first, node_second))
            if node_first.nodes_from:
                for node1_child, node2_child in zip(node_first.nodes_from, node_second.nodes_from):
                    nodes_set = structural_equivalent_nodes(node1_child, node2_child)
                    if nodes_set:
                        nodes += nodes_set
        return nodes

    pairs_set = structural_equivalent_nodes(chain_first.root_node, chain_second.root_node)
    assert isinstance(pairs_set, list)
    return pairs_set


def replace_subtrees(chain_first: Any, chain_second: Any, node_from_first: Any, node_from_second: Any,
                     layer_in_first: int, layer_in_second: int, max_depth: int):
    if isinstance(node_from_first, list):
        print('In REPLACE_SUBTREES there are few nodes as roots')
        if len(node_from_first) > 0:
            node_from_first = choice(node_from_first)
        else:
            node_from_first = None
    if isinstance(node_from_second, list):
        print('In REPLACE_SUBTREES there are few nodes as roots')
        if len(node_from_second) > 0:
            node_from_second = choice(node_from_second)
        else:
            node_from_second = None
    node_from_chain_first_copy = deepcopy(node_from_first)

    chain_first_copy = deepcopy(chain_first)
    #summary_depth = layer_in_first + node_depth(node_from_second)
    summary_depth = chain_first_copy.depth
    if summary_depth <= max_depth and summary_depth != 0:
        chain_first.replace_node_with_parents(node_from_first, node_from_second)

    chain_second_copy = deepcopy(chain_second)
    #summary_depth = layer_in_second + node_depth(node_from_first)
    summary_depth = chain_second_copy.depth
    if summary_depth <= max_depth and summary_depth != 0:
        chain_second.replace_node_with_parents(node_from_second, node_from_chain_first_copy)


def num_of_parents_in_crossover(num_of_final_inds: int) -> int:
    return num_of_final_inds if not num_of_final_inds % 2 else num_of_final_inds + 1


def evaluate_individuals(individuals_set, objective_function, is_multi_objective: bool, timer=None):
    num_of_successful_evals = 0
    reversed_set = individuals_set[::-1]
    for ind_num, ind in enumerate(reversed_set):
        ind.fitness = calculate_objective(ind, objective_function, is_multi_objective)
        if ind.fitness is None:
            individuals_set.remove(ind)
        else:
            num_of_successful_evals += 1
        if timer is not None and num_of_successful_evals:
            if timer.is_time_limit_reached():
                for del_ind_num in range(0, len(individuals_set) - num_of_successful_evals):
                    individuals_set.remove(individuals_set[0])
                break
    if len(individuals_set) == 0:
        raise AttributeError('List became empty after incorrect individuals removing.'
                             'It can occur because of too short model fitting time constraint')


def calculate_objective(ind: Any, objective_function: Callable, is_multi_objective: bool) -> Any:
    calculated_fitness = objective_function(ind)
    if calculated_fitness is None:
        return None
    else:
        if is_multi_objective:
            fitness = MultiObjFitness(values=calculated_fitness,
                                      weights=tuple([-1 for _ in range(len(calculated_fitness))]))
        else:
            fitness = calculated_fitness[0]
    return fitness


def filter_duplicates(archive, population) -> List[Any]:
    filtered_archive = []
    for ind in archive.items:
        has_duplicate_in_pop = False
        for pop_ind in population:
            if ind.fitness == pop_ind.fitness:
                has_duplicate_in_pop = True
                break
        if not has_duplicate_in_pop:
            filtered_archive.append(ind)
    return filtered_archive


def duplicates_filtration(archive, population):
    return list(filter(lambda x: not any([x.fitness == pop_ind.fitness for pop_ind in population]), archive.items))
