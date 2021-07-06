from copy import deepcopy
from fedot.core.chains.graph_node import PrimaryGraphNode
from random import choice, random
from typing import Any, List

from fedot.core.composer.constraint import constraint_function
from fedot.core.composer.optimisers.gp_comp.gp_operators import \
    (equivalent_subtree, node_depth,
     nodes_from_height, replace_subtrees)
from fedot.core.log import Log
from fedot.core.utils import ComparableEnum as Enum
from itertools import groupby

MAX_NUM_OF_ATTEMPTS = int(1e1)


class CrossoverTypesEnum(Enum):
    subtree = 'subtree'
    one_point = "one_point"
    none = 'none'


def will_crossover_be_applied(chain_first, chain_second, crossover_prob, crossover_type) -> bool:
    return not (chain_first is chain_second or
                random() > crossover_prob or
                crossover_type == CrossoverTypesEnum.none)


def crossover(types: List[CrossoverTypesEnum],
              chain_first: Any, chain_second: Any,
              max_depth: int, log: Log, crossover_prob: float = 0.8, chain_generation_params=None) -> Any:
    crossover_type = choice(types)
    rules = chain_generation_params.rules_for_constraint if chain_generation_params else None
    try:
        if will_crossover_be_applied(chain_first, chain_second, crossover_prob, crossover_type):
            if crossover_type in crossover_by_type.keys():
                for _ in range(MAX_NUM_OF_ATTEMPTS):
                    new_chains = crossover_by_type[crossover_type](deepcopy(chain_first),
                                                                   deepcopy(chain_second), max_depth)
                    are_correct = \
                        all([constraint_function(new_chain,
                                                 rules) for new_chain
                             in new_chains])
                    if are_correct:
                        return new_chains
            else:
                raise ValueError(f'Required crossover type not found: {crossover_type}')
            log.debug('Number of crossover attempts exceeded. '
                      'Please check composer requirements for correctness.')
    except Exception as ex:
        log.error(f'Crossover ex: {ex}')

    chain_first_copy = deepcopy(chain_first)
    chain_second_copy = deepcopy(chain_second)
    return chain_first_copy, chain_second_copy

def no_tabu_node(tabu_list: Any, chain: Any) -> Any:
    first = []
    for node in chain.nodes:
        if (isinstance(node, PrimaryGraphNode) or not node.nodes_from):
            first.append(node)
    first = [node for node in first if node not in tabu_list]
    if not first:
        return []
    else:
        visited = True
        curr = first
        while visited:
            next = []
            for node in curr:
                next.extend(chain.node_childs(node))
            next = [el for el, _ in groupby(next)]   
            next = [node for node in next if node not in tabu_list]
            if not next:
                visited = False
            else:
                first.extend(next)
                curr = next
        return choice(first)

def tabu_list(old_node: Any, chain: Any) -> Any:
    nodes = chain.nodes
    index = [True for _ in range(len(nodes))]
    index[nodes.index(old_node)] = False
    curr = [old_node]
    visited = True
    while visited:
        next = []
        for node in curr:
            if node.nodes_from:
                for parent in node.nodes_from:
                    if index[nodes.index(parent)]:
                        index[nodes.index(parent)] = False
                        next.append(parent)
        if next:
            curr = next
        else:
            visited = False
    return [node for node in nodes if index]


def subtree_crossover(chain_first: Any, chain_second: Any, max_depth: int) -> Any:
    """Performed by the replacement of random subtree
    in first selected parent to random subtree from the second parent"""
    """random_layer_in_chain_first = choice(range(chain_first.depth))
    min_second_layer = 1 if random_layer_in_chain_first == 0 else 0
    random_layer_in_chain_second = choice(range(min_second_layer, chain_second.depth))
    if isinstance(nodes_from_height(chain_first, random_layer_in_chain_first), list):
        node_from_chain_first = choice(nodes_from_height(chain_first, random_layer_in_chain_first))
    else:
        node_from_chain_first = nodes_from_height(chain_first, random_layer_in_chain_first)
    if isinstance(nodes_from_height(chain_second, random_layer_in_chain_second), list):
        node_from_chain_second = choice(nodes_from_height(chain_second, random_layer_in_chain_second))
    else:
        node_from_chain_second = nodes_from_height(chain_second, random_layer_in_chain_second)"""
    node_from_chain_first = choice(chain_first.nodes)
    node_from_chain_second = no_tabu_node(tabu_list(node_from_chain_first, chain_first), chain_second)
    random_layer_in_chain_first, random_layer_in_chain_second = 1, 1
    replace_subtrees(chain_first, chain_second, node_from_chain_first, node_from_chain_second,
                     random_layer_in_chain_first, random_layer_in_chain_second, max_depth)

    return chain_first, chain_second


def one_point_crossover(chain_first: Any, chain_second: Any, max_depth: int) -> Any:
    """Finds common structural parts between two trees, and after that randomly
    chooses the location of nodes, subtrees of which will be swapped"""
    pairs_of_nodes = equivalent_subtree(chain_first, chain_second)
    if pairs_of_nodes:
        node_from_chain_first, node_from_chain_second = choice(pairs_of_nodes)
        layer_in_chain_first = 0
        layer_in_chain_second = 0
        #layer_in_chain_first = node_depth(chain_first.root_node) - node_depth(node_from_chain_first)
        #layer_in_chain_second = node_depth(chain_second.root_node) - node_depth(node_from_chain_second)

        replace_subtrees(chain_first, chain_second, node_from_chain_first, node_from_chain_second,
                         layer_in_chain_first, layer_in_chain_second, max_depth)
    return chain_first, chain_second


crossover_by_type = {
    CrossoverTypesEnum.subtree: subtree_crossover,
    CrossoverTypesEnum.one_point: one_point_crossover,
}
