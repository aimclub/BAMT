from datetime import timedelta
import sys
import os
import pathlib
current_path = pathlib.Path().resolve()
parentdir = os.getcwd() 
sys.path.insert(0, parentdir)
sys.path.append(str(current_path) + '\\examples\\bn')
import pandas as pd
from sklearn import preprocessing
import bamt.preprocessors as pp
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.adapter import DirectAdapter
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.selection import SelectionTypesEnum
from golem.core.optimisers.genetic.operators.crossover import exchange_parents_one_crossover, exchange_parents_both_crossover, exchange_edges_crossover
from golem.core.optimisers.objective.objective import Objective
from golem.core.optimisers.optimizer import GraphGenerationParams
from examples.composite_bn.composite_model import CompositeModel
from examples.composite_bn.composite_node import CompositeNode
from sklearn.model_selection import train_test_split
from examples.composite_bn.composite_bn_genetic_operators import (
    custom_crossover_all_model as composite_crossover_all_model, 
    custom_mutation_add_structure as composite_mutation_add_structure, 
    custom_mutation_delete_structure as composite_mutation_delete_structure, 
    custom_mutation_reverse_structure as composite_mutation_reverse_structure,
    custom_mutation_add_model as composite_mutation_add_model,
)
from functools import partial
from examples.composite_bn.comparison import Comparison
from examples.composite_bn.fitness_function import FitnessFunction
from examples.composite_bn.rule import Rule
from examples.composite_bn.likelihood import Likelihood
from examples.composite_bn.write_txt import Write
from examples.bn.bn_genetic_operators import (
    custom_mutation_add_structure as classical_mutation_add_structure,
    custom_mutation_delete_structure as classical_mutation_delete_structure,
    custom_mutation_reverse_structure as classical_mutation_reverse_structure
)



def run_example(file):
    if exist_true_str:
        with open('examples/data/1000/txt/'+(file)+'.txt') as f:
            lines = f.readlines()
        true_net = []
        for l in lines:
            e0 = l.split()[0]
            e1 = l.split()[1].split('\n')[0]
            true_net.append((e0, e1))    
    
    fitness_function = FitnessFunction()
    FF_classical = fitness_function.classical_K2
    FF_composite = fitness_function.composite_metric

    if bn_type == 'classical':
        fitness_function_GA = FF_classical

        mutations = [
        classical_mutation_add_structure, 
        classical_mutation_delete_structure, 
        classical_mutation_reverse_structure
        ]

        crossovers = [
            exchange_parents_one_crossover,
            exchange_parents_both_crossover,
            exchange_edges_crossover
            ]
    elif bn_type == 'composite':

        composite_FF = FF_composite
        complexity_FF = FF_classical


        mutations = [
        composite_mutation_add_structure, 
        composite_mutation_delete_structure, 
        composite_mutation_reverse_structure, 
        composite_mutation_add_model    
        ]    

        crossovers = [
            exchange_parents_one_crossover,
            exchange_parents_both_crossover,
            exchange_edges_crossover,
            composite_crossover_all_model
            ]
    else:
        print('There is no such type of BN: "{}". You can only use "classical" or "composite".'.format(bn_type)) 
        return 


    if file in ['abalone', 'adult', 'australian_statlog', 'liver_disorders']:
        data = pd.read_csv('examples/data/1000/UCI/' + file + '.data') 
    else:
        data = pd.read_csv('examples/data/1000/csv/' + file + '.csv') 

    if 'Unnamed: 0' in data.columns:
        data.drop(['Unnamed: 0'], axis=1, inplace=True)
    data.dropna(inplace=True)
    data.reset_index(inplace=True, drop=True)
    vertices = list(data.columns)

    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

    p_for_composite = pp.Preprocessor([('encoder', encoder)])
    discretized_data_composite, _ = p_for_composite.apply(data)

    data_train_composite , data_val_composite = train_test_split(discretized_data_composite, test_size=0.2, shuffle = True, random_state=random_seed[number-1])
    data_train_composite , data_test_composite = train_test_split(data_train_composite, test_size=0.2, shuffle = True, random_state=random_seed[number-1])

    rules = Rule().bn_rules()

    initial = [CompositeModel(nodes=[CompositeNode(nodes_from=None,
                                                    content={'name': vertex,
                                                            'type': p_for_composite.nodes_types[vertex],
                                                            'parent_model': None}) 
                                                    for vertex in vertices])] 

  
    objective = Objective(
            {'fitness function': partial(composite_FF, data_train = data_train_composite, data_test = data_test_composite)},
            is_multi_objective=False,
        )

    requirements = GraphRequirements(
        max_arity=100,
        max_depth=100, 
        num_of_generations=n_generation,
        timeout=timedelta(minutes=time_m),
        early_stopping_iterations = early_stopping_iterations,
        n_jobs=-1
        )

    optimiser_parameters = GPAlgorithmParameters(
        multi_objective=objective.is_multi_objective,
        pop_size=pop_size,
        max_pop_size = pop_size,
        crossover_prob=crossover_probability, 
        mutation_prob=mutation_probability,
        genetic_scheme_type = GeneticSchemeTypesEnum.steady_state,
        selection_types = [SelectionTypesEnum.tournament],
        mutation_types = mutations,
        crossover_types = crossovers,
    )

    graph_generation_params = GraphGenerationParams(
        adapter=DirectAdapter(base_graph_class=CompositeModel, base_node_class=CompositeNode),
        rules_for_constraint=rules,
        )

    optimiser = EvoGraphOptimizer(
        graph_generation_params=graph_generation_params,
        graph_optimizer_params=optimiser_parameters,
        requirements=requirements,
        initial_graphs=initial,
        objective=objective)

    optimized_graphs = optimiser.optimise(objective)

    vars_of_interest = {}
    comparison = Comparison()
    LL = Likelihood()    

    for optimized_graph in optimized_graphs:
        optimized_structure = [(str(edge[0]), str(edge[1])) for edge in optimized_graph.get_edges()]
        score = composite_FF(optimized_graph, data_train = data_train_composite, data_test = data_val_composite)
        spent_time = optimiser.timer.minutes_from_start

        if bn_type == 'composite':
            p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)]) 
            discretized_data, _ = p.apply(data)
            data_train_test , data_val = train_test_split(discretized_data, test_size=0.2, shuffle = True, random_state=random_seed[number-1])
            data_train , data_test = train_test_split(data_train_test, test_size=0.2, shuffle = True, random_state=random_seed[number-1])

        likelihood_classical = LL.likelihood_function(optimized_graph, data_train=data_train, data_val=data_val)
        likelihood_composite = LL.likelihood_function_composite(optimized_graph, data_train_composite, data_val_composite)
        
        if exist_true_str:
            f1 = comparison.F1(optimized_structure, true_net)
            SHD = comparison.precision_recall(optimized_structure, true_net)['SHD']
        if bn_type == 'composite': 
            models = {node:node.content['parent_model'] for node in optimized_graph.nodes}

        vars_of_interest['Structure'] = optimized_structure
        vars_of_interest['Score'] = -score
        vars_of_interest['Likelihood classical'] = likelihood_classical
        vars_of_interest['Likelihood composite'] = likelihood_composite
        
        vars_of_interest['Spent time'] = spent_time
        if exist_true_str:
            vars_of_interest['f1'] = f1
            vars_of_interest['SHD'] = SHD
        if bn_type == 'composite': 
            vars_of_interest['Models'] = models

        vars_of_interest['Generation number'] = optimiser.current_generation_num
        vars_of_interest['Population number'] = optimiser.graph_optimizer_params.pop_size

        write = Write()
        write.write_txt(vars_of_interest, path = os.path.join(parentdir, 'examples'), file_name = 'composite_BN_' + file + '_run_' + str(number) + '.txt')
        


if __name__ == '__main__':
    files =  ['asia', 'cancer', 'earthquake', 'sachs', 'survey', 'healthcare', 'sangiovese', 'alarm', 'barley', 'child', 'insurance', 'mildew', 'water', 'ecoli70', 'magic-niab', 'mehra-complete', 'hailfinder']
    exist_true_str = True
    pop_size = 20 
    n_generation = 1000
    crossover_probability = 0.8
    mutation_probability = 0.9
    early_stopping_iterations = 10 
    time_m = 15
    random_seed = [87, 60, 37, 99, 42, 92, 48, 91, 86, 33]

    n = 10
    for file in files:

        for bn_type in ['composite']: 
            number = 1
            while number <= n:
                run_example(file) 
                number += 1 




