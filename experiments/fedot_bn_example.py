import sys
parentdir = 'C:\\Users\\Worker1\\Documents\\BAMT'
sys.path.insert(0,parentdir)
 
import pandas as pd
import random
from functools import partial
from sklearn import preprocessing
import seaborn as sns
 
import bamt.Preprocessors as pp
from bamt.Builders import StructureBuilder
import bamt.Networks as Nets
 
from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.dag.validation_rules import has_no_cycle, has_no_self_cycled_nodes
from fedot.core.log import default_log
from fedot.core.optimisers.adapters import DirectAdapter
from fedot.core.optimisers.gp_comp.gp_optimiser import EvoGraphOptimiser, GPGraphOptimiserParameters, \
    GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.selection import SelectionTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.pipelines.convert import graph_structure_as_nx_graph
from pgmpy.models import BayesianModel
from pgmpy.estimators import K2Score
 
# кастомный граф
class CustomGraphModel(OptGraph):
    def evaluate(self, data: pd.DataFrame):
        nodes = data.columns.to_list()
        _, labels = graph_structure_as_nx_graph(self)
        return len(nodes)
 
# кастомные узлы
class CustomGraphNode(OptNode):
    def __str__(self):
        return f'Node_{self.content["name"]}'
 
# кастомная метрика
def k2_metric(graph: CustomGraphModel, data: pd.DataFrame):
    nodes = data.columns.to_list()
    graph_nx, labels = graph_structure_as_nx_graph(graph)
    struct = []
    for pair in graph_nx.edges():
        l1 = str(labels[pair[0]])
        l2 = str(labels[pair[1]])
        if 'Node' in l1:
            l1 = l1.split('_')[1]
        if 'Node' in l2:
            l2 = l2.split('_')[1]
        struct.append([l1, l2])
    bn_model = BayesianModel(struct)
    no_nodes = []
    for node in nodes:
        if node not in bn_model.nodes():
            no_nodes.append(node)
    #return [random.random()]
    #density = 10000*(2*(len(struct)) / ((len(nodes) - 1)*len(nodes)))
    #nodes_have = 10000*(len(no_nodes) / len(nodes))
    score = K2Score(data).score(bn_model) #- density
    #score = score + nodes_have
    return [score]
def opt_graph_to_bamt(graph: CustomGraphModel):
    graph_nx, labels = graph_structure_as_nx_graph(graph)
    struct = []
    for pair in graph_nx.edges():
        l1 = str(labels[pair[0]])
        l2 = str(labels[pair[1]])
        if 'Node' in l1:
            l1 = l1.split('_')[1]
        if 'Node' in l2:
            l2 = l2.split('_')[1]
        struct.append((l1, l2))
    return struct
 
 
 
 
# проверка "нет дубликатов узлов"
def _has_no_duplicates(graph):
    _, labels = graph_structure_as_nx_graph(graph)
    if len(labels.values()) != len(set(labels.values())):
        raise ValueError('Custom graph has duplicates')
    return True
 
 
def _has_disc_parents(graph):
    graph, labels = graph_structure_as_nx_graph(graph)
    for pair in graph.edges():
        if (node_type[str(labels[pair[1]])] == 'disc') & (node_type[str(labels[pair[0]])] == 'cont'):
            raise ValueError(f'Discrete node has cont parent')
    return True
 
 
# кастомная мутация. ??? Здеь 10 раз пытается провести ориентированное ребро так, чтобы не появился цикл
def custom_mutation(graph: OptGraph, **kwargs):
    num_mut = 10
    try:
        for _ in range(num_mut):
            rid = random.choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[random.choice(range(len(graph.nodes)))]
            nodes_not_cycling = (random_node.descriptive_id not in
                                 [n.descriptive_id for n in other_random_node.ordered_subnodes_hierarchy()] and
                                 other_random_node.descriptive_id not in
                                 [n.descriptive_id for n in random_node.ordered_subnodes_hierarchy()])
            if random_node.nodes_from is not None and len(random_node.nodes_from) == 0:
                random_node.nodes_from = None
            if nodes_not_cycling:
                graph.operator.connect_nodes(random_node, other_random_node)
    except Exception as ex:
        graph.log.warn(f'Incorrect connection: {ex}')
    return graph
 
 
# главная функция, которая последовательно запускается
def run_example():
 
    # Импорт данных и дискретизация
    data=pd.read_csv(r'C:\\Users\\Worker1\\Documents\\BAMT\\Data\\hack_processed_with_rf.csv')
    nodes_types = ['Tectonic regime', 'Period', 'Lithology',
                    'Structural setting', 'Gross', 'Netpay',
                    'Porosity', 'Permeability', 'Depth']
    data = data[nodes_types]
    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    discretized_data, est = p.apply(data)
    #node_type = p.info['types']
    global node_type
    node_type=dict(('Node_'+key, value) for (key, value) in p.info['types'].items())
 
 
    # правила: нет петель, нет циклов, нет дибликатов узлов
    rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates, _has_disc_parents]
    # изициализация графа без связей, только с узлами
    initial = [CustomGraphModel(nodes=[CustomGraphNode(nodes_from=None,
                                                      content={'name': node_type}) for node_type in nodes_types])]
   
   
   
 
 # параметры ГА
    requirements = PipelineComposerRequirements(
        primary=nodes_types,
        secondary=nodes_types, max_arity=100,
        max_depth=100, pop_size=10, num_of_generations=50,
        crossover_prob=0.8, mutation_prob=0.9)
 
# Ещё параметры для ГА
# genetic_scheme_type -> [steady_state, generational, parameter_free]
# crossover_types -> [subtree, one_point, none]
# mutation_types -> [simple, reduce, growth, local_growth] MutationTypesEnum
# selection_types -> [tournament,nsga2, spea2] SelectionTypesEnum
    optimiser_parameters = GPGraphOptimiserParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
# добавила селекцию турниром        
        selection_types=[SelectionTypesEnum.tournament],
        mutation_types=[custom_mutation],
        crossover_types=[CrossoverTypesEnum.none],
        regularization_type=RegularizationTypesEnum.none)
 
# Параметры для генерации графов. Закидываем сюда граф, узлы и правила
    graph_generation_params = GraphGenerationParams(
        adapter=DirectAdapter(base_graph_class=CustomGraphModel, base_node_class=CustomGraphNode),
        rules_for_constraint=rules)
 
# Эволюционный оптимизатор графов  
    optimiser = EvoGraphOptimiser(
        graph_generation_params=graph_generation_params,
        metrics=[],
        parameters=optimiser_parameters,
        requirements=requirements, initial_graph=initial,
        log=default_log(logger_name='Bayesian', verbose_level=1))
 
# в partial указываем целевую функцию?
    optimized_graph = optimiser.optimise(partial(k2_metric, data=discretized_data))
# Сейчас optimized_graph хранит: {'depth': 9, 'length': 9, 'nodes': [Period, Netpay, ...]}
 
# Отличается от optimized_graph только добавлением 'Node_' к названиям узлов
    optimized_network = optimiser.graph_generation_params.adapter.restore(optimized_graph)
   
#    print(optimized_graph)
    # OF=k2_metric(optimized_graph, data=discretized_data)
    # print(OF)
#    print(graph_structure_as_nx_graph(optimized_graph))
   
    optimized_graph.show() 
    return(opt_graph_to_bamt(optimized_network))
    
   
    #optimized_graph.show(path='C:/Users/Worker1/Documents/BAMT/V1.png')
 
    #pdf.add_page()
    #pdf.set_font("Arial", size = 15)
    #pdf.cell(200, 10, txt = "K2_score = " + str(round(OF[0],2)),
    #        ln = 1, align = 'C')
    #pdf.image('C:/Users/Worker1/Documents/BAMT/V1.png',w=200, h=200)
     
 
 
 
 
if __name__ == '__main__':
    #from fpdf import FPDF
    #pdf = FPDF()
 
    data = pd.read_csv(r'C:\\Users\\Worker1\\Documents\\BAMT\\Data\\hack_processed_with_rf.csv')
    nodes_types = ['Tectonic regime', 'Period', 'Lithology',
                    'Structural setting', 'Gross', 'Netpay',
                    'Porosity', 'Permeability', 'Depth']
    data = data[nodes_types]
    structure = run_example()
   
    bn = Nets.HybridBN(has_logit = True, use_mixture=True)
    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    discretized_data, est = p.apply(data)
    info = p.info
    bn.add_nodes(info)
    tmp = StructureBuilder(info)
    tmp.skeleton = {
        'V': bn.nodes,
        'E': structure
    }
    tmp.get_family()
    bn.nodes = tmp.skeleton['V']
    bn.edges = tmp.skeleton['E']
    print(bn.nodes)
    print(bn.edges)
    print(bn.get_info())
    #bn.fit_parameters(data)
    #bn.get_params_tree('fedon_bn.json')
    # sample = bn.sample(100)
    # sns.displot(data['Netpay'])
    # sns.displot(sample['Netpay'])
 
 
 
 
 
    #pdf.output("GFG.pdf")
 

