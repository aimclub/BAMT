from typing import List, Callable, Dict, Any, Optional, Tuple
import random
import time
from datetime import datetime, timedelta
import multiprocessing
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import copy

from bamt.builders.evo_builders.deap_graph import Graph, Node


class GraphEvolutionOptimizer:
    """A graph structure optimization engine using DEAP framework."""

    def __init__(
        self,
        objective_function: Callable[[Graph, Any], float],
        constraints: List[Callable[[Graph], bool]],
        data: Optional[Any] = None,
        population_size: int = 50,
        generations: int = 100,
        tournament_size: int = 3,
        crossover_probability: float = 0.9,
        mutation_probability: float = 0.2,
        n_jobs: int = 1,
        early_stopping_rounds: int = 10,
        timeout_minutes: int = 60,
        maximize: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize the optimizer with evolutionary parameters.

        Args:
            objective_function: Function to evaluate graph fitness
            constraints: List of constraint functions that a graph must satisfy
            data: Data to be used by the objective function
            population_size: Size of the population in each generation
            generations: Maximum number of generations to evolve
            tournament_size: Size of tournament selection
            crossover_probability: Probability of crossover
            mutation_probability: Probability of mutation
            n_jobs: Number of parallel jobs (-1 for all available cores)
            early_stopping_rounds: Stop if no improvement after this many generations
            timeout_minutes: Maximum runtime in minutes
            maximize: If True, maximize the objective; otherwise minimize
            verbose: Whether to print progress information
        """
        self.objective_function = objective_function
        self.constraints = constraints
        self.data = data
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_probability
        self.mutation_prob = mutation_probability
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        self.early_stopping_rounds = early_stopping_rounds
        self.timeout = timedelta(minutes=timeout_minutes)
        self.maximize = maximize
        self.verbose = verbose

        # Register types with DEAP
        if not hasattr(creator, "FitnessMax"):
            if maximize:
                creator.create("FitnessMax", base.Fitness, weights=(1.0,))
                creator.create("Individual", list, fitness=creator.FitnessMax)
            else:
                creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
                creator.create("Individual", list, fitness=creator.FitnessMin)

    def _evaluate_graph(self, graph: Graph) -> float:
        """Evaluate a graph's fitness based on the objective function."""
        try:
            # Check if graph satisfies all constraints
            for constraint in self.constraints:
                if not constraint(graph):
                    return float("-inf") if self.maximize else float("inf")

            # Evaluate the graph
            return self.objective_function(graph, self.data)
        except Exception as e:
            if self.verbose:
                print(f"Error evaluating graph: {e}")
            return float("-inf") if self.maximize else float("inf")

    def optimize(self, initial_graphs: List[Graph]) -> List[Tuple[Graph, float]]:
        """
        Run the evolutionary optimization process.

        Args:
            initial_graphs: List of initial graphs to seed the population

        Returns:
            List of tuples (graph, fitness) sorted by fitness (best first)
        """
        start_time = datetime.now()

        # Create a new toolbox for this optimization run
        toolbox = base.Toolbox()

        # Define function to create a random graph (individual)
        def create_random_graph():
            if initial_graphs and random.random() < 0.5:
                # Clone a random initial graph
                return copy.deepcopy(random.choice(initial_graphs))
            else:
                # Create a new random graph based on the initial graphs
                # This is a placeholder - in a real implementation,
                # you would create a meaningful random graph
                return copy.deepcopy(random.choice(initial_graphs))

        # Register the individual creation function
        toolbox.register(
            "individual",
            tools.initIterate,
            creator.Individual,
            lambda: [create_random_graph()],
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Define evaluation function
        def evaluate_individual(individual):
            graph = individual[0]  # Extract graph from individual
            return (self._evaluate_graph(graph),)

        toolbox.register("evaluate", evaluate_individual)

        # Define selection, crossover, and mutation operators
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

        # Custom crossover operator for graphs
        def graph_crossover(ind1, ind2):
            graph1 = ind1[0]
            graph2 = ind2[0]

            # Implement graph crossover logic
            # For example, exchange subgraphs between the two graphs

            # This is a simplified example - in reality, you would implement
            # a more sophisticated crossover for graphs
            if random.random() < 0.5:
                # Randomly select a node from each graph and swap their outgoing edges
                if graph1.nodes and graph2.nodes:
                    node1 = random.choice(graph1.nodes)
                    node2 = random.choice(graph2.nodes)

                    # Swap children lists
                    temp_children = node1.children[:]
                    node1.children = node2.children[:]
                    node2.children = temp_children

                    # Update parent references
                    for child in node1.children:
                        if node2 in child.parents:
                            child.parents.remove(node2)
                        if node1 not in child.parents:
                            child.parents.append(node1)

                    for child in node2.children:
                        if node1 in child.parents:
                            child.parents.remove(node1)
                        if node2 not in child.parents:
                            child.parents.append(node2)

            # Return modified individuals
            return ind1, ind2

        toolbox.register("mate", graph_crossover)

        # Custom mutation operator for graphs
        def graph_mutate(individual):
            graph = individual[0]

            # Implement graph mutation logic
            # For example, add/remove edges or modify node attributes

            # This is a simplified example - in reality, you would implement
            # more sophisticated mutations specific to your problem
            if graph.nodes:
                mutation_type = random.choice(
                    ["add_edge", "remove_edge", "reverse_edge"]
                )

                if mutation_type == "add_edge" and len(graph.nodes) >= 2:
                    # Add a random edge
                    source = random.choice(graph.nodes)
                    target = random.choice([n for n in graph.nodes if n != source])

                    # Avoid creating cycles
                    test_graph = graph.copy()
                    test_graph.add_edge(source, target)
                    if not test_graph.has_cycle():
                        graph.add_edge(source, target)

                elif mutation_type == "remove_edge":
                    # Find nodes with children and remove a random edge
                    nodes_with_children = [n for n in graph.nodes if n.children]
                    if nodes_with_children:
                        source = random.choice(nodes_with_children)
                        if source.children:
                            target = random.choice(source.children)
                            graph.remove_edge(source, target)

                elif mutation_type == "reverse_edge":
                    # Find an edge and reverse it
                    edges = [
                        (node, child) for node in graph.nodes for child in node.children
                    ]
                    if edges:
                        source, target = random.choice(edges)

                        # Test if reversing would create a cycle
                        test_graph = graph.copy()
                        test_graph.remove_edge(source, target)
                        test_graph.add_edge(target, source)

                        if not test_graph.has_cycle():
                            graph.remove_edge(source, target)
                            graph.add_edge(target, source)

            return (individual,)

        toolbox.register("mutate", graph_mutate)

        # Parallel evaluation if needed
        if self.n_jobs > 1:
            pool = multiprocessing.Pool(processes=self.n_jobs)
            toolbox.register("map", pool.map)

        # Create initial population
        pop = toolbox.population(n=self.population_size)

        # Add initial graphs to the population
        for i, graph in enumerate(initial_graphs[: self.population_size]):
            if i < len(pop):
                pop[i][0] = copy.deepcopy(graph)

        # Variables for tracking the best individual and early stopping
        best_fitness = float("-inf") if self.maximize else float("inf")
        best_gen = 0
        best_individual = None

        # Evaluate the entire population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Begin the evolution
        for gen in range(self.generations):
            if (datetime.now() - start_time) > self.timeout:
                if self.verbose:
                    print(f"Timeout reached after {gen} generations")
                break

            # Select and clone the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation
            for i in range(1, len(offspring), 2):
                if random.random() < self.crossover_prob:
                    offspring[i - 1], offspring[i] = toolbox.mate(
                        offspring[i - 1], offspring[i]
                    )
                    del offspring[i - 1].fitness.values
                    del offspring[i].fitness.values

            for i in range(len(offspring)):
                if random.random() < self.mutation_prob:
                    (offspring[i],) = toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace the current population with the offspring
            pop[:] = offspring

            # Find the best individual in this generation
            current_best = tools.selBest(pop, 1)[0]
            current_best_fitness = current_best.fitness.values[0]

            # Check if we have a new best
            if (self.maximize and current_best_fitness > best_fitness) or (
                not self.maximize and current_best_fitness < best_fitness
            ):
                best_fitness = current_best_fitness
                best_individual = current_best
                best_gen = gen

                if self.verbose:
                    print(f"Generation {gen}: New best fitness = {best_fitness}")

            # Early stopping
            if gen - best_gen >= self.early_stopping_rounds:
                if self.verbose:
                    print(f"Early stopping triggered after {gen} generations")
                break

            if self.verbose and gen % 10 == 0:
                print(f"Generation {gen} completed")

        if self.n_jobs > 1:
            pool.close()

        # Return the best individuals sorted by fitness
        best_individuals = tools.selBest(pop, min(10, len(pop)))
        result = []
        for ind in best_individuals:
            result.append((ind[0], ind.fitness.values[0]))

        return sorted(result, key=lambda x: x[1], reverse=self.maximize)
