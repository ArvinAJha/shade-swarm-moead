import numpy as np
import matplotlib.pyplot as plt
from graph import dense_graph, Node
import random

"""
OBJECTIVE FUNCTIONS
"""
IDEAL_TEMPERATURE = 70
MAX_PEOPLE = 10     # we do not know this number in reality.
MAX_BOTS = 10       # we do not know this number in reality. This is the maximum number of bots that should be at a node. Beyond this, there is a penalty. Motivation: a little weird to have 100 bots at a node. We want to encourage the bots to spread out and cool more people and not be a nuisance. 

# def _num_bots_at_node(individual, node, timestep):
#     # this is a helper function to calculate the number of bots at a node at a given timestep. 
#     count = 0
#     for bot_id in range(individual.shape[0]):
#         if individual[bot_id, timestep] == node:
#             count += 1
#     return count

# helper objective functions & constants
def _total_cooling_at_node(node:Node, bot_count:int):
    # this is a helper function to calculate the total cooling at a node. It depends on the number of bots at the node and the number of people at the node. 
    # This is a S-curve. Bots have minimal cooling but work together to increase the cooling effect greatly. Until there is a plateau where adding more bots does not increase the cooling effect.

    L = IDEAL_TEMPERATURE * MAX_PEOPLE  # maximum cooling effect when there are enough bots to cool all people possible at the node
    k = 0.5  # steepness of the curve
    x0 = MAX_PEOPLE / 2  # the number of bots at which the cooling effect is half of the maximum
    cooling = L / (1 + np.exp(-k * (bot_count - x0)))
    return cooling

def _density_penalty_at_node(bot_count):
    if bot_count <= MAX_BOTS:
        return 0
    else:
        over_limit = bot_count - MAX_BOTS
        return over_limit ** 2  # quadratic penalty for exceeding max bots
    
# full objective functions - we want to minimize these. Some functions were originally maximize, but we can minimize the negative of them.
def cooling_deficit(individual):
    """
        Calcuate the total cooling of a swarm on the given map. We want to minimize this.
        @param individual: In the form (num_bots, timeline). Taken from an entry of the population. Is a swarm. 
    """
    
    total_deficit = 0
    timeline = individual.shape[1]
    for timestep in range(timeline):
        nodes, counts = np.unique(individual[:, timestep], return_counts=True)
        
        for node, bot_count in zip(nodes, counts):
            cooling_at_node = _total_cooling_at_node(node, bot_count)
            total_deficit += max(0, IDEAL_TEMPERATURE * node.people - cooling_at_node)

    return total_deficit

def density_penalty(individual):
    """
        Calculate the density penalty of a swarm on the given map. We want to minimize this.
        @param individual: In the form (num_bots, timeline). Taken from an entry of the population. Is a swarm. 
    """
    total_penalty = 0
    timeline = individual.shape[1]
    for timestep in range(timeline):
        nodes, counts = np.unique(individual[:, timestep], return_counts=True)
        
        for _, bot_count in zip(nodes, counts):
            total_penalty += _density_penalty_at_node(bot_count)
            
    return total_penalty

def battery_usage(individual):
    """
        Calculate the battery usage of a swarm on the given map. We want to minimize this.
        @param individual: In the form (num_bots, timeline). Taken from an entry of the population. Is a swarm. 
    """
    # we can model battery usage as the total distance traveled by all bots. Each node transition costs 1 unit of battery. 
    total_usage = 0
    num_bots, timeline = individual.shape
    for bot_id in range(num_bots):
        for step in range(1, timeline):
            if individual[bot_id, step] != individual[bot_id, step-1]:  # if the bot moves to a different node
                total_usage += 1
    return total_usage

class MOEAD:
    def __init__(self, pop_size_n=100, num_bots=5, map=dense_graph, timeline=50, neighborhood_size_percentage_k=0.2):
        self.pop_size_n = pop_size_n
        self.num_bots = num_bots
        self.timeline = timeline    # how many timesteps the bots have to cool down people.
        
        self.fv = np.empty((self.pop_size_n, 3))  # fitness values for each swarm individual and objective

        # NOTE: Neighborhood matrix represents the top K subproblems (weight vectors) closest (geometrically) to some subproblem i (weight vector). 
        self.neighborhood_size_k = int(self.pop_size_n * neighborhood_size_percentage_k)
        self.nb_matrix = np.empty((self.pop_size_n, self.neighborhood_size_k), dtype=int)  # neighborhood matrix

        self.lambdas = np.empty((pop_size_n, 3))  # weight vectors for the 3 objectives

        self.zi = np.array([float('inf')] * 3)  # ideal point, initialized to infinity for minimization

        # This is how the map is defined. Where the bots can go, where people are. 
        self.map = map

        # NOTE: GENOTYPE
        # The genotype is a set of node transitions for each bot in the swarm. Shape: (num_bots, timeline)
        # Example: individual[0] = [0, 1, 2, 3, 4] means bot 0 is at node 0 at timestep 0, node 1 at timestep 1, etc.
        # Each individual in the population represents a complete swarm deployment strategy.
        # The node that a bot can traverse to depends on what is available at the current node. 
        # Each node is an object. This object has properties such as: People, current_bots (changes), etc. People do not move but bots do. 
        self.population = np.empty((self.pop_size_n, self.num_bots, self.timeline), dtype=object) # for each member of the population, there is a swarm of bots, each with their own trajectory. 

        # external population of non-dominated solutions. Update after each generation.
        self.ep = []

    def initialize_weight_vectors(self):
        # TODO: Use Das and Dennis method to initialize weight vectors.
        self.lambdas = np.random.rand(self.pop_size_n, 3)
        self.lambdas = self.lambdas / np.sum(self.lambdas, axis=1, keepdims=True)
    
    def initialize_population(self):

        for i in range(self.pop_size_n):                                        # for each test swarm
            # randomly initialize the population with valid random walks. Each bot in the swarm has a random walk.

            for bot_id in range(self.num_bots):                                 # across each bot in some swarm
                
                full_random_walk = []
                for step in range(self.timeline):                               # for each timestep, let the bot make a valid random walk
                    if step == 0:
                        # start at a random node
                        full_random_walk.append(random.choice(list(self.map.keys())))
                    else:
                        # get the last node and choose a random neighbor
                        last_node = full_random_walk[-1]
                        neighbors = self.map[last_node]
                        full_random_walk.append(random.choice(list(neighbors)))
                self.population[i][bot_id] = full_random_walk

    def initialize_neighborhoods(self):
        # calculate distance between neighbors and (for each neighbor) put the closest K into their neighborhood matrix. 
        
        for i in range(self.pop_size_n):
            distances = np.linalg.norm(self.lambdas - self.lambdas[i], axis=1)
            neighbor_indices = np.argsort(distances)[:self.neighborhood_size_k]
            self.nb_matrix[i] = neighbor_indices

    def _calc_individual_fitness(self, individual, objf1, objf2, objf3):
        # helper function 

        f1 = objf1(individual)
        f2 = objf2(individual)
        f3 = objf3(individual)
        return np.array([f1, f2, f3])
    
    def calculate_fitness_values(self):
        for i in range(self.pop_size_n):
            individual = self.population[i]
            self.fv[i] = self._calc_individual_fitness(individual, cooling_deficit, density_penalty, battery_usage)

            # update global ideal point
            self.zi[0] = min(self.zi[0], self.fv[i][0])
            self.zi[1] = min(self.zi[1], self.fv[i][1])
            self.zi[2] = min(self.zi[2], self.fv[i][2])

    def chevy_silverado(self, weight_vector, child_fitness):
        """
        Calculate the Chebyshev scalarization score.
        @param weight_vector: 1D array of 3 weights (e.g., self.lambdas[neighbor_idx])
        @param child_fitness: 1D array of the 3 objective scores for the new swarm
        """
        chevy = np.max(weight_vector * np.abs(child_fitness - self.zi))
        return chevy

    def update_neighborhood(self, child_individual, child_fitness, subproblem_index):
        # for each neighbor of the subproblem, calculate the chevy score with the child. If better than current neighbor, overwrite population and fitness value. 
        
        # get neighbors (closests weight vectors) of the subproblem
        neighbors = self.nb_matrix[subproblem_index]

        # for each neighbor
        for neighbor_idx in neighbors:
            neighbor_weights = self.lambdas[neighbor_idx]

            # compare the scores of the child and all current neighbors
            child_chevy = self.chevy_silverado(neighbor_weights, child_fitness)

            # resident 
            resident_fitness = self.fv[neighbor_idx]
            current_chevy = self.chevy_silverado(neighbor_weights, resident_fitness)

            if child_chevy < current_chevy:  # if the child is better (lower) than the current neighbor
                self.population[neighbor_idx] = child_individual
                self.fv[neighbor_idx] = child_fitness




    # reproduction. when calling, elites are removed. they are appended to population after reproduction (cloning)
    def crossover(self, parent1, parent2):
        return None
    def mutation(self, individual):
        return None

    def _dominates(self, fitness1, fitness2): 
        return np.all(fitness1 <= fitness2) and np.any(fitness1 < fitness2)
    
    # finding EP 
    def find_non_dominated_solutions(self, child, child_fitness):
        if not self.ep: # if empty, just child and give up 
            self.ep.append((child.copy(), child_fitness.copy()))
            return
        
        dominated_solutions = []
        child_is_failure = False
        
        # if not empty, compare child against each solution. 
        # if child dominates some solution, remove that solution. 
        # if child is dominated by some solution, don't add and stop. 
        # if child is non-dominated after all solutions, add to EP.
        for index, (_, sol_fitness) in enumerate(self.ep):

            # if equal, don't add to EP. To encourage diversity.
            if np.array_equal(child_fitness, sol_fitness):
                child_is_failure = True
                break

            if self._dominates(child_fitness, sol_fitness):
                dominated_solutions.append(index)
            elif self._dominates(sol_fitness, child_fitness):
                child_is_failure = True
                break  # child is FAILURE, do not add. Send to room. 

        # Remove dominated solutions (reversed order to avoid index shifting. SO SMART! ALL THESE YEARS! I'VE BEEN A FOOL! Doing a -1 or +1...god)
        for idx in reversed(dominated_solutions):
            self.ep.pop(idx)

        # Add the child solution
        if not child_is_failure:
            self.ep.append((child.copy(), child_fitness.copy()))

def main(MAX_GENERATIONS=100): 
    algo = MOEAD()

    algo.initialize_weight_vectors()
    algo.initialize_neighborhoods()
    algo.initialize_population()

    # evaluate initial population
    algo.calculate_fitness_values()
    # at this point, best solution weights are stored in Zi

    for generation in range(MAX_GENERATIONS):

        # loop each swarm (individual) and do: 
        for i in range(algo.pop_size_n):

            # select 2 random neighboring swarms. These are the parents.
            neighbor_indices = algo.nb_matrix[i]
            parent_swarm_1_index, parent_swarm_2_index = random.sample(list(neighbor_indices), 2)
            parent1 = algo.population[parent_swarm_1_index]
            parent2 = algo.population[parent_swarm_2_index]

            # optional elitism: find the non-dominated solutions in the current population and clone them. 

            # reproduce. crossover and mutation. 
            child = algo.crossover(parent1, parent2)
            child = algo.mutation(child)

            if child is None: 
                child = parent1  # if crossover fails, just clone parent1

            # 2. repair
            
            # 3. Evaluate on objectives. Update Z. 
            child_fitness = algo._calc_individual_fitness(child, cooling_deficit, density_penalty, battery_usage)
            # update global ideal point
            algo.zi[0] = min(algo.zi[0], child_fitness[0])
            algo.zi[1] = min(algo.zi[1], child_fitness[1])
            algo.zi[2] = min(algo.zi[2], child_fitness[2])

            # 3.5. Update EP.
            algo.find_non_dominated_solutions(child, child_fitness)

            # 4. Update neighbors. Calc chyvy score with new child y using weight vecors and Z. If lower (better) than current neighbors, overwrite population. 
            algo.update_neighborhood(child, child_fitness, i)
        
        print(f"Final Ideal Point (Z): {algo.zi} at {generation} generations.")
    
    print(f"Evolution Complete. Found {len(algo.ep)} optimal trade-off solutions.")


if __name__ == "__main__":
    MAX_GENERATIONS = 100
    main(MAX_GENERATIONS)