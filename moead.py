import numpy as np
import matplotlib.pyplot as plt
from graph import dense_graph, Node, visualize_graph
import random
import networkx as nx
import imageio
import os

"""
OBJECTIVE FUNCTIONS
"""
IDEAL_TEMPERATURE = 70
MAX_PEOPLE = 16     # we do not know this number in reality.
MAX_BOTS = 5       # we do not know this number in reality. This is the maximum number of bots that should be at a node. Beyond this, there is a penalty. Motivation: a little weird to have 100 bots at a node. We want to encourage the bots to spread out and cool more people and not be a nuisance. 

# helper objective functions & constants
def _total_cooling_at_node(node:Node, bot_count:int):
    # this is a helper function to calculate the total cooling at a node. It depends on the number of bots at the node and the number of people at the node. 
    # This is a S-curve. Bots have minimal cooling but work together to increase the cooling effect greatly. Until there is a plateau where adding more bots does not increase the cooling effect.

    L = IDEAL_TEMPERATURE * MAX_PEOPLE  # maximum cooling effect when there are enough bots to cool all people possible at the node
    k = 1.5  # steepness of the curve
    x0 = MAX_PEOPLE / 2  # the number of bots at which the cooling effect is half of the maximum
    cooling = L / (1 + np.exp(-k * (bot_count - x0)))
    return cooling

def _density_penalty_at_node(bot_count):
    if bot_count <= MAX_BOTS:
        return 0
    else:
        over_limit = bot_count - MAX_BOTS
        return over_limit ** 3  # cubic penalty for exceeding max bots
    
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
                total_usage += 15
            else: 
                total_usage += 5
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
        self.nadir = np.array([-float('inf')] * 3)  # nadir point, use this to track the worst objective values. Used to normalize values later. 

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
        # randomly weight vectors for each subproblem. 
        # sub of weights should add to 1.
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

            self.nadir[0] = max(self.nadir[0], self.fv[i][0])
            self.nadir[1] = max(self.nadir[1], self.fv[i][1])
            self.nadir[2] = max(self.nadir[2], self.fv[i][2])
    def chevy_silverado(self, weight_vector, child_fitness):
        """
        Calculate the Chebyshev scalarization score WITH NORMALIZATION.
        """
        # Calculate the range (max - min) to normalize the objective.
        range_z = self.nadir - self.zi
        range_z[range_z == 0] = 1e-5  # add machine epsilon to avoid / 0 (not really epsilon but small value)
        
        # Normalize the fitness to a 0.0 - 1.0 scale
        normalized_fitness = (child_fitness - self.zi) / range_z
        
        chevy = np.max(weight_vector * np.abs(normalized_fitness))
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

    def crossover(self, parent1, parent2, crossover_probability=0.9):
        
        if random.random() > crossover_probability:
            return parent1.copy()
        
        child = np.empty_like(parent1)
        
        for bot_id in range(self.num_bots):
            if random.random() < 0.5:
                child[bot_id] = parent1[bot_id].copy()
            else:
                child[bot_id] = parent2[bot_id].copy()
                
        return child
    
    def mutation(self, individual, mutation_probability=0.4):
        mutated_individual = individual.copy()
        num_bots, timeline = individual.shape
        
        for bot_id in range(num_bots):
            # With probability mutation_probability, mutate this bot
            if random.random() < mutation_probability:
                # Select a random timestep to start the new random walk
                mutation_start_timestep = random.randint(1, timeline - 1)

                # Keep the node at the previous timestep, start from there
                last_node = mutated_individual[bot_id, mutation_start_timestep - 1]
                
                # Regenerate the walk from mutation_start_timestep onward
                for step in range(mutation_start_timestep, timeline):
                    
                    # Pick a random neighbor (or stay if isolated)
                    neighbors = self.map[last_node]
                    if neighbors:
                        mutated_individual[bot_id, step] = random.choice(list(neighbors))
                    else:
                        mutated_individual[bot_id, step] = last_node
        
        return mutated_individual

    def _dominates(self, fitness1, fitness2): 
        return np.all(fitness1 <= fitness2) and np.any(fitness1 < fitness2)
    
    def visualize_objective_space(self, fitness_history, filename="objective_space.png"):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        
        # Flatten fitness history into one array per generation with generation labels
        num_generations = len(fitness_history)
        
        # Calculate normalization ranges
        range_z = self.nadir - self.zi
        range_z[range_z == 0] = 1e-5  # avoid division by zero
        
        for gen_idx, gen_fitness in enumerate(fitness_history):
            # gen_fitness is (pop_size_n, 3)
            gen_fitness = np.array(gen_fitness)
            
            # Normalize fitness values to [0, 1]
            normalized_fitness = (gen_fitness - self.zi) / range_z
            
            # Color: light blue (gen 0) to dark blue (final generation)
            color_intensity = gen_idx / max(1, num_generations - 1)  # 0 to 1
            color = plt.cm.Blues(0.3 + 0.7 * color_intensity)  # 0.3 to 1.0 in Blues colormap
            
            # Plot all solutions from this generation
            ax.scatter(normalized_fitness[:, 0], normalized_fitness[:, 1], normalized_fitness[:, 2],
                      color=color, s=30, alpha=0.6)
        
        # Highlight EP solutions in red
        if self.ep:
            ep_fitness = np.array([fitness for _, fitness in self.ep])
            normalized_ep_fitness = (ep_fitness - self.zi) / range_z
            ax.scatter(normalized_ep_fitness[:, 0], normalized_ep_fitness[:, 1], normalized_ep_fitness[:, 2],
                      color='red', s=100, marker='*', edgecolors='darkred', linewidths=2, 
                      label='Efficient Set (EP)', zorder=5)
        
        # Labels and formatting
        ax.set_xlabel('Cooling Deficit (normalized)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Density Penalty (normalized)', fontsize=11, fontweight='bold')
        ax.set_zlabel('Battery Usage (normalized)', fontsize=11, fontweight='bold')
        ax.set_title('3D Objective Space: Evolution Progress', fontsize=13, fontweight='bold')
        
        # Set normalized axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        
        # Set isometric view
        ax.view_init(elev=20, azim=45)
        
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {filename}")
        plt.close()
    
    def visualize_swarm_trajectory(self, individual, graph, episode_name="swarm_trajectory", fps=2):
        frames_dir = f"{episode_name}_frames"
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)
        
        G = graph
        
        # Use circular layout for clarity and consistency
        pos = nx.circular_layout(G)
        
        frame_files = []
        num_bots, timeline = individual.shape
        
        # Color palette for individual bots
        bot_colors = plt.cm.tab20(np.linspace(0, 1, num_bots))
        
        # Create frame for each timestep
        for timestep in range(timeline):
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Get bot positions at current and next timestep
            current_bots = individual[:, timestep]
            next_bots = individual[:, timestep + 1] if timestep < timeline - 1 else current_bots

            # Draw edges in light gray
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, width=1.5, edge_color='gray')
            
            # Draw individual bots offset from node positions - arranged around nodes
            offset_distance = 0.2
            for bot_id in range(num_bots):
                current_node_id = current_bots[bot_id].id
                next_node_id = next_bots[bot_id].id
                
                current_node_pos = pos[current_node_id]
                next_node_pos = pos[next_node_id]
                
                # Arrange bots in a circle around their node (not stacked)
                # Each bot gets its own angle based on its ID
                angle_per_bot = 2 * np.pi / num_bots
                bot_angle = bot_id * angle_per_bot
                
                # Position current bot around current node
                bot_current_pos = (current_node_pos[0] + offset_distance * np.cos(bot_angle),
                                  current_node_pos[1] + offset_distance * np.sin(bot_angle))
                
                # Position next bot around next node
                bot_next_pos = (next_node_pos[0] + offset_distance * np.cos(bot_angle),
                               next_node_pos[1] + offset_distance * np.sin(bot_angle))
                
                # Draw arrow from current to next position
                if timestep < timeline - 1:
                    dx = bot_next_pos[0] - bot_current_pos[0]
                    dy = bot_next_pos[1] - bot_current_pos[1]
                    
                    ax.arrow(bot_current_pos[0], bot_current_pos[1], dx * 0.85, dy * 0.85,
                            head_width=0.06, head_length=0.04, fc=bot_colors[bot_id], 
                            ec='black', alpha=0.6, linewidth=1.5, zorder=3)
                
                # Draw bot as a small circle at offset position
                ax.scatter(bot_current_pos[0], bot_current_pos[1], s=150, c=[bot_colors[bot_id]],
                          edgecolors='black', linewidths=1.2, zorder=4, marker='o')
                
                # "B" for robot
                ax.text(bot_current_pos[0], bot_current_pos[1], "B", 
                       ha='center', va='center', fontsize=7, fontweight='bold', zorder=5, color='black')
            
            # Title
            ax.set_title(f'Timestep {timestep}/{timeline - 1}', fontsize=12, fontweight='bold', pad=12)
            
            ax.axis('off')
            ax.set_aspect('equal')
            
            plt.tight_layout()
            
            # Save frame
            frame_file = os.path.join(frames_dir, f"frame_{timestep:02d}.png")
            plt.savefig(frame_file, dpi=120, bbox_inches='tight', facecolor='white')
            frame_files.append(frame_file)
            plt.close()
        
        print(f"Frames are saved to {frames_dir}")

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

            # if equal, don't add to EP. Solution already exists. 
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

def main(MAX_GENERATIONS=100, num_bots=5): 
    algo = MOEAD(num_bots=num_bots)

    algo.initialize_weight_vectors()
    algo.initialize_neighborhoods()
    algo.initialize_population()

    # evaluate initial population
    algo.calculate_fitness_values()
    # at this point, best solution weights are stored in Zi

    population = []

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

            # 2. repair (not needed)
            
            # 3. Evaluate on objectives. Update Z. 
            child_fitness = algo._calc_individual_fitness(child, cooling_deficit, density_penalty, battery_usage)
            # update global ideal point
            algo.zi[0] = min(algo.zi[0], child_fitness[0])
            algo.zi[1] = min(algo.zi[1], child_fitness[1])
            algo.zi[2] = min(algo.zi[2], child_fitness[2])

            algo.nadir[0] = max(algo.nadir[0], child_fitness[0])
            algo.nadir[1] = max(algo.nadir[1], child_fitness[1])
            algo.nadir[2] = max(algo.nadir[2], child_fitness[2])

            # 3.5. Update EP.
            algo.find_non_dominated_solutions(child, child_fitness)

            # 4. Update neighbors. Calc chyvy score with new child y using weight vecors and Z. If lower (better) than current neighbors, overwrite population. 
            algo.update_neighborhood(child, child_fitness, i)
            
            population.append(list(algo.fv[:int(np.floor(len(algo.fv)*1.0))].copy()))  # for visualization 
        
        print(f"Final Ideal Point (Z): {algo.zi} at {generation} generations.")
    
    print(f"Evolution Complete. Found {len(algo.ep)} optimal trade-off solutions.")
    
    # Visualize the 3D objective space evolution
    algo.visualize_objective_space(population, filename="moead_objective_space.png")
    
    # Visualize swarm trajectory for the first EP solution
    if algo.ep:
        best_individual, best_fitness = algo.ep[0]
        algo.visualize_swarm_trajectory(best_individual, dense_graph, episode_name="ep_solution_trajectory", fps=2)


if __name__ == "__main__":

    visualize_graph(dense_graph)

    MAX_GENERATIONS = 200
    main(MAX_GENERATIONS, num_bots=30)