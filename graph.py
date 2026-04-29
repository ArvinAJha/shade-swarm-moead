import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class Node: 
    def __init__(self, id, people=0):
        self.id = id
        self.people = people
        self.temperature = 100

    def __hash__(self) -> int:
        return hash(self.id)
    def __eq__(self, other) -> bool:
        return self.id == other.id
    def __lt__(self, other):            # for sorting nodes by id. Maybe helps speedup?
        return self.id < other.id

# Create a k-regular graph: each node connects to exactly k neighbors
# This is simple, sparse, and fully connected
def create_kregular_graph(num_nodes=70, k=4):
    """
    Create a k-regular graph where each node has exactly k connections.
    
    @param num_nodes: Number of nodes in the graph
    @param k: Degree of each node (number of connections per node)
    @return: Dictionary representation of the graph
    """
    nodes = [Node(i, people=np.ceil(np.random.normal(12, 5))) for i in range(num_nodes)]
    graph = {node: set() for node in nodes}
    
    # Connect each node to the next k nodes in the ring (cyclically)
    for i, node in enumerate(nodes):
        for j in range(1, k // 2 + 1):  # TODO: review this cause doesnt seem what we want as regular? or connecting to first few nodes? 
            # Connect to node (i+j) % num_nodes and (i-j) % num_nodes
            neighbor1 = nodes[(i + j) % num_nodes]
            neighbor2 = nodes[(i - j) % num_nodes]
            graph[node].add(neighbor1)
            graph[node].add(neighbor2)
    
    return graph

def visualize_graph(graph, filename="campus_graph.png", layout="spring"):
    # Convert to networkx graph
    G = nx.Graph()
    
    # Add nodes with people attribute
    for node in graph.keys():
        G.add_node(node.id, people=node.people)
    
    # Add edges
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            G.add_edge(node.id, neighbor.id)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Color nodes by people count
    people_counts = [G.nodes[node]['people'] for node in G.nodes()]
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=1.5)
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=500,
                                   node_color=people_counts, cmap='YlOrRd',
                                   alpha=0.8, edgecolors='black', linewidths=1.5)
    
    # Draw labels (node ID and people count)
    labels = {node: f"{G.nodes[node]['people']}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8, font_weight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(nodes, ax=ax, label='People Count')
    
    # Formatting
    ax.set_title('TITLE HERE', fontsize=28, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Graph visualization saved to {filename}")
    plt.close()

# Create graph with 70 nodes, each with ~4 neighbors
dense_graph = create_kregular_graph(num_nodes=70, k=4)