import random
import numpy as np

class Node: 
    def __init__(self, id, people=0, current_bots=0):
        self.id = id
        self.people = people
        self.current_bots = current_bots
        self.temperature = 100

    def __hash__(self) -> int:
        return hash(self.id)
    def __eq__(self, other) -> bool:
        return self.id == other.id
    def __lt__(self, other):            # for sorting nodes by id. Maybe helps speedup?
        return self.id < other.id

# This makes a dense graph. We do not want a dense graph. This is a start. 
nodes = [Node(i, people=random.randint(0, 10)) for i in range(100)]
dense_graph = {node: {other for other in nodes if other != node} for node in nodes}