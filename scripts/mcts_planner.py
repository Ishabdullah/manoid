import math
import random
import torch

class Node:
    def __init__(self, state_idx, parent=None, action=None):
        self.state_idx = state_idx
        self.parent = parent
        self.action = action
        self.N = 0
        self.W = 0.0
        self.children = {}
        self.untried_actions = [0, 1, 2, 3]

class MCTSPlanner:
    def __init__(self, world_model, tier3_symbolic, target_idx=24, grid_size=5, walls=None, dead_ends=None):
        self.world_model = world_model
        self.tier3_symbolic = tier3_symbolic
        self.target_idx = target_idx
        self.grid_size = grid_size
        self.exploration_weight = math.sqrt(2)
        
        self.dead_end_simulations = 0
        
        if walls is None:
            self.walls = [(1,1), (2,1), (3,1), (1,2), (3,2)]
        else:
            self.walls = walls
            
        if dead_ends is None:
            self.dead_ends = [(2,0), (2,1), (2,2), (1,1), (1,2), (3,1), (3,2)] 
        else:
            self.dead_ends = dead_ends

    def search(self, root_state_idx, num_simulations=50):
        root = Node(root_state_idx)
        
        for sim in range(num_simulations):
            node = root
            while len(node.untried_actions) == 0 and len(node.children) > 0:
                node = self.select(node)
                
            if len(node.untried_actions) > 0:
                node = self.expand(node)
                
            value = self.evaluate(node)
            self.backpropagate(node, value)
            
        if not root.children:
            return random.choice([0, 1, 2, 3])
            
        best_n = max(child.N for child in root.children.values())
        best_actions = [a for a, child in root.children.items() if child.N == best_n]
        best_action = max(best_actions, key=lambda a: root.children[a].W / root.children[a].N if root.children[a].N > 0 else float('-inf'))
        return best_action

    def select(self, node):
        best_score = float('-inf')
        best_child = None
        for action, child in node.children.items():
            if child.N == 0:
                return child
            ucb_score = (child.W / child.N) + self.exploration_weight * math.sqrt(math.log(node.N) / child.N)
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        return best_child

    def expand(self, node):
        action = node.untried_actions.pop()
        
        dx, dy = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}[action]
        pos_x = node.state_idx % self.grid_size
        pos_y = node.state_idx // self.grid_size
        
        next_x = max(0, min(self.grid_size - 1, pos_x + dx))
        next_y = max(0, min(self.grid_size - 1, pos_y + dy))
        
        if (next_x, next_y) in self.walls:
            next_x, next_y = pos_x, pos_y
            
        next_state_idx = next_y * self.grid_size + next_x
        child_node = Node(next_state_idx, parent=node, action=action)
        node.children[action] = child_node
        return child_node

    def evaluate(self, node):
        if node.state_idx == self.target_idx:
            return 1000
            
        pos_x = node.state_idx % self.grid_size
        pos_y = node.state_idx // self.grid_size
        target_x = self.target_idx % self.grid_size
        target_y = self.target_idx // self.grid_size
        
        distance = abs(pos_x - target_x) + abs(pos_y - target_y)
        value = -distance
        
        parent_x = node.parent.state_idx % self.grid_size if node.parent else pos_x
        parent_y = node.parent.state_idx // self.grid_size if node.parent else pos_y
        
        if pos_x == parent_x and pos_y == parent_y:
            self.dead_end_simulations += 1
            value -= 50
            
        if (pos_x, pos_y) == (2,2) or (pos_x, pos_y) == (2,1):
            self.dead_end_simulations += 1
            value -= 100 
            
        return value

    def backpropagate(self, node, value):
        while node is not None:
            node.N += 1
            node.W += value
            node = node.parent
