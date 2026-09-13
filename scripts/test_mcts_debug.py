import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.core_model.world_model import WorldModel
from agent.consolidation.symbolic import SymbolicRuleBase
from scripts.mcts_planner import MCTSPlanner

world_model = WorldModel()
tier3 = SymbolicRuleBase()

class MCTSPlannerDebug(MCTSPlanner):
    def search(self, root_state_idx, num_simulations=50):
        from scripts.mcts_planner import Node
        root = Node(root_state_idx)
        for sim in range(num_simulations):
            node = root
            while len(node.untried_actions) == 0 and len(node.children) > 0:
                node = self.select(node)
            if len(node.untried_actions) > 0:
                node = self.expand(node)
            value = self.evaluate(node)
            self.backpropagate(node, value)
        
        for a, child in root.children.items():
            print(f"Action {a}: N={child.N}, W={child.W}, Q={child.W/child.N if child.N > 0 else 0}")
        best_action = max(root.children.items(), key=lambda item: item[1].N)[0]
        print(f"Best Action: {best_action}")
        return best_action

debug_planner = MCTSPlannerDebug(world_model, tier3, target_idx=24, grid_size=5)
debug_planner.search(4, num_simulations=50)
