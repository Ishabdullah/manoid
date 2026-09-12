import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.core_model.world_model import WorldModel
from agent.consolidation.symbolic import SymbolicRuleBase
from scripts.mcts_planner import MCTSPlanner

def run_trap_test():
    print("=== Phase 8: MCTS Trap Test Verification ===")
    print("Environment: 5x5 Gridworld with U-Shaped Trap")
    print("Start: (0,0) | Goal: (4,4)")
    print("Walls: (1,1), (2,1), (3,1), (1,2), (3,2)")
    print("--------------------------------------------------")
    
    world_model = WorldModel()
    tier3_symbolic = SymbolicRuleBase()
    
    walls = [(1,1), (2,1), (3,1), (1,2), (3,2)]
    planner = MCTSPlanner(world_model, tier3_symbolic, target_idx=24, grid_size=5, walls=walls)
    
    pos_x, pos_y = 0, 0
    path = [(pos_x, pos_y)]
    action_names = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}
    
    step = 0
    while (pos_x, pos_y) != (4,4) and step < 25:
        step += 1
        s_idx = pos_y * 5 + pos_x
        
        # Reset simulation counters
        planner.dead_end_simulations = 0
        a_idx = planner.search(s_idx, num_simulations=50)
        
        print(f"[Step {step}] At ({pos_x},{pos_y})")
        print(f"  -> Simulated dead-ends / wall hits: {planner.dead_end_simulations}")
        print(f"  -> Chose Action: {action_names[a_idx]}")
        
        dx, dy = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}[a_idx]
        next_x = max(0, min(4, pos_x + dx))
        next_y = max(0, min(4, pos_y + dy))
        
        if (next_x, next_y) not in walls:
            pos_x, pos_y = next_x, next_y
            
        path.append((pos_x, pos_y))
        
    print("\n[Final Result]")
    if (pos_x, pos_y) == (4,4):
        print("SUCCESS! Reached goal (4,4) avoiding the trap.")
    else:
        print("FAILED to reach goal.")
    print("Physical Path Taken:", path)

if __name__ == "__main__":
    run_trap_test()
