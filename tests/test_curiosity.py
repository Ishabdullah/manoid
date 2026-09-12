import torch
import torch.optim as optim
import sys
import os
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.environment.symbolic import SymbolicGridEnv
from agent.core_model.world_model import WorldModel
from agent.policy.curiosity import CuriosityPolicy

def run_agent_loop(policy_type: str, steps: int = 250):
    env = SymbolicGridEnv(grid_size=5)
    world_model = WorldModel(num_heads=3) # Use 3-head ensemble
    wm_optimizer = optim.Adam(world_model.parameters(), lr=5e-4) # lower LR for stability
    
    curiosity = CuriosityPolicy() if policy_type == "curiosity" else None
    
    obs = env.reset()
    total_surprise = 0.0
    visited_states = set()
    
    for step in range(steps):
        visited_states.add(torch.argmax(obs).item())
        
        # 1. Encode state
        z_t = world_model.encode(obs)
        
        # 2. Select Action
        if policy_type == "curiosity":
            eps = max(0.05, 1.0 - (step / 75.0))
            a_idx = curiosity.select_action(world_model, z_t.detach(), epsilon=eps)
        else:
            # Random policy
            a_idx = random.randint(0, 3)
            
        action_tensor = env.encode_action(a_idx)
        
        # 3. Take environment step
        next_obs, _, _ = env.step(a_idx)
        
        # 4. Encode actual next state
        z_actual_next = world_model.encode(next_obs)
        
        # 5. Predict ensemble next states and compute loss
        wm_optimizer.zero_grad()
        
        # Forward pass for loss
        z_preds = world_model.predict_ensemble(z_t, action_tensor)
        
        # Loss is the sum of MSE across all predictor heads
        loss = 0.0
        for pred in z_preds:
            # Each head must predict the actual next encoded state
            loss += torch.nn.functional.mse_loss(pred, z_actual_next.detach())
            
        loss.backward()
        wm_optimizer.step()
        
        # Track total surprise (mean MSE) for logging
        total_surprise += (loss.item() / 3.0)
            
        obs = next_obs
        
    # Evaluate final uncertainty across all 25 grid states.
    final_uncertainty = 0.0
    with torch.no_grad():
        for s_idx in range(25):
            test_obs = torch.zeros(1, 256)
            test_obs[0, s_idx] = 1.0
            z_test = world_model.encode(test_obs)
            for a_idx in range(4):
                test_a = env.encode_action(a_idx)
                # Compute pseudo-next state for evaluation
                dummy_next, _, _ = env.step(a_idx)
                z_test_next = world_model.encode(dummy_next)
                
                # Check how well the ensemble predicts this transition
                z_preds = world_model.predict_ensemble(z_test, test_a)
                for pred in z_preds:
                    surp = world_model.compute_surprise(pred, z_test_next).item()
                    final_uncertainty += surp
                
    return total_surprise, len(visited_states), final_uncertainty

def verify_curiosity():
    print("--- Phase 4: Live Verifier ---")
    
    # Run Random Agent
    torch.manual_seed(42)
    random.seed(42)
    print("Running Random Exploration Agent (250 steps)...")
    rand_surprise, rand_visited, rand_uncert = run_agent_loop(policy_type="random", steps=250)
    print(f"Random Agent -> Total Surprise: {rand_surprise:.4f}, Unique States: {rand_visited}, Final Grid Uncertainty: {rand_uncert:.4f}")
    
    # Run Curiosity Agent
    torch.manual_seed(42)
    random.seed(42)
    print("\nRunning Curiosity-Driven Agent (250 steps)...")
    cur_surprise, cur_visited, cur_uncert = run_agent_loop(policy_type="curiosity", steps=250)
    print(f"Curiosity Agent -> Total Surprise: {cur_surprise:.4f}, Unique States: {cur_visited}, Final Grid Uncertainty: {cur_uncert:.4f}")
    
    print(f"Did Curiosity cover the entire state space better than Random? {'Yes' if cur_visited > rand_visited else 'No'}")
    
    # Curiosity ensures it finds all 25 states, random often misses some corners
    assert cur_visited == 25, "Curiosity failed to explore the full grid!"
    assert cur_visited >= rand_visited, "Curiosity agent failed to explore as well as random!"
    
    print("\n--- Live Verification Success! ---")

if __name__ == "__main__":
    verify_curiosity()
