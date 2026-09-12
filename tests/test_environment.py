import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.environment.symbolic import SymbolicGridEnv

def verify_environment():
    print("--- Phase 3: Live Verifier ---")
    
    # 1. Initialize environment
    env = SymbolicGridEnv(grid_size=5, output_dim=256, action_dim=16)
    print("Initialized Custom Symbolic Grid Environment (5x5).")
    
    state = env.reset()
    
    # Ground truth tracking check
    # Start: (0,0) -> Flat index 0
    assert torch.argmax(state).item() == 0, "Initial state should be at origin index 0"
    
    # 2. Perform do(X) interventions
    actions_to_take = [
        1, # Right -> expected pos (1, 0) -> idx 1
        2, # Down -> expected pos (1, 1) -> idx 6
        2, # Down -> expected pos (1, 2) -> idx 11
        3, # Left -> expected pos (0, 2) -> idx 10
    ]
    
    expected_indices = [1, 6, 11, 10]
    
    print("\nExecuting do(X) sequence and logging transitions...")
    for step, (a_idx, expected_idx) in enumerate(zip(actions_to_take, expected_indices)):
        action_tensor = env.encode_action(a_idx)
        next_state, _, _ = env.step(a_idx)
        
        # Log transition
        log_str = env.log_transition(state, action_tensor, next_state)
        print(f"Step {step+1}: {log_str}")
        
        # Verify internal tracking matches ground truth
        actual_idx = torch.argmax(next_state).item()
        assert actual_idx == expected_idx, f"Mismatch! Expected pos {expected_idx}, got {actual_idx}"
        
        state = next_state
        
    print("\nWall collision test...")
    # Currently at index 10 (0, 2). Moving Left (3) should hit the wall and stay at 10.
    a_idx = 3
    action_tensor = env.encode_action(a_idx)
    next_state, _, _ = env.step(a_idx)
    log_str = env.log_transition(state, action_tensor, next_state)
    print(log_str)
    
    actual_idx = torch.argmax(next_state).item()
    assert actual_idx == 10, "Agent passed through the wall!"
    
    print("\n--- Live Verification Success! Internal state tracking perfectly matches environmental ground truth. ---")

if __name__ == "__main__":
    verify_environment()
