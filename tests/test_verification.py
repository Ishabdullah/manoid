import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.memory.episodic import EpisodicMemory
from agent.environment.symbolic import SymbolicGridEnv
from agent.core_model.world_model import WorldModel

def verify_falsification():
    print("--- Phase 5: Live Verifier ---")
    
    env = SymbolicGridEnv(grid_size=5)
    world_model = WorldModel()
    memory = EpisodicMemory(max_size=100)
    
    # 1. Prepare known environment setup
    obs = env.reset() # starts at index 0
    z_start = world_model.encode(obs).detach()
    
    # Action 1 (Right) -> normally leads to index 1
    action_1_tensor = env.encode_action(1)
    # Action 2 (Down) -> normally leads to index 5 in a 5x5 grid
    action_2_tensor = env.encode_action(2)
    
    # True outcome (Right)
    dummy_obs_1, _, _ = env.step(1)
    env.reset() # Reset back to 0
    z_true_outcome = world_model.encode(dummy_obs_1).detach()
    
    # False outcome (Suppose someone injected a memory that Down from 0 leads to index 24!)
    fake_obs = torch.zeros(1, 256)
    fake_obs[0, 24] = 1.0
    z_false_outcome = world_model.encode(fake_obs).detach()
    
    # 2. Inject hypotheses into Episodic Memory (Initial confidence 0.5)
    print("Injecting Hypothesis A (True) and Hypothesis B (False) with confidence 0.5")
    memory.store(z_start, action_1_tensor, z_true_outcome, torch.tensor([[0.5]]))
    memory.store(z_start, action_2_tensor, z_false_outcome, torch.tensor([[0.5]]))
    
    # Check what is stored
    _, conf_A, _, idx_A = memory.retrieve(z_start, action_1_tensor)
    _, conf_B, _, idx_B = memory.retrieve(z_start, action_2_tensor)
    
    assert conf_A.item() == 0.5
    assert conf_B.item() == 0.5
    
    # 3. Agent Testing Loop (Corroboration)
    print("\nAgent tests Hypothesis A (Right)...")
    # Take real action
    real_next_obs_A, _, _ = env.step(1)
    z_real_next_A = world_model.encode(real_next_obs_A).detach()
    
    # Compare real to retrieved expectation (MSE)
    retrieved_out_A, _, _, ret_idx_A = memory.retrieve(z_start, action_1_tensor)
    mse_A = torch.nn.functional.mse_loss(z_real_next_A, retrieved_out_A.squeeze(1)).item()
    print(f"MSE between reality and Hypothesis A expectation: {mse_A:.6f}")
    
    # Corroborate
    if mse_A < 0.01:
        print("Hypothesis A corroborated! Confidence +0.45")
        memory.update_confidence(ret_idx_A, 0.45)
    else:
        print("Hypothesis A falsified! Confidence -0.45")
        memory.update_confidence(ret_idx_A, -0.45)
        
    env.reset() # Reset back to 0
    
    print("\nAgent tests Hypothesis B (Down)...")
    # Take real action
    real_next_obs_B, _, _ = env.step(2)
    z_real_next_B = world_model.encode(real_next_obs_B).detach()
    
    # Compare real to retrieved expectation
    retrieved_out_B, _, _, ret_idx_B = memory.retrieve(z_start, action_2_tensor)
    mse_B = torch.nn.functional.mse_loss(z_real_next_B, retrieved_out_B.squeeze(1)).item()
    print(f"MSE between reality and Hypothesis B expectation: {mse_B:.6f}")
    
    if mse_B < 0.01:
        print("Hypothesis B corroborated! Confidence +0.45")
        memory.update_confidence(ret_idx_B, 0.45)
    else:
        print("Hypothesis B falsified! Confidence -0.45")
        memory.update_confidence(ret_idx_B, -0.45)
        
    # 4. Final verification
    _, final_conf_A, _, _ = memory.retrieve(z_start, action_1_tensor)
    _, final_conf_B, _, _ = memory.retrieve(z_start, action_2_tensor)
    
    print(f"\nFinal Confidence Hypothesis A (True): {final_conf_A.item():.2f}")
    print(f"Final Confidence Hypothesis B (False): {final_conf_B.item():.2f}")
    
    print(f"Will Hypothesis A be sent to consolidation? {'Yes' if final_conf_A.item() > 0.9 else 'No'}")
    print(f"Will Hypothesis B be sent to consolidation? {'Yes' if final_conf_B.item() > 0.9 else 'No'}")
    
    assert final_conf_A.item() >= 0.9, "True hypothesis was not corroborated properly!"
    assert final_conf_B.item() <= 0.1, "False hypothesis was not falsified properly!"
    
    print("\n--- Live Verification Success! ---")

if __name__ == "__main__":
    verify_falsification()
