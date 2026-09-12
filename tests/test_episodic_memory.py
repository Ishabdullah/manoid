import torch
import sys
import os

# Add parent directory to path so we can import agent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.memory.episodic import EpisodicMemory

def verify_episodic_memory():
    print("--- Phase 2: Live Verifier ---")
    
    # Initialize Memory
    mem = EpisodicMemory(state_dim=512, action_dim=16, max_size=100)
    print("Initialized PyTorch Tensor Buffer Episodic Cache.")
    
    # 1. Define a novel fact (state, action, outcome, confidence)
    novel_state = torch.randn(1, 512)
    novel_action = torch.randn(1, 16)
    novel_outcome = torch.randn(1, 512)
    novel_confidence = torch.tensor([[0.99]])
    
    # 2. Query before storing (should be empty)
    print("\nQuerying cache before storing...")
    out, conf, sim = mem.retrieve(novel_state, novel_action)
    print(f"Result before store: {out}")
    assert out is None, "Memory should be empty initially!"
    
    # 3. Store the novel fact
    print("\nStoring novel fact [state, action, outcome, confidence]...")
    mem.store(novel_state, novel_action, novel_outcome, novel_confidence)
    print(f"Memory size after store: {mem.size}")
    
    # 4. Query again (should retrieve instantly with high similarity)
    print("\nQuerying cache after storing...")
    ret_out, ret_conf, sim = mem.retrieve(novel_state, novel_action)
    
    print(f"Similarity score: {sim.item():.4f}")
    print(f"Retrieved confidence: {ret_conf.item():.4f}")
    
    # Verify exact match
    mse = torch.nn.functional.mse_loss(ret_out.squeeze(1), novel_outcome)
    print(f"MSE between stored and retrieved outcome: {mse.item():.6f}")
    
    assert sim.item() > 0.99, "Similarity should be ~1.0 for exact query"
    assert mse.item() < 1e-5, "Retrieved outcome must exactly match stored outcome"
    
    print("\n--- Live Verification Success! ---")

if __name__ == "__main__":
    verify_episodic_memory()
