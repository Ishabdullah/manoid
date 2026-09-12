import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.memory.episodic import EpisodicMemory
from agent.environment.symbolic import SymbolicGridEnv
from agent.core_model.world_model import WorldModel
from agent.consolidation.pipeline import ConsolidationPipeline

def test_consolidation():
    print("--- Phase 6: Multi-Tiered Consolidation Verifier ---")
    
    env = SymbolicGridEnv(grid_size=5)
    world_model = WorldModel()
    memory = EpisodicMemory(max_size=100)
    pipeline = ConsolidationPipeline(world_model)
    
    # 1. Generate Fake High-Confidence Memories for all 3 Tiers
    # Tier 1: Normal Movement (0 -> Action 1 (Right) -> 1)
    env.reset()
    s0 = world_model.encode(env._get_state()).detach()
    a1 = env.encode_action(1)
    env.step(1)
    o1 = world_model.encode(env._get_state()).detach()
    memory.store(s0, a1, o1, torch.tensor([[0.95]]))
    
    # Tier 2: Resistance / Boundary (4 -> Action 1 (Right) -> 4)
    env.pos_x = 4
    env.pos_y = 0
    s4 = world_model.encode(env._get_state()).detach()
    a_bound = env.encode_action(1)
    env.step(1) # Hits right wall, stays at 4
    o4 = world_model.encode(env._get_state()).detach()
    memory.store(s4, a_bound, o4, torch.tensor([[0.95]]))
    
    # Tier 3: Magic Teleportation Anomaly (24 -> Action 0 -> 0)
    # This defies normal grid physics
    env.pos_x = 4
    env.pos_y = 4
    s24 = world_model.encode(env._get_state()).detach()
    a_magic = env.encode_action(0)
    # We fake the outcome tensor to be 0
    fake_obs = torch.zeros(1, 256)
    fake_obs[0, 0] = 1.0
    o_magic = world_model.encode(fake_obs).detach()
    memory.store(s24, a_magic, o_magic, torch.tensor([[0.95]]))
    
    # 2. Measure Initial MSE (Pre-Sleep)
    world_model.eval()
    with torch.no_grad():
        initial_t1_mse = torch.nn.functional.mse_loss(world_model.predict_next(s0, a1), o1).item()
        initial_t2_mse = torch.nn.functional.mse_loss(world_model.predict_next(s4, a_bound), o4).item()
        
    print(f"Pre-Sleep Tier 1 (Normal Movement) MSE: {initial_t1_mse:.6f}")
    print(f"Pre-Sleep Tier 2 (Boundary Collision) MSE: {initial_t2_mse:.6f}")
    
    # 3. Run Sleep Cycle (Consolidation)
    print("\nInitiating Sleep Cycle / Consolidation...")
    pipeline.consolidate_sleep_cycle(memory)
    
    # 4. Measure Post-Sleep Results
    print("\nWaking up. Verifying Consolidation...")
    
    # Tier 1 check
    world_model.eval()
    with torch.no_grad():
        post_t1_mse = torch.nn.functional.mse_loss(world_model.predict_next(s0, a1), o1).item()
    print(f"Post-Sleep Tier 1 MSE: {post_t1_mse:.6f}")
    assert post_t1_mse < initial_t1_mse, "Tier 1 Base Intuition failed to improve!"
    
    # Tier 2 check (Requires activating the LoRA adapter)
    print("\nEvaluating Base Model on Boundary Collision (Should still be high because it's an anomaly):")
    with torch.no_grad():
        base_t2_mse = torch.nn.functional.mse_loss(world_model.predict_next(s4, a_bound), o4).item()
    print(f"Base Model Tier 2 MSE: {base_t2_mse:.6f}")
    
    print("\nActivating 'Resistance' LoRA Adapter for Analogical Reasoning...")
    world_model.set_active_lora("Resistance")
    with torch.no_grad():
        lora_t2_mse = torch.nn.functional.mse_loss(world_model.predict_next(s4, a_bound), o4).item()
    print(f"LoRA Adapter Tier 2 MSE: {lora_t2_mse:.6f}")
    assert lora_t2_mse < base_t2_mse, "LoRA Adapter failed to learn the boundary resistance concept!"
    world_model.deactivate_lora()
    
    # Tier 3 check
    print("\nChecking Symbolic Extraction for Anomaly...")
    magic_s_idx = torch.argmax(s24).item()
    magic_o_idx = torch.argmax(o_magic).item()
    magic_rule_outcome = pipeline.tier3_symbolic.predict(magic_s_idx, 0)
    print(f"Symbolic Rule Dictionary returns outcome index: {magic_rule_outcome}")
    assert magic_rule_outcome == magic_o_idx, "Tier 3 Symbolic logic failed to extract the anomaly!"
    
    print("\n--- Live Verification Success! ---")

if __name__ == "__main__":
    test_consolidation()
