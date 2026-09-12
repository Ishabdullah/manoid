import torch

from agent.consolidation.replay import BaseIntuitionReplay
from agent.consolidation.lora import AnalogicalReasoningLoRA
from agent.consolidation.symbolic import SymbolicRuleBase

class ConsolidationPipeline:
    """
    The Orchestrator for Phase 6.
    Triages high-confidence episodic memories into the 3-Tier Cognitive Pipeline.
    """
    def __init__(self, world_model):
        self.world_model = world_model
        
        self.tier1_replay = BaseIntuitionReplay(world_model)
        self.tier2_lora = AnalogicalReasoningLoRA(world_model)
        self.tier3_symbolic = SymbolicRuleBase()
        
    def consolidate_sleep_cycle(self, episodic_memory, mse_history=None):
        """
        Runs the full triage and consolidation offline.
        """
        if episodic_memory.size == 0:
            return
            
        # 1. Filter highly confident memories
        valid_indices = (episodic_memory.confidences[:episodic_memory.size] > 0.9).squeeze(-1)
        if not valid_indices.any():
            return
            
        states = episodic_memory.states[:episodic_memory.size][valid_indices]
        actions = episodic_memory.actions[:episodic_memory.size][valid_indices]
        outcomes = episodic_memory.outcomes[:episodic_memory.size][valid_indices]
        
        # 2. Triage Logic
        # For our 5x5 sandbox, we define:
        # Tier 1 (Base): Normal spatial transitions where state != outcome
        # Tier 2 (LoRA): Boundary collisions where state == outcome (Concept: "Resistance")
        # Tier 3 (Symbolic): Absolute anomalies. Here we'll artificially flag anomalies in testing.
        
        tier1_indices = []
        tier2_indices = []
        tier3_indices = []
        
        for i in range(states.shape[0]):
            # Use cosine similarity to detect if state didn't change (Boundary Resistance)
            sim = torch.nn.functional.cosine_similarity(states[i:i+1], outcomes[i:i+1]).item()
            a_idx = torch.argmax(actions[i]).item()
            
            if sim > 0.99:
                # Abstract Concept: Boundary Collision / "Resistance" (state == outcome)
                tier2_indices.append(i)
            elif a_idx == 0:
                # Artificial "Magic Tile" teleportation anomaly for testing Tier 3
                tier3_indices.append(i)
            else:
                # Normal movement
                tier1_indices.append(i)
                
        # Dynamic Plasticity Logic for Hybrid Router
        skip_tier2 = False
        if mse_history is None or len(mse_history) < 3:
            # Not enough history to evaluate a twice-up regression. Default to TRAIN.
            print("  [Manoid] Plasticity: Insufficient history. Defaulting to Tier 2 consolidation.")
        else:
            prev2_mse = mse_history[-3]
            prev1_mse = mse_history[-2]
            current_mse = mse_history[-1]
            
            if current_mse > prev1_mse and prev1_mse > prev2_mse:
                print("  [Manoid] Plasticity: Twice-up regression detected. Triggering Tier 2 consolidation.")
                pass # Condition A (Twice-Up Regression): TRAIN
            else:
                skip_tier2 = True # Condition B (Stable, Improving, or Noise): SKIP
                print("  [Manoid] Plasticity: Error stable/recovering. Skipping Tier 2 consolidation.")
                
        if skip_tier2:
            tier1_indices.extend(tier2_indices)
            tier2_indices = []
            
        # Cap Tier 2 Routing to 60% of the batch to prevent "Everything is a Nail" problem
        batch_size = states.shape[0]
        max_tier2_size = int(0.6 * batch_size)
        if len(tier2_indices) > max_tier2_size:
            # Force the excess into Tier 1 (Replay) to maintain base instincts
            excess_indices = tier2_indices[max_tier2_size:]
            tier1_indices.extend(excess_indices)
            tier2_indices = tier2_indices[:max_tier2_size]
        
        # Tier 1: Base Intuition
        if tier1_indices:
            print(f"Routing {len(tier1_indices)} experiences to Tier 1 (Replay)")
            t1_memories = (states[tier1_indices], actions[tier1_indices], outcomes[tier1_indices])
            self.tier1_replay.consolidate(t1_memories, epochs=50)
            
        # Tier 2: Analogical Reasoning (LoRA)
        if tier2_indices:
            print(f"Routing {len(tier2_indices)} experiences to Tier 2 (LoRA Adapter: 'Resistance')")
            t2_memories = (states[tier2_indices], actions[tier2_indices], outcomes[tier2_indices])
            self.tier2_lora.consolidate("Resistance", t2_memories, rank=4, epochs=50)
            
        # Tier 3: Hard Symbolic Rules
        if tier3_indices:
            print(f"Routing {len(tier3_indices)} experiences to Tier 3 (Symbolic Rule)")
            t3_memories = (states[tier3_indices], actions[tier3_indices], outcomes[tier3_indices])
            self.tier3_symbolic.consolidate(t3_memories)
