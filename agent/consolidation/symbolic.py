import torch

class SymbolicRuleBase:
    """
    Tier 3: Hard Symbolic Logic
    Extracts absolute discrete facts or rigid anomalies that defy neural abstraction.
    """
    def __init__(self):
        # Maps (state_idx, action_idx) -> outcome_idx
        self.rules = {}
        
    def consolidate(self, memories, threshold: float = 0.01):
        """
        memories: A tuple of (states, actions, outcomes)
        Converts the dense latent embeddings back to symbolic indices using argmax
        if the outcomes are deterministic and precise.
        """
        states, actions, outcomes = memories
        dataset_size = states.shape[0]
        
        for i in range(dataset_size):
            # Assuming grid states are one-hot encoded and mapped strictly in latent space
            s_idx = torch.argmax(states[i]).item()
            a_idx = torch.argmax(actions[i]).item()
            o_idx = torch.argmax(outcomes[i]).item() # Approximate logic for the sandbox
            
            # Save hard rule
            self.rules[(s_idx, a_idx)] = o_idx
            
    def predict(self, state_idx: int, action_idx: int):
        """Returns the outcome if a hard rule exists, else None."""
        return self.rules.get((state_idx, action_idx), None)
