import torch
import random

class CuriosityPolicy:
    """
    Action selection weighted by epistemic uncertainty (Option B).
    The agent prefers actions where the WorldModel's ensemble of predictors disagree.
    This naturally solves the 'Noisy TV' trap because the ensemble quickly agrees 
    on non-linear but deterministic boundaries.
    """
    def __init__(self, num_actions: int = 4, action_dim: int = 16):
        self.num_actions = num_actions
        self.action_dim = action_dim
        
    def select_action(self, world_model, z: torch.Tensor, epsilon: float = 0.05) -> int:
        """
        Selects an action based on highest ensemble variance.
        """
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
            
        variances = []
        
        # Don't track gradients during action selection
        world_model.eval()
        with torch.no_grad():
            for a_idx in range(self.num_actions):
                a_tensor = torch.zeros(1, self.action_dim, device=z.device)
                a_tensor[0, a_idx] = 1.0
                
                # Ask the world model how uncertain it is about this action
                var = world_model.compute_epistemic_uncertainty(z, a_tensor)
                variances.append(var.item())
        world_model.train()
                
        max_var = max(variances)
        
        # Boredom Threshold: If we perfectly understand the state (uncertainty < 0.01),
        # stop being curious and exploit what we know (which without extrinsic reward just means random/default).
        if max_var < 0.01:
            return random.randint(0, self.num_actions - 1)
            
        # Return action with MAXIMUM uncertainty/disagreement
        return int(torch.argmax(torch.tensor(variances)).item())
