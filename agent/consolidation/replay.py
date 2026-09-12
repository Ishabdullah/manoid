import torch
import torch.nn as nn
import torch.optim as optim

class BaseIntuitionReplay:
    """
    Tier 1: Base Intuition (Neural Replay)
    Runs standard backpropagation on the WorldModel using a random sample of
    highly confident memories to ingrain basic spatial correlations.
    """
    def __init__(self, world_model, learning_rate: float = 1e-4):
        self.world_model = world_model
        # Use a small learning rate to avoid catastrophic forgetting during offline replay
        self.optimizer = optim.Adam(self.world_model.parameters(), lr=learning_rate)
        
    def consolidate(self, memories, epochs: int = 5, batch_size: int = 32):
        """
        memories: A tuple of (states, actions, outcomes) tensors
        """
        states, actions, outcomes = memories
        dataset_size = states.shape[0]
        
        if dataset_size == 0:
            return
            
        self.world_model.train()
        for epoch in range(epochs):
            permutation = torch.randperm(dataset_size)
            
            for i in range(0, dataset_size, batch_size):
                indices = permutation[i:i+batch_size]
                
                b_states = states[indices]
                b_actions = actions[indices]
                b_outcomes = outcomes[indices]
                
                self.optimizer.zero_grad()
                
                # Predict ensemble next states
                z_preds = self.world_model.predict_ensemble(b_states, b_actions)
                
                # Loss is sum of MSE across all predictor heads
                loss = 0.0
                for pred in z_preds:
                    loss += torch.nn.functional.mse_loss(pred, b_outcomes)
                    
                loss.backward()
                self.optimizer.step()
