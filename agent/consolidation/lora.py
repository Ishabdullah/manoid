import torch
import torch.nn as nn
import torch.optim as optim

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation wrapper for a Linear layer.
    """
    def __init__(self, linear_layer: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.alpha = alpha
        
        # Original frozen weights
        self.linear = linear_layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
            
        # LoRA A and B matrices
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, self.rank))
        self.lora_B = nn.Parameter(torch.zeros(self.rank, self.out_features))
        self.scaling = self.alpha / self.rank
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B) # B starts at 0 so initial adapter output is 0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out

class AnalogicalReasoningLoRA:
    """
    Tier 2: Analogical Reasoning
    Trains modular LoRA adapters on abstract structural dynamics (e.g. Resistance/Boundaries).
    """
    def __init__(self, world_model):
        self.world_model = world_model
        
    def consolidate(self, concept_name: str, memories, rank: int = 4, epochs: int = 10, lr: float = 1e-3):
        states, actions, outcomes = memories
        if states.shape[0] == 0:
            return
            
        # Apply LoRA specifically for this concept
        self.world_model.apply_lora_adapter(concept_name, rank=rank)
        self.world_model.set_active_lora(concept_name)
        
        # Only train the newly added LoRA parameters
        trainable_params = [p for p in self.world_model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=lr)
        
        # Implement learning rate decay to prevent "adapter shocks"
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        self.world_model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            z_preds = self.world_model.predict_ensemble(states, actions)
            
            loss = 0.0
            for pred in z_preds:
                loss += torch.nn.functional.mse_loss(pred, outcomes)
                
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        self.world_model.eval()
        # Deactivate so base intuition is preserved until adapter is explicitly requested
        self.world_model.deactivate_lora()
