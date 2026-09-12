import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Maps raw environment observations into a dense latent state.
    """
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Predictor(nn.Module):
    """
    Predicts the next latent state given the current latent state and an action vector.
    """
    def __init__(self, latent_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, latent_dim)
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, action], dim=-1)
        return self.net(x)

class WorldModel(nn.Module):
    """
    The Core Intelligence (Phase 1 & 4).
    A ~50M parameter Joint Embedding Predictive Architecture using an Ensemble.
    The Ensemble allows us to natively measure Epistemic Uncertainty (Curiosity Option B).
    """
    def __init__(self, input_dim: int = 256, action_dim: int = 16, latent_dim: int = 512, num_heads: int = 3):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        # Ensemble of Predictors to solve Noisy TV trap
        self.predictors = nn.ModuleList([Predictor(latent_dim, action_dim) for _ in range(num_heads)])
        
        # Track original architecture modules for LoRA swapping
        self._lora_adapters = nn.ModuleDict()
        self._active_lora = None

    def apply_lora_adapter(self, concept_name: str, rank: int = 4):
        """
        Creates a LoRA adapter for the given concept.
        We apply LoRA to the final linear layer of each predictor head.
        """
        if concept_name in self._lora_adapters:
            return
            
        import copy
        from agent.consolidation.lora import LoRALinear
        
        concept_predictors = nn.ModuleList()
        for p in self.predictors:
            # Copy the predictor to maintain base weights securely
            p_copy = copy.deepcopy(p)
            
            # Find the last linear layer in the sequential block
            # For our architecture, it's net[6]
            original_linear = p_copy.net[6]
            
            # Replace with LoRALinear
            p_copy.net[6] = LoRALinear(original_linear, rank=rank)
            concept_predictors.append(p_copy)
            
        self._lora_adapters[concept_name] = concept_predictors

    def merge_adapters(self, active_concepts: list, new_concept_name: str = "merged_consensus"):
        """
        Task Vector Merging (Consensus Pruning).
        Resolves sign conflicts when multiple adapters are applied simultaneously.
        """
        if len(active_concepts) < 2:
            return
            
        import copy
        
        # Create a new set of predictors for the merged concept
        merged_predictors = nn.ModuleList()
        
        for p_idx, p_base in enumerate(self.predictors):
            p_merged = copy.deepcopy(p_base)
            original_linear = p_merged.net[6]
            
            # We will compute the consensus delta_W explicitly
            delta_w_sum = torch.zeros_like(original_linear.weight)
            sign_consensus = torch.zeros_like(original_linear.weight)
            
            for concept in active_concepts:
                adapter_net = self._lora_adapters[concept][p_idx].net[6] # LoRALinear
                A = adapter_net.lora_A
                B = adapter_net.lora_B
                scaling = adapter_net.scaling
                
                # PyTorch linear layer weight shape is (out_features, in_features)
                # x @ A @ B is equivalent to x @ (A @ B)
                # Thus the effective weight delta is ((A @ B) * scaling).t()
                delta_w = (A @ B).t() * scaling 
                
                delta_w_sum += delta_w
                sign_consensus += torch.sign(delta_w)
                
            # Consensus pruning: If signs conflict (sum of signs is close to 0), we prune the weight.
            # e.g., if 2 adapters agree, abs(sign_consensus) == 2. If they conflict, it's 0.
            consensus_mask = (torch.abs(sign_consensus) == len(active_concepts)).float()
            
            # Apply mask to the sum to nullify conflicted weights (they cancel out cleanly instead of adding noise)
            pruned_delta_w = delta_w_sum * consensus_mask
            
            # Inject the pruned_delta_w directly into a static base weight for this merged predictor
            # This avoids tracking multiple active adapters dynamically during the forward pass.
            merged_linear = nn.Linear(original_linear.in_features, original_linear.out_features)
            merged_linear.weight = nn.Parameter(original_linear.weight + pruned_delta_w)
            if original_linear.bias is not None:
                merged_linear.bias = nn.Parameter(original_linear.bias)
                
            # Freeze it
            merged_linear.weight.requires_grad = False
            if merged_linear.bias is not None:
                merged_linear.bias.requires_grad = False
                
            p_merged.net[6] = merged_linear
            merged_predictors.append(p_merged)
            
        self._lora_adapters[new_concept_name] = merged_predictors
        
    def set_active_lora(self, concept_name: str):
        """Swaps the active predictors to the LoRA-adapted ones."""
        if concept_name in self._lora_adapters:
            # We temporarily swap self.predictors (though nn.ModuleList behavior requires care)
            # To be safe and clean, we'll just redirect prediction through them
            self._active_lora = concept_name
            
    def deactivate_lora(self):
        """Returns to the base intuition model."""
        self._active_lora = None
        
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from raw observations."""
        return self.encoder(obs)
        
    def predict_ensemble(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns predictions from all heads. Shape: (num_heads, batch, latent_dim)"""
        if self._active_lora is not None:
            active_preds = self._lora_adapters[self._active_lora]
            return torch.stack([p(z, action) for p in active_preds])
        return torch.stack([p(z, action) for p in self.predictors])
        
    def predict_next(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Generate mean prediction for the next state."""
        return self.predict_ensemble(z, action).mean(dim=0)
        
    def compute_epistemic_uncertainty(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Curiosity metric: How much do the heads disagree?
        Returns the variance across the ensemble, averaged over the latent dimensions.
        """
        preds = self.predict_ensemble(z, action)
        return preds.var(dim=0).mean(dim=-1)

    def compute_surprise(self, z_pred: torch.Tensor, z_actual: torch.Tensor) -> torch.Tensor:
        """
        Standard forward prediction error (MSE).
        """
        return F.mse_loss(z_pred, z_actual, reduction='none').mean(dim=-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
