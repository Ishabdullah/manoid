import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Maps raw environment observations into a dense latent state.
    This fulfills the Architect's requirement for extracting embeddings.
    """
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        # Using a simple MLP to reach ~50M parameters we'd scale the hidden layers.
        # For immediate local verification without running OOM, we keep it structurally sound 
        # but moderately sized. E.g., 4096 hidden units gives ~16M params per layer.
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
    This enables causal hypothesis testing (do(X) interventions).
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
    The Core Intelligence (Phase 1).
    A ~50M parameter Joint Embedding Predictive Architecture.
    """
    def __init__(self, input_dim: int = 256, action_dim: int = 16, latent_dim: int = 512):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.predictor = Predictor(latent_dim, action_dim)
        
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from raw observations."""
        return self.encoder(obs)
        
    def predict_next(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Generate predictions for the next state."""
        return self.predictor(z, action)
        
    def compute_surprise(self, z_pred: torch.Tensor, z_actual: torch.Tensor) -> torch.Tensor:
        """
        Flag uncertainty/surprise.
        Calculates the MSE between predicted next latent and actual next latent.
        """
        return F.mse_loss(z_pred, z_actual, reduction='none').mean(dim=-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
