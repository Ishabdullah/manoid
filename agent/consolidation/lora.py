"""
lora.py — Geodesic LoRA: Spherical Weight-Matrix Rotation
==========================================================
Replaces standard additive LoRA (W' = W + AB) with a spherical geodesic
rotation of weight matrices in the complex weight space.

Mathematical basis
------------------
Given a complex weight matrix W ∈ ℂ^{out×in}, we parameterise an
adaptation as a **rotation along the geodesic** from W to a target
direction T on the complex unit sphere:

    W'(θ) = cos(θ)·W̃ + sin(θ)·R

where:
    W̃ = W / ‖W‖_F         (unit Frobenius-normalised base weight)
    R  = low-rank tangent component perpendicular to W̃
    θ  ∈ [0, π/2]          (geodesic angle, learned parameter)

The tangent component R is parameterised by two low-rank cfloat matrices:
    R = B_cfloat @ A_cfloat   (rank-r, stored in cfloat)

initialised so that θ=0 recovers W exactly (zero adaptation at init).

This has two advantages over additive LoRA:
1. The weight matrix always remains on a sphere of radius ‖W‖_F, 
   preventing catastrophic magnitude growth during fine-tuning.
2. The geodesic path is the shortest path on the weight manifold, 
   minimising the "distance" from the base model for a given adaptation.

GeodesicLoRALinear wraps a ComplexLinear from world_model.py.

AnalogicalReasoningLoRA  (updated)
    Now trains GeodesicLoRALinear adapters on complex latent states.
    Loss is computed in the complex plane: |z_pred - z_target|².
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

PI2 = math.pi ** 2


# ---------------------------------------------------------------------------
# Backward-compatible real LoRALinear (used by ChessValueModelPhase14)
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    Standard Low-Rank Adaptation wrapper for a real nn.Linear layer.
    Kept for backward compatibility with ChessValueModelPhase14.
    """

    def __init__(self, linear_layer: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.in_features  = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank    = rank
        self.alpha   = alpha

        self.linear  = linear_layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.lora_A  = nn.Parameter(torch.zeros(self.in_features, rank))
        self.lora_B  = nn.Parameter(torch.zeros(rank, self.out_features))
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out


# ---------------------------------------------------------------------------
# GeodesicLoRALinear (complex weight rotation)
# ---------------------------------------------------------------------------

class GeodesicLoRALinear(nn.Module):
    """
    Geodesic LoRA adapter for ComplexLinear layers.

    Parameterises weight adaptation as a rotation along the shortest
    geodesic on the complex weight hypersphere.

    W'(θ) = cos(θ) · W̃ + sin(θ) · R_normalised

    where:
        W̃          = W / ‖W‖_F   (Frobenius-unit base)
        R           = lora_B @ lora_A   (low-rank cfloat tangent)
        R_orth      = R - (R·W̃†)·W̃     (orthogonal component)
        R_normalised= R_orth / ‖R_orth‖_F
        θ           = geodesic_angle (scalar learnable parameter)

    The actual weight applied during forward is:
        W_eff = (cos(θ)·W̃ + sin(θ)·R_normalised) · ‖W‖_F
    """

    def __init__(self, complex_linear, rank: int = 4):
        """
        complex_linear: instance of ComplexLinear from world_model.py
        rank          : rank of the low-rank tangent matrices
        """
        super().__init__()
        self.in_features  = complex_linear.in_features
        self.out_features = complex_linear.out_features
        self.rank = rank

        # Freeze base weights
        self.weight_r = nn.Parameter(complex_linear.weight_r.data.clone(),
                                     requires_grad=False)
        self.weight_i = nn.Parameter(complex_linear.weight_i.data.clone(),
                                     requires_grad=False)

        if complex_linear.bias_r is not None:
            self.bias_r = nn.Parameter(complex_linear.bias_r.data.clone(),
                                       requires_grad=False)
            self.bias_i = nn.Parameter(complex_linear.bias_i.data.clone(),
                                       requires_grad=False)
        else:
            self.bias_r = self.bias_i = None

        # Learnable geodesic angle — initialised to 0 (identity)
        self.geodesic_angle = nn.Parameter(torch.zeros(1))

        # Low-rank cfloat tangent matrices (stored as real+imag pairs)
        scale = 1.0 / math.sqrt(self.in_features)
        self.lora_A_r = nn.Parameter(
            torch.empty(rank, self.in_features).uniform_(-scale, scale))
        self.lora_A_i = nn.Parameter(
            torch.empty(rank, self.in_features).uniform_(-scale, scale))
        # B starts at zeros so initial rotation is zero
        self.lora_B_r = nn.Parameter(torch.zeros(self.out_features, rank))
        self.lora_B_i = nn.Parameter(torch.zeros(self.out_features, rank))

    def _base_complex_weight(self) -> torch.Tensor:
        """Returns the base weight as a cfloat matrix."""
        return torch.complex(self.weight_r, self.weight_i)   # (out, in)

    def _frobenius_norm(self, W: torch.Tensor) -> torch.Tensor:
        """‖W‖_F for complex matrix W."""
        return W.abs().norm()

    def _low_rank_tangent(self) -> torch.Tensor:
        """
        Compute R = B @ A in ℂ^{out×in} from the low-rank cfloat parameters.
        B: (out, rank), A: (rank, in)
        Complex matmul: (Br+iBi)(Ar+iAi) = (BrAr-BiAi) + i(BrAi+BiAr)
        """
        Br, Bi = self.lora_B_r, self.lora_B_i
        Ar, Ai = self.lora_A_r, self.lora_A_i

        R_r = Br @ Ar - Bi @ Ai   # (out, in)
        R_i = Br @ Ai + Bi @ Ar   # (out, in)
        return torch.complex(R_r, R_i)

    def _effective_weight(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute (W_eff_r, W_eff_i) representing the geodesic-rotated weight.
        """
        W = self._base_complex_weight()                 # (out, in)
        W_norm = self._frobenius_norm(W).clamp(min=1e-8)
        W_unit = W / W_norm                             # unit sphere

        R = self._low_rank_tangent()                    # (out, in)

        # Orthogonalise R against W_unit
        # proj = Re(<R, W_unit>_F) · W_unit  (Frobenius inner product)
        inner = (R * W_unit.conj()).sum()               # complex scalar
        R_orth = R - inner.real * W_unit                # (out, in) cfloat

        R_norm = self._frobenius_norm(R_orth).clamp(min=1e-8)
        R_unit = R_orth / R_norm

        theta = self.geodesic_angle                     # (1,) float
        W_eff = (torch.cos(theta) * W_unit + torch.sin(theta) * R_unit) * W_norm
        return W_eff.real, W_eff.imag

    def get_geodesic_delta(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (Δ_r, Δ_i) = W_eff - W_base as real tensors.
        Used by WorldModel.merge_adapters() for consensus pruning.
        """
        W_eff_r, W_eff_i = self._effective_weight()
        return (W_eff_r - self.weight_r), (W_eff_i - self.weight_i)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_features) cfloat
        returns: (..., out_features) cfloat
        """
        import torch.nn.functional as F
        W_eff_r, W_eff_i = self._effective_weight()

        xr, xi = x.real, x.imag
        out_r = F.linear(xr, W_eff_r) - F.linear(xi, W_eff_i)
        out_i = F.linear(xi, W_eff_r) + F.linear(xr, W_eff_i)

        if self.bias_r is not None:
            out_r = out_r + self.bias_r
            out_i = out_i + self.bias_i

        return torch.complex(out_r, out_i)


# ---------------------------------------------------------------------------
# AnalogicalReasoningLoRA  (updated for complex tensors)
# ---------------------------------------------------------------------------

class AnalogicalReasoningLoRA:
    """
    Tier 2: Analogical Reasoning via Geodesic LoRA Adapters.
    Trains GeodesicLoRALinear adapters on complex latent states.

    Loss: mean squared magnitude error in ℂ^D
        L = mean(|z_pred - z_target|²)
    """

    def __init__(self, world_model):
        self.world_model = world_model

    def consolidate(self, concept_name: str, memories,
                    rank: int = 4, epochs: int = 10, lr: float = 1e-3):
        """
        memories: (states, actions, outcomes) — may be cfloat or float32 tensors.
        """
        states, actions, outcomes = memories
        if states.shape[0] == 0:
            return

        # Ensure complex tensors
        states   = _ensure_cfloat(states)
        actions  = _ensure_cfloat(actions)
        outcomes = _ensure_cfloat(outcomes)

        self.world_model.apply_lora_adapter(concept_name, rank=rank)
        self.world_model.set_active_lora(concept_name)

        # Only train geodesic LoRA parameters
        trainable_params = [p for p in self.world_model.parameters()
                            if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        self.world_model.train()
        for _ in range(epochs):
            optimizer.zero_grad()

            z = self.world_model.encode(states)
            z_targets = self.world_model.encode(outcomes)  # encode into latent space

            preds = self.world_model.predict_ensemble(z, actions)

            loss = torch.tensor(0.0, requires_grad=True)
            for pred in preds:
                diff = pred - z_targets
                loss = loss + (diff.abs() ** 2).mean()

            loss.backward()
            optimizer.step()
            scheduler.step()

        self.world_model.eval()
        self.world_model.deactivate_lora()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _ensure_cfloat(t: torch.Tensor) -> torch.Tensor:
    """Promote float tensors to cfloat with zero imaginary part."""
    if t.dtype in (torch.float32, torch.float16, torch.float64):
        return torch.complex(t.float(), torch.zeros_like(t.float()))
    return t
