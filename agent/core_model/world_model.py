"""
world_model.py — Complex-Valued Phase-Routing World Model
==========================================================
Replaces flat matrix multiplications and standard attention layers with
Complex-Valued Phase Routing using torch.cfloat (complex64) tensors.

Architecture
------------
ComplexEncoder
    Takes a complex input tensor (board spectrum or grid spectrum), projects
    it through a series of complex linear layers.  Each weight matrix is a
    cfloat tensor so that matrix-vector products propagate both amplitude and
    phase simultaneously.

ComplexPredictor (Ensemble head)
    Receives a complex latent state z ∈ ℂ^{latent_dim} and a complex action
    embedding a ∈ ℂ^{action_dim}, concatenates them, then routes through a
    complex MLP.  The final layer produces z_next ∈ ℂ^{latent_dim}.

ComplexPhaseAttention
    A lightweight self-attention operating in the complex plane.  Queries and
    keys are conjugate-dot-product matched; the attention score is the real
    part of the normalised inner product (harmonic resonance).  This detects
    constructive interference between piece-influence waves automatically.

WorldModel (updated)
    Wraps ComplexEncoder + ensemble of ComplexPredictors.
    LoRA adapters are stored as complex weight rotations (see lora.py).

Epistemic uncertainty
    Variance across ensemble heads computed in the complex plane:
        Var_complex(z) = E[|z - μ|²]  (real-valued scalar per latent dim)
    A π² variance spike (variance > π²/latent_dim) flags high-uncertainty /
    novel states — the curiosity signal.

Backward compatibility
    WorldModel.encode() still accepts real float tensors — they are promoted
    to cfloat automatically by padding the imaginary part with zeros.
    predict_ensemble() and predict_next() return cfloat tensors; callers that
    need real outputs should call .real on the result.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

PI2 = math.pi ** 2


# ---------------------------------------------------------------------------
# Complex Linear layer
# ---------------------------------------------------------------------------

class ComplexLinear(nn.Module):
    """
    A linear layer operating on complex64 tensors.

    For an input x ∈ ℂ^{in} and weight W ∈ ℂ^{out×in}:
        y = x W^H + b          (conjugate transpose to mimic the standard
                                 real convention W @ x^T, adapted for complex)

    Implemented via two real matmuls to stay autograd-compatible on all
    PyTorch backends (some do not support complex autograd for nn.Parameter).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight stored as real and imaginary components separately
        scale = 1.0 / math.sqrt(in_features)
        self.weight_r = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(-scale, scale))
        self.weight_i = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(-scale, scale))

        if bias:
            self.bias_r = nn.Parameter(torch.zeros(out_features))
            self.bias_i = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_r', None)
            self.register_parameter('bias_i', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_features) complex64
        returns: (..., out_features) complex64

        Complex matmul:  (a + ib)(c + id)^H  →  (ac + bd) + i(bc - ad)
        where a,b = x.real, x.imag  and  c,d = weight_r, weight_i
        """
        xr, xi = x.real, x.imag
        wr, wi = self.weight_r, self.weight_i

        out_r = F.linear(xr, wr) - F.linear(xi, wi)
        out_i = F.linear(xi, wr) + F.linear(xr, wi)

        if self.bias_r is not None:
            out_r = out_r + self.bias_r
            out_i = out_i + self.bias_i

        return torch.complex(out_r, out_i)


# ---------------------------------------------------------------------------
# Complex Layer Norm (operates on magnitude, preserves phase)
# ---------------------------------------------------------------------------

class ComplexLayerNorm(nn.Module):
    """Normalises the magnitude of complex activations while preserving phase."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Magnitude normalisation
        mag = x.abs()                                       # (..., D) real
        mag_mean = mag.mean(dim=-1, keepdim=True)
        mag_std  = mag.std(dim=-1, keepdim=True) + self.eps
        mag_norm = (mag - mag_mean) / mag_std               # zero-mean unit-std mag
        # Restore phase: x / |x| gives the unit phasor; scale magnitude
        phase = x / (x.abs().clamp(min=self.eps))
        return phase * (mag_norm * self.scale).abs().to(torch.cfloat)


# ---------------------------------------------------------------------------
# Complex Phase Activation  (analogue of GELU in the complex plane)
# ---------------------------------------------------------------------------

def complex_gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Complex GELU: apply standard GELU independently to real and imaginary parts.
    This is the most gradient-stable complex nonlinearity for training.
    """
    return torch.complex(F.gelu(x.real), F.gelu(x.imag))


# ---------------------------------------------------------------------------
# Complex Phase Attention
# ---------------------------------------------------------------------------

class ComplexPhaseAttention(nn.Module):
    """
    Self-attention in the complex plane using harmonic resonance scores.

    Attention score between query q and key k:
        score(q, k) = Re(q · k̄) / √d_k     (real part of conjugate dot product)

    This is equivalent to cos(phase_q - phase_k) × |q|×|k| / √d_k, so the
    attention weight is highest when two piece-influence waves are in phase
    (constructive interference → tactical coordination).

    π² variance detection
    ----------------------
    After computing attention weights, we compute the phase variance across
    heads.  Bins where this variance > π²/num_heads are flagged as interference
    hotspots (candidate forks / discoveries).
    """

    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj = ComplexLinear(embed_dim, embed_dim)
        self.k_proj = ComplexLinear(embed_dim, embed_dim)
        self.v_proj = ComplexLinear(embed_dim, embed_dim)
        self.out_proj = ComplexLinear(embed_dim, embed_dim)

        # Stores the last computed phase-variance map for external inspection
        self.last_phase_variance: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, embed_dim) cfloat
        returns: (batch, seq_len, embed_dim) cfloat
        """
        B, S, D = x.shape
        H, Hd = self.num_heads, self.head_dim

        Q = self.q_proj(x).reshape(B, S, H, Hd).permute(0, 2, 1, 3)  # (B,H,S,Hd)
        K = self.k_proj(x).reshape(B, S, H, Hd).permute(0, 2, 1, 3)
        V = self.v_proj(x).reshape(B, S, H, Hd).permute(0, 2, 1, 3)

        # Harmonic resonance score: Re(Q · K̄^T) / √Hd
        # Q: (B,H,S,Hd)  K̄: conjugate of K → (B,H,Hd,S)
        # batched matmul of complex tensors
        attn_scores = (Q @ K.conj().transpose(-2, -1)) * self.scale  # (B,H,S,S) cfloat
        attn_real   = attn_scores.real                                 # harmonic resonance
        attn_weights = torch.softmax(attn_real, dim=-1)                # (B,H,S,S) float

        # π² variance spike detection across heads
        phase_angles = attn_scores.angle()                             # (B,H,S,S)
        phase_var = phase_angles.var(dim=1)                            # (B,S,S)
        spike_threshold = PI2 / self.num_heads
        self.last_phase_variance = (phase_var > spike_threshold).float()

        # Apply attention to values (promote real weights to cfloat)
        attn_weights_c = attn_weights.to(torch.cfloat)
        out = (attn_weights_c @ V)                                     # (B,H,S,Hd)
        out = out.permute(0, 2, 1, 3).reshape(B, S, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Complex Encoder
# ---------------------------------------------------------------------------

class ComplexEncoder(nn.Module):
    """
    Maps raw complex environment observations into a dense complex latent state.

    Accepts either:
      - (batch, 12, 8, 8) cfloat  chess board spectrum
      - (batch, N, N) cfloat      grid spectrum
      - (batch, input_dim) float32  (real → promoted to cfloat)
    """

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim  = input_dim
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            ComplexLinear(input_dim, 2048),
            ComplexLayerNorm(2048),
        )
        self.net2 = nn.Sequential(
            ComplexLinear(2048, 2048),
            ComplexLayerNorm(2048),
        )
        self.out = ComplexLinear(2048, latent_dim)

    def _promote(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure x is cfloat, flattening and promoting if necessary."""
        if x.dtype in (torch.float32, torch.float16, torch.float64):
            x = x.to(torch.float32)
            x = torch.complex(x, torch.zeros_like(x))
        # Flatten all dims after batch dim
        return x.reshape(x.shape[0], -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self._promote(x)

        # Pad or truncate to input_dim
        D = self.input_dim
        if z.shape[-1] < D:
            pad = torch.zeros(z.shape[0], D - z.shape[-1], dtype=torch.cfloat, device=z.device)
            z = torch.cat([z, pad], dim=-1)
        elif z.shape[-1] > D:
            z = z[:, :D]

        h = complex_gelu(self.net[1](self.net[0](z)))
        h = complex_gelu(self.net2[1](self.net2[0](h)))
        return self.out(h)


# ---------------------------------------------------------------------------
# Complex Predictor (ensemble head)
# ---------------------------------------------------------------------------

class ComplexPredictor(nn.Module):
    """
    Predicts the next complex latent state from (z, action).
    Both inputs are complex; concatenation doubles the real-equivalent width.
    """

    def __init__(self, latent_dim: int, action_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        combined = latent_dim + action_dim

        self.net = nn.Sequential(
            ComplexLinear(combined, 2048),
            ComplexLayerNorm(2048),
        )
        self.net2 = nn.Sequential(
            ComplexLinear(2048, 2048),
            ComplexLayerNorm(2048),
        )
        self.out = ComplexLinear(2048, latent_dim)

        # Phase attention module — operates on the sequence (z, a) of length 2
        self.phase_attn = ComplexPhaseAttention(embed_dim=latent_dim, num_heads=4)

    def _promote_action(self, action: torch.Tensor, latent_dim: int) -> torch.Tensor:
        """Ensure action is cfloat and padded / trimmed to action_dim."""
        if action.dtype in (torch.float32, torch.float16, torch.float64):
            action = torch.complex(action.float(), torch.zeros_like(action.float()))
        action = action.reshape(action.shape[0], -1)
        D = self.action_dim
        if action.shape[-1] < D:
            pad = torch.zeros(action.shape[0], D - action.shape[-1],
                              dtype=torch.cfloat, device=action.device)
            action = torch.cat([action, pad], dim=-1)
        return action[:, :D]

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        z      : (batch, latent_dim) cfloat
        action : (batch, action_dim) float32 or cfloat
        returns: (batch, latent_dim) cfloat
        """
        a = self._promote_action(action, self.latent_dim)

        # Holographic phase attention between state and action
        # Treat z and a as a 2-token sequence of latent_dim-dim vectors
        # (action padded to latent_dim for the attention module)
        a_padded = torch.zeros(a.shape[0], self.latent_dim,
                               dtype=torch.cfloat, device=a.device)
        a_padded[:, :self.action_dim] = a

        seq = torch.stack([z, a_padded], dim=1)   # (B, 2, latent_dim)
        attended = self.phase_attn(seq)            # (B, 2, latent_dim)
        z_attended = attended[:, 0, :]             # use attended z
        a_attended = attended[:, 1, :self.action_dim]

        combined = torch.cat([z_attended, a_attended], dim=-1)  # (B, latent+action)

        h = complex_gelu(self.net[1](self.net[0](combined)))
        h = complex_gelu(self.net2[1](self.net2[0](h)))
        return self.out(h)


# ---------------------------------------------------------------------------
# WorldModel (complex-phase routing)
# ---------------------------------------------------------------------------

class WorldModel(nn.Module):
    """
    Core Intelligence: Complex-Valued Phase Routing World Model.

    Equivalent to the original ~50 M parameter JEPA ensemble, now operating
    entirely in the complex plane.  Epistemic uncertainty is computed as the
    π²-normalised variance of the ensemble predictions in ℂ^{latent_dim}.

    LoRA adapters (see lora.py) are stored as spherical geodesic rotations of
    the weight matrices and can be hot-swapped without rebuilding the graph.
    """

    def __init__(self, input_dim: int = 256, action_dim: int = 16,
                 latent_dim: int = 512, num_heads: int = 3):
        super().__init__()
        self.latent_dim  = latent_dim
        self.action_dim  = action_dim
        self.input_dim   = input_dim

        self.encoder = ComplexEncoder(input_dim, latent_dim)
        self.predictors = nn.ModuleList(
            [ComplexPredictor(latent_dim, action_dim) for _ in range(num_heads)]
        )

        self._lora_adapters: nn.ModuleDict = nn.ModuleDict()
        self._active_lora: Optional[str] = None

    # ------------------------------------------------------------------
    # LoRA adapter management (complex geodesic)
    # ------------------------------------------------------------------

    def apply_lora_adapter(self, concept_name: str, rank: int = 4):
        """
        Creates a complex LoRA adapter for the given concept.
        Delegates to the GeodesicLoRALinear wrapper in lora.py which rotates
        weight matrices along spherical geodesics in complex weight space.
        """
        if concept_name in self._lora_adapters:
            return

        import copy
        from agent.consolidation.lora import GeodesicLoRALinear

        concept_predictors = nn.ModuleList()
        for p in self.predictors:
            p_copy = copy.deepcopy(p)
            # Replace the output ComplexLinear with a GeodesicLoRALinear wrapper
            p_copy.out = GeodesicLoRALinear(p_copy.out, rank=rank)
            concept_predictors.append(p_copy)

        self._lora_adapters[concept_name] = concept_predictors

    def merge_adapters(self, active_concepts: List[str],
                       new_concept_name: str = "merged_consensus"):
        """
        Task Vector Merging via complex consensus pruning.
        Sign conflicts are resolved in the complex magnitude domain.
        """
        if len(active_concepts) < 2:
            return

        import copy

        merged_predictors = nn.ModuleList()
        for p_idx, p_base in enumerate(self.predictors):
            p_merged = copy.deepcopy(p_base)
            base_linear = p_merged.out

            # Sum geodesic deltas; mask by consensus
            delta_r_sum = torch.zeros_like(base_linear.weight_r)
            delta_i_sum = torch.zeros_like(base_linear.weight_i)
            sign_r_consensus = torch.zeros_like(base_linear.weight_r)
            sign_i_consensus = torch.zeros_like(base_linear.weight_i)

            for concept in active_concepts:
                adapter = self._lora_adapters[concept][p_idx].out  # GeodesicLoRALinear
                dr, di = adapter.get_geodesic_delta()
                delta_r_sum += dr
                delta_i_sum += di
                sign_r_consensus += torch.sign(dr)
                sign_i_consensus += torch.sign(di)

            n = len(active_concepts)
            mask_r = (torch.abs(sign_r_consensus) == n).float()
            mask_i = (torch.abs(sign_i_consensus) == n).float()

            p_merged.out.weight_r = nn.Parameter(
                base_linear.weight_r + delta_r_sum * mask_r)
            p_merged.out.weight_i = nn.Parameter(
                base_linear.weight_i + delta_i_sum * mask_i)
            merged_predictors.append(p_merged)

        self._lora_adapters[new_concept_name] = merged_predictors

    def set_active_lora(self, concept_name: str):
        """Activates a named LoRA adapter."""
        if concept_name in self._lora_adapters:
            self._active_lora = concept_name

    def deactivate_lora(self):
        """Returns to the base intuition model."""
        self._active_lora = None

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Extract complex latent embedding from raw observations.
        Accepts real float tensors (auto-promoted) or cfloat tensors.
        Returns (batch, latent_dim) cfloat.
        """
        return self.encoder(obs)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _active_predictors(self):
        if self._active_lora is not None and self._active_lora in self._lora_adapters:
            return self._lora_adapters[self._active_lora]
        return self.predictors

    def predict_ensemble(self, z: torch.Tensor,
                         action: torch.Tensor) -> torch.Tensor:
        """
        Returns complex predictions from all heads.
        Shape: (num_heads, batch, latent_dim) cfloat
        """
        preds = [p(z, action) for p in self._active_predictors()]
        return torch.stack(preds)

    def predict_next(self, z: torch.Tensor,
                     action: torch.Tensor) -> torch.Tensor:
        """Mean prediction across ensemble heads. (batch, latent_dim) cfloat"""
        return self.predict_ensemble(z, action).mean(dim=0)

    # ------------------------------------------------------------------
    # Uncertainty / curiosity metrics
    # ------------------------------------------------------------------

    def compute_epistemic_uncertainty(self, z: torch.Tensor,
                                      action: torch.Tensor) -> torch.Tensor:
        """
        Curiosity signal: complex-plane ensemble variance.

            Var_complex = E[|z_pred - μ|²]   (real-valued per latent dim)

        A π² spike (variance > π² / latent_dim) signals a novel or
        contested board configuration.

        Returns (batch,) float scalar uncertainty per sample.
        """
        preds = self.predict_ensemble(z, action)          # (H, B, D) cfloat
        mu    = preds.mean(dim=0, keepdim=True)           # (1, B, D)
        sq_dev = (preds - mu).abs() ** 2                  # (H, B, D) real
        var   = sq_dev.mean(dim=0).mean(dim=-1)           # (B,)    real
        return var

    def compute_surprise(self, z_pred: torch.Tensor,
                         z_actual: torch.Tensor) -> torch.Tensor:
        """
        Forward-model surprise: mean squared magnitude error in ℂ.
        |z_pred - z_actual|² averaged over latent dimensions.
        """
        diff = z_pred - z_actual
        return (diff.abs() ** 2).mean(dim=-1)

    def detect_pi2_spikes(self, z: torch.Tensor,
                          action: torch.Tensor) -> torch.Tensor:
        """
        Returns a boolean mask (batch,) that is True for samples where
        the epistemic uncertainty exceeds the π²-normalised threshold.
        These correspond to tactical interference patterns (forks, discoveries).
        """
        uncertainty = self.compute_epistemic_uncertainty(z, action)
        threshold   = PI2 / self.latent_dim
        return uncertainty > threshold

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
