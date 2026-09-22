"""
curiosity.py — Harmonic Resonance Curiosity Policy
====================================================
Replaces the flat epistemic-uncertainty policy with a complex-plane
harmonic resonance evaluator.

Policy logic
------------
For each legal action a_i the policy asks the WorldModel to predict the
next complex latent state z_{t+1} and computes two signals:

1. **Epistemic Uncertainty** (complex ensemble variance):
       U(a) = E[|z_pred - μ|²]   — disagreement between predictor heads.

2. **π² Resonance Score** (interference pattern strength):
       R(a) = Var_heads[ angle(z_pred) ] / (π²/num_heads)
   This is high when the predicted future state lies in a high-variance
   region of the complex phase space — i.e. when piece influence waves
   from that state would produce strong interference patterns (tactically
   rich positions like forks, pins, or discovered attacks).

The final score is a weighted combination:
       score(a) = α · U(a) + (1 - α) · R(a)

α is adaptive: when the model is well-trained (low mean uncertainty) the
resonance term dominates, steering towards tactically interesting moves.
When the model is uncertain (exploration phase) epistemic uncertainty
dominates, steering towards novel states.

Boredom threshold
    If max_score < π²/latent_dim the agent has nothing novel to explore
    and falls back to random action selection (unchanged from original).
"""

import math
import random
import torch

PI2 = math.pi ** 2


class CuriosityPolicy:
    """
    Harmonic-resonance action selector.

    Parameters
    ----------
    num_actions : int
        Number of discrete actions (4 for the grid, or the size of the
        legal-moves list for chess — passed dynamically via select_action_chess).
    action_dim : int
        Dimension of the action embedding passed to the WorldModel.
    alpha : float
        Blend weight between epistemic uncertainty (α) and phase resonance (1-α).
        Adapts automatically based on the mean uncertainty level.
    """

    def __init__(self, num_actions: int = 4, action_dim: int = 16,
                 alpha: float = 0.5):
        self.num_actions = num_actions
        self.action_dim  = action_dim
        self.alpha       = alpha

    # ------------------------------------------------------------------
    # Core resonance scoring
    # ------------------------------------------------------------------

    def _compute_resonance(self, preds: torch.Tensor) -> torch.Tensor:
        """
        preds: (num_heads, batch, latent_dim) cfloat

        Computes the inter-head phase variance per batch sample as a
        proxy for interference pattern strength.

        Returns (batch,) float tensor.
        """
        angles = preds.angle()                         # (H, B, D) real
        phase_var = angles.var(dim=0).mean(dim=-1)     # (B,) real
        # Normalise to the π² scale: resonance = 1 when variance = π²/H
        H = preds.shape[0]
        normaliser = PI2 / H
        return phase_var / (normaliser + 1e-8)

    def _compute_uncertainty(self, preds: torch.Tensor) -> torch.Tensor:
        """
        Complex ensemble variance (epistemic uncertainty).
        Returns (batch,) float tensor.
        """
        mu    = preds.mean(dim=0, keepdim=True)        # (1, B, D)
        sq_dev = (preds - mu).abs() ** 2               # (H, B, D)
        return sq_dev.mean(dim=0).mean(dim=-1)         # (B,)

    def _adaptive_alpha(self, mean_uncertainty: float) -> float:
        """
        Adaptively blend epistemic uncertainty vs resonance.
        When uncertainty is high → exploration (α→1).
        When uncertainty is low  → resonance exploitation (α→0).
        Transition threshold: π² / latent_dim.
        """
        threshold = PI2 / 512.0   # latent_dim=512 default
        if mean_uncertainty > threshold:
            return min(1.0, self.alpha + 0.2)
        else:
            return max(0.0, self.alpha - 0.2)

    # ------------------------------------------------------------------
    # Grid / symbolic environment action selection
    # ------------------------------------------------------------------

    def select_action(self, world_model, z: torch.Tensor,
                      epsilon: float = 0.05) -> int:
        """
        Select an action for the symbolic grid environment.
        z: (1, latent_dim) cfloat latent state.
        Returns a discrete action index.
        """
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)

        scores = []
        world_model.eval()
        with torch.no_grad():
            mean_unc_list = []

            for a_idx in range(self.num_actions):
                # Build a complex action embedding: unit phasor at action phase
                a_tensor = torch.zeros(1, self.action_dim,
                                       dtype=torch.cfloat, device=z.device)
                phase = PI2 * a_idx / self.action_dim
                a_tensor[0, a_idx] = complex(math.cos(phase), math.sin(phase))

                preds = world_model.predict_ensemble(z, a_tensor)  # (H, 1, D)
                unc     = self._compute_uncertainty(preds)          # (1,)
                res     = self._compute_resonance(preds)            # (1,)
                mean_unc_list.append(unc.item())
                scores.append((unc.item(), res.item()))

        world_model.train()

        mean_unc = sum(u for u, _ in scores) / len(scores)
        alpha    = self._adaptive_alpha(mean_unc)

        combined = [alpha * u + (1 - alpha) * r for u, r in scores]
        max_score = max(combined)

        # Boredom threshold: if no action is interesting, act randomly
        boredom_threshold = PI2 / 512.0
        if max_score < boredom_threshold:
            return random.randint(0, self.num_actions - 1)

        return int(torch.argmax(torch.tensor(combined)).item())

    # ------------------------------------------------------------------
    # Chess action selection (legal-move list)
    # ------------------------------------------------------------------

    def select_action_chess(self, world_model, z: torch.Tensor,
                            legal_encodings: list,
                            epsilon: float = 0.05) -> int:
        """
        Select from a list of pre-encoded legal move tensors.

        legal_encodings : List of (action_tensor,) where each tensor is
                          (1, action_dim) cfloat or float32.
        Returns the index into legal_encodings of the selected action.
        """
        if not legal_encodings:
            return 0

        if random.random() < epsilon:
            return random.randint(0, len(legal_encodings) - 1)

        scores = []
        world_model.eval()
        with torch.no_grad():
            for a_tensor in legal_encodings:
                preds = world_model.predict_ensemble(z, a_tensor)
                unc   = self._compute_uncertainty(preds)
                res   = self._compute_resonance(preds)
                scores.append((unc.item(), res.item()))

        world_model.train()

        mean_unc = sum(u for u, _ in scores) / max(len(scores), 1)
        alpha    = self._adaptive_alpha(mean_unc)

        combined = [alpha * u + (1 - alpha) * r for u, r in scores]
        max_score = max(combined)

        boredom_threshold = PI2 / 512.0
        if max_score < boredom_threshold:
            return random.randint(0, len(legal_encodings) - 1)

        return int(torch.argmax(torch.tensor(combined)).item())

    # ------------------------------------------------------------------
    # Tactical spike detector (for MCTS annotation)
    # ------------------------------------------------------------------

    def detect_interference_spikes(self, world_model, z: torch.Tensor,
                                   action_tensors: list) -> list:
        """
        Returns a boolean list indicating which actions lead to board states
        with π² variance spikes — i.e. positions with strong interference
        patterns (forks, discovered checks, etc.).

        Used by the MCTS engine to boost exploration of tactically sharp lines.
        """
        spikes = []
        world_model.eval()
        with torch.no_grad():
            for a_tensor in action_tensors:
                preds = world_model.predict_ensemble(z, a_tensor)  # (H, 1, D)
                # Check if any head predicts a π²-spike state
                angles = preds.angle()
                phase_var = angles.var(dim=0).mean(dim=-1)          # (1,)
                threshold = PI2 / preds.shape[0]
                spikes.append(bool((phase_var > threshold).any().item()))
        world_model.train()
        return spikes
