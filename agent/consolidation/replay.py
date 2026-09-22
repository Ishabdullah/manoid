"""
replay.py — Complex Phase-Space Base Intuition Replay
======================================================
Updates BaseIntuitionReplay to train on complex latent tensors (cfloat).

Changes from the original
-------------------------
* Loss is now the mean squared magnitude error in ℂ^D:
      L = mean(|z_pred - z_target|²)  (real scalar, autograd-compatible)
* Memories are promoted to cfloat if real tensors are passed (backward compat).
* The MCTS complex tensor pipeline can pass cfloat states/actions/outcomes
  directly without flattening.

Phase-space batching
--------------------
Each training batch samples from the HRR episodic buffer.  Since HRR
stores hyperspherical phase coordinates, the batch tensors are already in
cfloat format.  The world model's ComplexEncoder handles real promotion
internally, so both real and complex batches are valid inputs.
"""

import torch
import torch.nn as nn
import torch.optim as optim


def _ensure_cfloat(t: torch.Tensor) -> torch.Tensor:
    """Promote float tensor to cfloat with zero imaginary component."""
    if t.dtype in (torch.float32, torch.float16, torch.float64):
        return torch.complex(t.float(), torch.zeros_like(t.float()))
    return t


class BaseIntuitionReplay:
    """
    Tier 1: Base Intuition — Complex Phase-Space Neural Replay.

    Runs gradient descent on the WorldModel using complex latent memory
    samples from the HRR episodic buffer.  Loss is computed entirely in
    the complex plane to preserve phase information during consolidation.
    """

    def __init__(self, world_model, learning_rate: float = 1e-4):
        self.world_model = world_model
        # Small LR to avoid catastrophic forgetting
        self.optimizer = optim.Adam(self.world_model.parameters(),
                                    lr=learning_rate)

    def consolidate(self, memories, epochs: int = 5, batch_size: int = 32):
        """
        memories: (states, actions, outcomes) — float32 or cfloat tensors.
        Trains WorldModel to predict outcome phase coordinates from (state, action).
        """
        states, actions, outcomes = memories
        dataset_size = states.shape[0]

        if dataset_size == 0:
            return

        # Promote to complex
        states   = _ensure_cfloat(states)
        actions  = _ensure_cfloat(actions)
        outcomes = _ensure_cfloat(outcomes)

        self.world_model.train()
        for epoch in range(epochs):
            permutation = torch.randperm(dataset_size)

            for i in range(0, dataset_size, batch_size):
                indices   = permutation[i:i + batch_size]
                b_states  = states[indices]
                b_actions = actions[indices]
                b_outcomes = outcomes[indices]

                self.optimizer.zero_grad()

                # Encode states in the complex latent space
                z = self.world_model.encode(b_states)             # (B, D) cfloat

                # Encode outcomes into the same latent space as prediction targets
                z_targets = self.world_model.encode(b_outcomes)   # (B, D) cfloat

                # Predict ensemble next states
                z_preds = self.world_model.predict_ensemble(z, b_actions)  # (H,B,D)

                # Complex MSE loss: mean over heads, samples, and latent dims
                loss = torch.tensor(0.0, requires_grad=True)
                for pred in z_preds:
                    diff = pred - z_targets
                    loss = loss + (diff.abs() ** 2).mean()

                loss.backward()
                self.optimizer.step()
