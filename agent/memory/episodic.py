"""
episodic.py — Holographic Reduced Representation Episodic Memory
================================================================
Migrates the flat tensor replay buffer to a Holographic Reduced
Representation (HRR) store.

What is HRR?
------------
Holographic Reduced Representations (Plate, 1995) encode structured
data as unit-length complex vectors (phasors) on the hypersphere in ℂ^D.
Key properties exploited here:

* **Binding**   : z_state ⊛ z_action  via circular convolution
                  (= element-wise multiplication in the frequency domain).
* **Superposition**: Multiple (state, action, outcome) triples can be
                  summed into a single memory trace vector without
                  interference proportional to their dot product.
* **Retrieval** : Decode a stored trace by circular correlation
                  (element-wise multiply by conjugate of key).

Storage geometry
----------------
All latent vectors are projected onto the complex unit hypersphere
(‖z‖₂ = 1 in ℂ^D) before storage.  This guarantees that the similarity
metric (real part of the complex dot product) is bounded in [-1, 1] and
that the buffer occupies a fixed hyperspherical surface rather than the
full latent space volume.

Schema  (each row in the buffer)
    hrr_trace : (D,) cfloat   — bound HRR of (state ⊛ action)
    outcome   : (D,) cfloat   — phase coordinate of outcome state
    confidence: (1,) float32  — Bayesian confidence weight

Retrieval
    Deconvolve: candidate_outcome ≈ hrr_trace ⊛ inv(query_action)
    Similarity computed as |Re(z_query · z_stored†)| (inner product mag).
"""

import os
import math
import torch
import torch.nn.functional as F

PI2 = math.pi ** 2


def _to_unit_sphere(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Project complex tensor onto the unit hypersphere in ℂ^D."""
    if z.dtype not in (torch.cfloat, torch.cdouble):
        z = torch.complex(z.float(), torch.zeros_like(z.float()))
    norms = z.abs().norm(dim=-1, keepdim=True).clamp(min=eps)
    return z / norms


def _circular_convolve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Element-wise multiplication in the frequency domain = circular convolution.
    HRR binding: a ⊛ b  ≡  ifft(fft(a) * fft(b))
    For unit vectors this keeps the result on the hypersphere.
    Both a, b: (..., D) cfloat.
    """
    return torch.fft.ifft(torch.fft.fft(a) * torch.fft.fft(b))


def _circular_correlate(trace: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """
    HRR decoding: trace ⊛⁻¹ key  ≡  ifft(fft(trace) * conj(fft(key)))
    """
    return torch.fft.ifft(torch.fft.fft(trace) * torch.fft.fft(key).conj())


def _encode_phase_coords(z: torch.Tensor) -> torch.Tensor:
    """
    Encode a real or complex latent vector as hyperspherical phase coordinates.
    If real: convert to a complex phasor using z as magnitudes with phase
    angles π² · i / D (unique per dimension).
    """
    if z.dtype in (torch.float32, torch.float16, torch.float64):
        D = z.shape[-1]
        phases = torch.tensor(
            [PI2 * i / D for i in range(D)], dtype=torch.float32, device=z.device)
        # Broadcast phases to batch dimensions
        while phases.dim() < z.dim():
            phases = phases.unsqueeze(0)
        z_complex = torch.complex(
            z.float() * torch.cos(phases),
            z.float() * torch.sin(phases)
        )
        return _to_unit_sphere(z_complex)
    return _to_unit_sphere(z)


class EpisodicMemory:
    """
    HRR-based Episodic Memory with hyperspherical phase coordinates.

    Storage schema (pre-allocated ring buffer):
        hrr_traces   : (max_size, D) cfloat — HRR-bound (state ⊛ action)
        outcomes     : (max_size, D) cfloat — phase coordinates of outcomes
        confidences  : (max_size, 1) float32 — Bayesian confidence

    Interface unchanged: store(), retrieve(), update_confidence(), save(), load().
    """

    def __init__(self, state_dim: int = 512, action_dim: int = 16,
                 max_size: int = 10000, device: str = 'cpu'):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.max_size   = max_size
        self.device     = device

        # HRR traces are always in the state_dim hypersphere (action embedded too)
        self.hrr_traces  = torch.zeros((max_size, state_dim), dtype=torch.cfloat,
                                       device=device)
        # Raw HRR of states (needed for retrieval without full decode)
        self.state_hrrs  = torch.zeros((max_size, state_dim), dtype=torch.cfloat,
                                       device=device)
        # Raw HRR of actions (needed for decoding)
        self.action_hrrs = torch.zeros((max_size, state_dim), dtype=torch.cfloat,
                                       device=device)
        self.outcomes    = torch.zeros((max_size, state_dim), dtype=torch.cfloat,
                                       device=device)
        self.confidences = torch.zeros((max_size, 1), dtype=torch.float32,
                                       device=device)

        self.ptr  = 0
        self.size = 0

    # ------------------------------------------------------------------
    # Internal encoding helpers
    # ------------------------------------------------------------------

    def _embed_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Embed an action (float one-hot or cfloat) into state_dim phase space.
        Uses sinusoidal positional encoding scaled by π².
        """
        B = action.shape[0]
        D = self.state_dim
        A = self.action_dim

        # Promote to float
        if action.dtype in (torch.cfloat, torch.cdouble):
            action_real = action.real.float()
        else:
            action_real = action.float()

        # Pad / trim to action_dim
        action_real = action_real.reshape(B, -1)
        if action_real.shape[-1] < A:
            pad = torch.zeros(B, A - action_real.shape[-1],
                              dtype=torch.float32, device=action_real.device)
            action_real = torch.cat([action_real, pad], dim=-1)
        action_real = action_real[:, :A]

        # Tile action into D-dim space
        repeats = math.ceil(D / A)
        tiled   = action_real.repeat(1, repeats)[:, :D]   # (B, D)

        # Apply unique π²-scaled phases per dimension
        phases = torch.tensor(
            [PI2 * i / D for i in range(D)], dtype=torch.float32,
            device=action_real.device).unsqueeze(0)
        a_complex = torch.complex(tiled * torch.cos(phases),
                                  tiled * torch.sin(phases))
        return _to_unit_sphere(a_complex)   # (B, D) cfloat

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(self, state: torch.Tensor, action: torch.Tensor,
              outcome: torch.Tensor, confidence: torch.Tensor):
        """
        Encode and store a (state, action, outcome, confidence) experience.

        Inputs may be real float or cfloat tensors.
        Internally converts to HRR phase coordinates and stores the
        circular convolution of (state_hrr ⊛ action_hrr) as the trace.
        """
        # Normalise batch dimension
        if state.dim() == 1:
            state      = state.unsqueeze(0)
            action     = action.unsqueeze(0)
            outcome    = outcome.unsqueeze(0)
            confidence = confidence.unsqueeze(0)

        B = state.shape[0]

        # Convert to hyperspherical phase coordinates
        s_hrr = _encode_phase_coords(state.to(self.device))    # (B, D) cfloat
        a_hrr = self._embed_action(action.to(self.device))      # (B, D) cfloat
        o_hrr = _encode_phase_coords(outcome.to(self.device))   # (B, D) cfloat

        # HRR binding: trace = state ⊛ action (circular convolution)
        trace = _circular_convolve(s_hrr, a_hrr)                # (B, D) cfloat
        trace = _to_unit_sphere(trace)                          # project back

        for i in range(B):
            self.hrr_traces[self.ptr]  = trace[i]
            self.state_hrrs[self.ptr]  = s_hrr[i]
            self.action_hrrs[self.ptr] = a_hrr[i]
            self.outcomes[self.ptr]    = o_hrr[i]
            self.confidences[self.ptr] = confidence[i].float().reshape(1)

            self.ptr  = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, query_state: torch.Tensor, query_action: torch.Tensor,
                 k: int = 1):
        """
        Query the HRR buffer for the most similar past experiences.

        Similarity is measured as the real part of the complex inner product
        between the query HRR trace and stored traces (cosine similarity in ℂ^D).

        Returns:
            retrieved_outcomes  : (1, k, D) cfloat  — decoded outcome HRRs
            retrieved_confidences: (1, k, 1) float
            top_sims            : (1, k) float
            top_indices         : (1, k) long
        """
        if self.size == 0:
            return None, None, None, None

        if query_state.dim() == 1:
            query_state  = query_state.unsqueeze(0)
            query_action = query_action.unsqueeze(0)

        # Encode query to HRR
        q_s = _encode_phase_coords(query_state.to(self.device))
        q_a = self._embed_action(query_action.to(self.device))
        q_trace = _to_unit_sphere(_circular_convolve(q_s, q_a))  # (1, D)

        # Cosine similarity in ℂ^D = Re(q · z†) / (|q|·|z|)  — unit vecs so =Re(q·z†)
        valid_traces = self.hrr_traces[:self.size]                 # (N, D)
        # Re(q · conj(z)) per pair: matmul of real + imag
        inner_real = (q_trace.real @ valid_traces.real.T +
                      q_trace.imag @ valid_traces.imag.T)         # (1, N)
        similarities = inner_real                                  # bounded [-1,1]

        top_k = min(k, self.size)
        top_sims, top_indices = torch.topk(similarities, k=top_k, dim=-1)

        retrieved_outcomes     = self.outcomes[top_indices]       # (1, k, D)
        retrieved_confidences  = self.confidences[top_indices]    # (1, k, 1)
        return retrieved_outcomes, retrieved_confidences, top_sims, top_indices

    def decode_outcome(self, hrr_outcome: torch.Tensor,
                       query_action: torch.Tensor) -> torch.Tensor:
        """
        Decode a stored HRR outcome back to a real-valued latent vector
        by computing the circular correlation with the inverse action.

        hrr_outcome : (D,) cfloat
        query_action: (1, action_dim) float or cfloat
        Returns: (D,) cfloat  — approximate recovered state HRR
        """
        q_a = self._embed_action(query_action.to(self.device)).squeeze(0)
        decoded = _circular_correlate(hrr_outcome, q_a)
        return decoded

    # ------------------------------------------------------------------
    # Bayesian confidence update
    # ------------------------------------------------------------------

    def update_confidence(self, indices: torch.Tensor, delta: float):
        """Bayesian confidence update (Phase 5 mechanism, unchanged interface)."""
        idx_flat = indices.flatten()
        current  = self.confidences[idx_flat]
        self.confidences[idx_flat] = torch.clamp(current + delta, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str):
        """Persist HRR memory buffer to disk."""
        torch.save({
            'hrr_traces':   self.hrr_traces[:self.size],
            'state_hrrs':   self.state_hrrs[:self.size],
            'action_hrrs':  self.action_hrrs[:self.size],
            'outcomes':     self.outcomes[:self.size],
            'confidences':  self.confidences[:self.size],
            'ptr':          self.ptr,
            'size':         self.size,
        }, filepath)

    def load(self, filepath: str):
        """Load HRR memory buffer from disk."""
        if os.path.exists(filepath):
            data = torch.load(filepath, map_location=self.device)
            self.size = data['size']
            self.ptr  = data['ptr']
            self.hrr_traces[:self.size]  = data['hrr_traces']
            self.state_hrrs[:self.size]  = data.get(
                'state_hrrs', self.state_hrrs[:self.size])
            self.action_hrrs[:self.size] = data.get(
                'action_hrrs', self.action_hrrs[:self.size])
            self.outcomes[:self.size]    = data['outcomes']
            self.confidences[:self.size] = data['confidences']
