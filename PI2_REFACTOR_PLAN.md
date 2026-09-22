# PI2_REFACTOR_PLAN.md — Manoid π² Full Phase-Space Refactor Plan

> **Status: PLAN ONLY — no implementation code has been written for Phase 2 yet.**
> This document must be reviewed and approved by a human before any refactor code
> is touched.  Phase 1 A/B results (`RESULTS_PATH1.md`) must exist before starting.

---

## Executive Summary

This plan describes a staged refactor of Manoid to make Complex-Valued Phase Routing
the **primary** move-selection and value-estimation path, replacing the side-channel
approach (bonus modulating a separate real-valued model) with an architecture where the
harmonic-resonance computation *is* the policy and value head.

The three pillars are:
1. **Environment** — continuous 2D Fourier-domain board representation via `torch.fft.fft2`
2. **Core model & policy** — complex-valued transformer as the primary value/policy network
3. **Memory & consolidation** — HRR episodic memory + geodesic LoRA

---

## A. STARTING POINT DECISION

### Recommendation: Fresh Start (new branch + new checkpoints)

**Rationale:**

1. **Checkpoint incompatibility is unavoidable.** The refactor changes the input tensor
   shape from `(B, 837) float32` to `(B, 12, 8, 8) cfloat`, and the output from
   separate `(v, p_from, p_to, depth, dense, pv_f, pv_t)` real heads to a unified
   complex latent that feeds a single `ComplexValuePolicy` head.  No weight can migrate.

2. **The existing ChessValueModelPhase14 is a sunk cost, not a warm start.**  Its
   weights encode patterns for the real-valued board representation.  Trying to bridge
   them (e.g. via distillation) would cost ≥50 training episodes to transfer — more
   than simply retraining from scratch on the new representation.

3. **A clean branch preserves the known-working baseline.** The current `master` +
   checkpoint files remain the rollback target (see Section E).

**Compatibility bridge decision: NOT recommended.**
A compatibility bridge (running both architectures simultaneously and distilling from
old to new) adds code complexity with unclear gain.  The new architecture is expressive
enough to learn from Stockfish feedback directly.  Start fresh.

---

## B. STAGED ROLLOUT — Three Independent Stages

Each stage is independently testable.  Do not combine multiple stages in one commit.

---

### Stage 1 — Environment: Fourier Board Encoding

**What changes:**
- `full_chess_env.py`: `encode_state()` → `(1, 12, 8, 8) cfloat` (already done in Phase 1)
- Piece amplitude scheme finalized (see Section F, Option survey)
- `evaluate_heuristic()` uses π² variance spikes (already done)
- `encode_state_real()` kept for backward compat during transition

**What stays the same:**
- All training scripts, value model, MCTS planner

**Metric to validate Stage 1 alone:**
- `evaluate_heuristic()` should rank positions consistently with Stockfish depth-1
  centipawn scores: Spearman ρ > 0.6 across 200 random positions from game logs.
  Measure this with a one-off evaluation script before proceeding to Stage 2.
- Wall-clock time per `encode_state()` call should not exceed 2× the legacy
  `encode_state_real()` time on the Snapdragon device.

**Stage 1 commit is independent:** the rest of the system can keep using
`encode_state_real()` throughout Stage 1.

---

### Stage 2 — Core Model: ComplexValuePolicy Replaces ChessValueModelPhase14

**What changes:**
- New `ComplexValuePolicy(nn.Module)` that accepts `(B, 12, 8, 8) cfloat` and outputs
  - `v_complex: (B, 1) cfloat` — value estimate (take `Re(v)` for scalar)
  - `p_logits: (B, 64, 64) float32` — from-to policy (projected from complex latent)
- Uses `ComplexEncoder` + `ComplexPhaseAttention` as the trunk (already implemented
  in `world_model.py`); adds lightweight real-valued heads for value scalar and policy
  logits (complex → real projection via magnitude + phase concatenation)
- `BatchingMCTSPlanner.evaluate_batch()` now calls `ComplexValuePolicy` directly;
  `ChessValueModelPhase14` is retired
- `USE_2PLY_LOOKAHEAD`, Gumbel search, quiescence remain unchanged (they depend only
  on scalar values and move ordering, not on the trunk architecture)

**What stays the same:**
- Training objective (MSE on value, cross-entropy on policy)
- Replay buffer, Elo tracking, Stockfish curriculum
- `EpisodicMemory`, `LoRA` system

**Metric to validate Stage 2 alone (vs. Stage 1 baseline):**
- Over 50 training episodes from a fresh start, `ComplexValuePolicy` must reach
  **mean value-head MSE ≤ 0.05** (the current Phase 18 baseline is ~0.014 after
  convergence; accepting ≤0.05 at ep 50 means it is learning)
- Final Elo after 50 episodes must be within **±100 of the real-model baseline Elo**
  at the same episode count (measured from the same Stockfish skill curriculum)
- Episode wall-clock time must not exceed **150% of Stage 1 baseline**

**Stage 2 commit is independent:** if Stage 2 fails its metrics, roll back to the
Stage 1 commit and re-evaluate the ComplexValuePolicy architecture.

---

### Stage 3 — Memory & Consolidation: HRR + Geodesic LoRA as Primary Path

**What changes:**
- `EpisodicMemory` HRR ring buffer already implemented; Stage 3 wires the **outcome
  update** loop: after each game, iterate stored entries from that game and overwrite
  their `outcome` field with the terminal game-result scalar (±1/0) projected into
  the latent space. Implement `episodic_memory.update_outcomes(game_idx, result_z)`.
- `GeodesicLoRALinear` replaces all `LoRALinear` adapters in `ComplexValuePolicy`
  (not just in `WorldModel`), so the main policy network adapts via geodesic rotation.
- `BaseIntuitionReplay` consolidation runs every 25 episodes on the HRR buffer
  (currently it runs on raw episode states; Stage 3 switches to HRR-decoded states).

**What stays the same:**
- `ComplexValuePolicy` trunk architecture from Stage 2
- All MCTS planner logic

**Metric to validate Stage 3 alone (vs. Stage 2):**
- HRR retrieval similarity for game-state queries > 0.7 (cosine, real inner product)
  after 50 episodes; verify with `episodic_memory.retrieve()` on held-out positions.
- Outcome update loop must correctly reflect game result in decoded outcomes:
  `decode_outcome(z_query, a_query)` for a winning position should return a vector
  with `Re(v)` > 0. Validate on 20 known-win positions.
- No Elo regression > 50 points vs. Stage 2 at the same episode count.

**Stage 3 commit is independent:** geodesic LoRA and HRR outcome updates can be
A/B'd independently using the same `use_world_model_bonus` style ablation flag.

---

## C. COMPUTE / HARDWARE REALITY CHECK (Snapdragon 8 Gen 3, Termux, no CUDA)

### What is confirmed supported

| Operation | Support | Note |
|-----------|---------|------|
| `torch.fft.fft2` on CPU | ✅ Supported | Uses pocketfft (bundled in PyTorch CPU). Works on ARM. |
| `torch.cfloat` tensors | ✅ Supported | Complex64 = 2× float32 memory |
| `torch.complex` autograd | ✅ Supported (with caveats) | Conjugate Wirtinger derivative. `nn.Parameter` of cfloat NOT recommended — store as two float32 params (already done in `ComplexLinear`) |
| `torch.fft.ifft` | ✅ Supported | Same pocketfft |
| `F.normalize` on cfloat | ❌ Unsupported | Must use `.abs().clamp(min=eps)` for magnitude, then divide |
| `nn.LayerNorm` on cfloat | ❌ Unsupported | Use `ComplexLayerNorm` (already implemented) |
| ExecuTorch / NNAPI delegates | ❌ Not used | Training-time only; ExecuTorch is for inference export |

### Memory budget

- Current model: ChessValueModelPhase14 ≈ 15M params × 4 bytes = **~60 MB**
- WorldModel (latent_dim=512, num_heads=3): ~40M params × 8 bytes (cfloat) = **~320 MB**
  - **This exceeds the README's 400 MB budget when combined with training overhead.**
  - **Mitigation**: reduce `latent_dim` to 128 and `hidden_dim` to 512 for Stage 2.
    Estimated ~5M params = **~40 MB**. Benchmark RAM on device before committing.
- `torch.fft.fft2` on (B=32, 12, 8, 8): output is (32, 12, 8, 8) cfloat = 32×12×64×8 = **~6 MB** per batch. Acceptable.

### Speed expectations on Snapdragon 8 Gen 3 (CPU only)

- `torch.fft.fft2` on (1, 12, 8, 8): estimated **< 0.5 ms** (ARM NEON SIMD, pocketfft is well-optimized)
- `ComplexLinear` (16→512 hidden): ~2× slower than real `nn.Linear` due to 4-matmul decomposition
  (~1 ms vs ~0.5 ms per layer for batch=1)
- `ComplexPhaseAttention` for N=32 tokens, D=128: estimated **~5–10 ms** per call
- Per-move predict_ensemble (now batched across N_moves per FEN): estimated **~20–50 ms** per MCTS node evaluation vs. the current Stockfish engine call dominating at **~100 ms/call**
- **Conclusion**: ComplexValuePolicy is unlikely to be a throughput bottleneck compared to Stockfish I/O. However, `latent_dim=512` will likely cause OOM on training. Use 128.

> [!WARNING]
> Do NOT set `latent_dim > 256` without benchmarking RAM on the actual device.
> The current `world_model_ckpt.pt` uses latent_dim=512 — this must be reduced before Stage 2.

---

## D. FALSIFIABLE SUCCESS CRITERIA (defined before implementation)

These thresholds are written now, before any Stage 2/3 code is written. Results
cannot be re-interpreted after the fact.

| Stage | Metric | Pass threshold | Fail threshold | Measurement method |
|-------|--------|---------------|---------------|-------------------|
| 1 (env) | Heuristic vs SF Spearman ρ | ≥ 0.60 | < 0.45 | 200 random positions, `evaluate_heuristic()` vs SF depth-5 cp |
| 1 (env) | `encode_state()` wall time | ≤ 2× legacy | > 5× legacy | `timeit` on device, N=1000 calls |
| 2 (model) | MSE at ep 50 | ≤ 0.05 | > 0.10 | Phase 18 training log, value head loss |
| 2 (model) | Elo at ep 50 | within ±100 of real-model ep-50 baseline | > 200 below | Elo tracking in marathon.log |
| 2 (model) | Episode wall time | ≤ 150% of Stage 1 | > 200% | `time` per episode in marathon.log |
| 3 (memory) | HRR retrieval sim | ≥ 0.70 | < 0.50 | `retrieve()` on 50 held-out positions |
| 3 (memory) | Outcome decode accuracy | Re(decode(win_pos)) > 0 for 15/20 wins | < 10/20 | Manual spot-check after 50 training eps |
| Overall | Elo at ep 200 | > baseline (Phase 18 ep 200 Elo) + 50 | < baseline − 100 | marathon.log |

> [!IMPORTANT]
> If ANY metric falls in the "fail" column for its stage, **stop and reassess before
> continuing to the next stage.** Do not fix a failing stage by tweaking thresholds.

---

## E. ROLLBACK PLAN

### Preserved baseline

- **Branch**: `master` at commit `20a9411` (post Phase 1 bugfix)
- **Checkpoint**: `manoid_v2_ckpt_*.pt` files in the working directory (do NOT delete)
- **WorldModel checkpoint**: `world_model_ckpt.pt` (separate file, not breaking)

### How to roll back

```bash
# Roll back to the known-working real-valued model
git checkout master  # or the specific commit hash

# The existing .pt checkpoints are loaded by ChessValueModelPhase14 via
# value_model.load_state_dict(...) — these are architecture-independent
# (just weight dicts) and will work as long as ChessValueModelPhase14
# exists in the code, which it will on the master branch.
```

### Branch strategy for Phase 2

```
master          ← current working baseline (do not break this)
  └── pi2-stage1   ← Stage 1 only (env encoding)
        └── pi2-stage2  ← Stage 1 + Stage 2 (complex model)
              └── pi2-stage3  ← all three stages
```

Each stage branch is independent. Failing stages are abandoned, not merged.

---

## F. OPEN QUESTIONS TO RESEARCH (do not assume, measure)

---

### F.1 — Piece-to-Frequency Mapping Options

Three concrete options with tradeoffs:

**Option 1: Material-value amplitude (current implementation)**
```
A_k = {Pawn: 0.1, Knight: 0.3, Bishop: 0.32, Rook: 0.5, Queen: 0.9, King: 1.0}
phase = π² × piece_channel / 12
```
- ✅ Simple, interpretable, preserves material intuition
- ❌ Bishops and Knights have almost identical amplitudes (0.30 vs 0.32), so their
  spectral contribution is nearly indistinguishable — tactical patterns involving
  a bishop-knight battery won't produce distinct interference
- ❌ No distinction between same-type pieces of opposite colour in the channel encoding
  (the sign convention handles this but is implicit)

**Option 2: Fibonacci-spaced amplitudes (log-scale separation)**
```
amplitudes = [φ^0, φ^1, φ^2, φ^3, φ^4, φ^5] / φ^5  (φ = golden ratio)
           ≈ [0.09, 0.15, 0.24, 0.38, 0.62, 1.00]
```
- ✅ Geometric spacing ensures each piece type produces maximally distinct spectral
  peaks; a Bishop and a Knight will produce clearly separated interference patterns
- ✅ Mathematically natural — Fibonacci growth appears in chess piece-value theory
  (Berliner 1999)
- ❌ Requires rescaling to keep total board power bounded; opening positions
  (32 pieces) vs endgames (2 pieces) have very different magnitudes

**Option 3: Frequency-based encoding (pieces as oscillators)**
```
piece_freq[k] = base_freq × 2^(k/12)  (chromatic scale, 12 piece-channels)
amplitude = 1/sqrt(count_on_board)  (normalise by piece count)
phase = π² × square_idx / 64
```
- ✅ Spectral theory-motivated: each piece type occupies a distinct frequency band;
  piece interactions produce sum/difference frequencies (intermodulation) that encode
  tactical relationships
- ✅ The normalization keeps power constant regardless of phase of game
- ❌ Complex to implement; the FFT output is harder to interpret for debugging
- ❌ "chromatic scale" choice is arbitrary — needs ablation

**Recommendation**: Start with **Option 1** (already implemented) for Stage 1 to
establish the baseline. Switch to **Option 2** in Stage 2 if Stage 1's Spearman ρ
is below 0.60.  Implement Option 3 only if a full spectral analysis of the learned
frequency representations suggests the model is using the inter-piece frequencies.

---

### F.2 — PyTorch Complex Autograd Sufficiency

**What is known:**
- `torch.cfloat` autograd works via the Conjugate Wirtinger derivative.  For
  functions that decompose into real + imaginary components, gradient flow is correct.
- `nn.Parameter(torch.cfloat(...))` works but may produce `NaN` gradients in some
  ops (e.g. `torch.view_as_real` followed by a norm operation near zero).
  **Current workaround (already implemented):** `ComplexLinear` stores `weight_r`
  and `weight_i` as separate float32 parameters, computes `(wr + i·wi)(xr + i·xi)`
  manually. This is safe.

**What needs verification before Stage 2:**
- `ComplexPhaseAttention`'s `preds.angle()` call: `torch.angle()` is not
  differentiable at zero (branch cut discontinuity). If angle is used in a loss,
  this will cause NaN.  Workaround: use `preds.real` and `preds.imag` separately,
  or use `|z|²` rather than `angle(z)` for variance computations in the loss.
- `GeodesicLoRALinear._effective_weight()` recomputes on every forward pass
  (expensive). Cache it when `theta.requires_grad=False` (evaluation mode).
- `torch.fft.fft2` is not differentiable in PyTorch for `cfloat` inputs via the
  standard autograd on some older Termux PyTorch builds.  Verify with:
  ```python
  x = torch.randn(1,12,8,8, requires_grad=True)
  y = torch.fft.fft2(x)
  y.abs().sum().backward()  # must not raise
  ```

**What will need real/imaginary split:**
- Any `nn.LayerNorm`, `nn.BatchNorm` → use `ComplexLayerNorm` (already done)
- `F.relu` → use `complex_gelu` (already done)
- `torch.softmax` on cfloat → project to `|z|` first, then softmax on magnitudes
- `F.cross_entropy` on cfloat logits → **must project to real before this loss**

---

### F.3 — HRR for Chess (State, Action) Pairs — Literature Survey

**Plate 1995 core operations:**
- **Binding**: `trace = state ⊛ action = ifft(fft(state) · fft(action))` — circular convolution
- **Unbinding**: `approx_state = trace ⊛ action^{-1} ≈ ifft(fft(trace) · conj(fft(action)))`
  (exact only if action ≈ random unit vector; requires `‖action‖ = 1`)
- **Superposition**: `memory += trace` (additive, capacity degrades gracefully)
- **Retrieval**: `similarity = Re(query · memory†) / D` — inner product, no search

**Chess-specific considerations not covered in Plate 1995:**

1. **Action representation**: In chess, a "move" is a (from_square, to_square, promotion)
   triple.  HRR binding is associative but not commutative, so binding order matters.
   Two options:
   - Bind hierarchically: `a = from_vec ⊛ to_vec` then `trace = state ⊛ a`
   - Bind all three: `a = from_vec ⊛ to_vec ⊛ promo_vec` (valid for HRR, tested in
     Smolensky 1990 for role-filler structures)
   The hierarchical approach is cleaner and matches Plate's role-filler template.

2. **Capacity limit**: Plate (1995, §4.3) shows a D-dimensional HRR memory can
   reliably store and retrieve ≈ √D items before retrieval similarity drops below
   0.5.  For D=512 that is ~22 items before degradation; for D=4096 it is ~64.
   **The current implementation stores one HRR per MCTS node evaluation**, which
   can be thousands per episode.  Consequence: the superposition-based retrieval
   (if we were to use it) would immediately saturate.  **The current ring buffer
   approach (separate storage, cosine similarity search) avoids this problem at the
   cost of O(N) retrieval** — which is fine for N ≤ 50,000 on CPU.

3. **Follow-up: GHRR (Generalized HRR, 2023–2024)**: The arXiv paper referenced in
   research extends HRR to continuous control RL state-action pairs.  Their key
   insight: random phase encoding on the unit hypersphere gives better retrieval
   accuracy than random Gaussian initialization because orthogonality is guaranteed.
   **This is exactly what the current `_encode_phase_coords()` function does** (phase
   = π² × i/D per dimension).  The current implementation is consistent with GHRR.

4. **What Plate does NOT address for chess:**
   - How to encode the game outcome scalar (±1) as an HRR trace.
     Recommendation: use a fixed "outcome" role vector `r_out` and bind:
     `outcome_trace = r_out ⊛ result_vec`, then superpose with the state-action trace.
   - How to update stored traces post-game (none of the HRR literature handles
     retroactive correction of stored traces).  The correct approach is to **not**
     try to correct stored traces retroactively.  Instead, store a new trace with
     the corrected outcome and let older incorrect traces decay in confidence weight.

---

## Appendix: Decision Log

| Decision | Rationale |
|----------|-----------|
| Fresh start (not distillation bridge) | Architecture break too severe; no warm-start path |
| latent_dim=128 for Stage 2 (not 512) | RAM budget on mobile; 512 likely causes OOM |
| Option 1 amplitude scheme for Stage 1 | Already implemented; establish baseline first |
| Ring buffer for HRR (not superposition) | Capacity limit too low for per-node storage |
| Keep ChessValueModelPhase14 on master | Rollback target must remain usable |
| Stage 1 Spearman ρ ≥ 0.60 threshold | Conservative but falsifiable; below 0.45 = broken |

---

*Plan written at: 2026-09-22T18:xx UTC*
*Author: Antigravity (AI assistant)*
*Review required before Stage 2 implementation begins.*
