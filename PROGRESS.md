# Project Progress: Manoid

## Phase 0 — The Termux-to-Colab Bridge
**Status:** Complete
**Goal:** A resumable connection between the local S24 Ultra Termux environment and Colab compute that survives disconnects.

### Steps:
- [x] Research: Confirm colab-cli syntax.
- [x] Architect: Checkpoint contract.
- [x] Implementer: Build scripts/colab_run.sh and smoke-test.
- [x] Verifier: Kill job mid-run, resume, and show terminal output.

## Phase 1 — The Core Intelligence (Seed or Scratch)
**Status:** Complete
**Goal:** Establish the foundational model on Colab (or locally if small enough).

### Steps:
- [x] Research: Evaluate whether to train a tiny foundational world model from scratch (50M parameters) or use an open-weight seed model (1B-3B quantized).
  - *Findings:* Decided to build a custom ~12M-50M parameter Joint Embedding Predictive Architecture (JEPA). This isolates our thesis (sample efficiency via episodic/causal learning) from pre-trained LLM heuristics. The architecture features an Encoder for latent representation and a Predictor for next-state simulation. We will strictly pre-train only on basic environment physics to avoid corrupting the model with the Phase 7 test rule.
- [x] Architect: Define the interface for generating predictions, extracting embeddings, and flagging uncertainty.
- [x] Implementer: Build the wrapper.
- [x] Verifier: Run the wrapper and paste real output into the log.
  - *Output:*
    ```
    --- Phase 1: Live Verifier ---
    Model initialized with 12,116,992 trainable parameters.
    
    Executing forward pass...
    Encoded shape (z_t): torch.Size([4, 512]) | Expected: (4, 512)
    Predicted next shape (z_pred_next): torch.Size([4, 512]) | Expected: (4, 512)
    Surprise metric shape: torch.Size([4]) | Expected: (4,)
    Surprise values: [0.2723057270050049, 0.3145035207271576, 0.28503814339637756, 0.27288946509361267]
    
    --- Live Verification Success! ---
    ```

## Phase 2 — Fast-Binding Episodic Memory
**Status:** In Progress
**Goal:** The model can instantly store one-shot experiences without retraining weights.

### Steps:
- [ ] Architect: Design the Complementary Learning System episodic cache. Schema must capture [state, action, outcome, confidence].
- [ ] Implementer: Build the memory store.
- [ ] Verifier: Introduce a novel fact to the system. Prove that on the second query, the agent retrieves it from the episodic store instantly.
