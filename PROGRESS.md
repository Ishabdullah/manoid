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
**Status:** Complete
**Goal:** The model can instantly store one-shot experiences without retraining weights.

### Steps:
- [x] Architect: Design the Complementary Learning System episodic cache. Schema must capture [state, action, outcome, confidence].
- [x] Implementer: Build the memory store.
- [x] Verifier: Introduce a novel fact to the system. Prove that on the second query, the agent retrieves it from the episodic store instantly.
  - *Output:*
    ```
    --- Phase 2: Live Verifier ---
    Initialized PyTorch Tensor Buffer Episodic Cache.
    
    Querying cache before storing...
    Result before store: None
    
    Storing novel fact [state, action, outcome, confidence]...
    Memory size after store: 1
    
    Querying cache after storing...
    Similarity score: 1.0000
    Retrieved confidence: 0.9900
    MSE between stored and retrieved outcome: 0.000000
    
    --- Live Verification Success! ---
    ```

## Phase 3 — Environment & Causal World Model
**Status:** In Progress
**Goal:** A playground where actions have observable consequences, so the agent can test hypotheses.

### Steps:
- [ ] Research: Evaluate a minimal environment (e.g., MiniGrid, Crafter) vs. a custom symbolic grid.
- [ ] Architect: Define how the agent interacts (do(X) interventions) and logs transitions.
- [ ] Implementer: Wire the environment, log real transitions.
- [ ] Verifier: Prove the agent's internal state tracking matches the environment's ground truth.
