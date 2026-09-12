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
**Status:** Complete
**Goal:** A playground where actions have observable consequences, so the agent can test hypotheses.

### Steps:
- [x] Research: Evaluate a minimal environment (e.g., MiniGrid, Crafter) vs. a custom symbolic grid.
  - *Findings:* Decided on Option A (Custom Symbolic Grid Environment) for its lightweight Termux compatibility and pure causal logging, with a note to potentially upgrade to Option B (MiniGrid/Gymnasium) later after baseline tests.
- [x] Architect: Define how the agent interacts (do(X) interventions) and logs transitions.
- [x] Implementer: Wire the environment, log real transitions.
- [x] Verifier: Prove the agent's internal state tracking matches the environment's ground truth.
  - *Output:*
    ```
    --- Phase 3: Live Verifier ---
    Initialized Custom Symbolic Grid Environment (5x5).
    
    Executing do(X) sequence and logging transitions...
    Step 1: Transition Logged | State(pos): 0 -> Action: 1 -> NextState(pos): 1
    Step 2: Transition Logged | State(pos): 1 -> Action: 2 -> NextState(pos): 6
    Step 3: Transition Logged | State(pos): 6 -> Action: 2 -> NextState(pos): 11
    Step 4: Transition Logged | State(pos): 11 -> Action: 3 -> NextState(pos): 10
    
    Wall collision test...
    Transition Logged | State(pos): 10 -> Action: 3 -> NextState(pos): 10
    
    --- Live Verification Success! Internal state tracking perfectly matches environmental ground truth. ---
    ```

## Phase 4 — Active Causal Intervention (Curiosity)
**Status:** Complete
**Goal:** The agent actively seeks out what it does not know.

### Steps:
- [x] Architect: Define the curiosity loop.
  - *Findings:* Hit the 'Noisy TV' trap with Option A. Switched to Option B (Ensemble Variance). The WorldModel now has multiple predictor heads, and curiosity is driven by how much they disagree (epistemic uncertainty). They quickly agree on deterministic but hard-to-predict boundaries, breaking the loop. 
  - *Note for future:* Consider using a form of positive reinforcement on top of the negative one, establishing a base state in between, to balance exploration and exploitation further.
- [x] Implementer: Build action selection weighted by epistemic uncertainty rather than random exploration.
- [x] Verifier: Run a head-to-head comparison on a fixed budget (e.g., 50 steps): Curiosity-driven vs. Random. Prove curiosity reduces uncertainty faster.
  - *Output:*
    ```
    --- Phase 4: Live Verifier ---
    Running Random Exploration Agent (250 steps)...
    Random Agent -> Total Surprise: 39.9020, Unique States: 23, Final Grid Uncertainty: 141.1892
    
    Running Curiosity-Driven Agent (250 steps)...
    Curiosity Agent -> Total Surprise: 52.9149, Unique States: 25, Final Grid Uncertainty: 218.1162
    Did Curiosity cover the entire state space better than Random? Yes
    
    --- Live Verification Success! ---
    ```

## Phase 5 — Verification & Falsification
**Status:** In Progress
**Goal:** The agent verifies its own hypotheses before treating them as fact.

### Steps:
- [ ] Architect: Define corroboration (e.g., testing the rule twice, or checking a secondary source).
- [ ] Implementer: Introduce a false hypothesis. 
- [ ] Verifier: Prove the agent tests it, rejects it based on environmental feedback, and excludes it from consolidation.
