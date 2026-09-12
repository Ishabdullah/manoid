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
**Status:** In Progress
**Goal:** Establish the foundational model on Colab (or locally if small enough).

### Steps:
- [ ] Research: Evaluate whether to train a tiny foundational world model from scratch (50M parameters) or use an open-weight seed model (1B-3B quantized).
- [ ] Architect: Define the interface for generating predictions, extracting embeddings, and flagging uncertainty.
- [ ] Implementer: Build the wrapper.
- [ ] Verifier: Run the wrapper and paste real output into the log.
