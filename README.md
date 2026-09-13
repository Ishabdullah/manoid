# Manoid ♟️📱
**An Edge-Optimized Autonomous Reinforcement Learning Chess Engine built for mobile hardware.**

[![Platform](https://img.shields.io/badge/Platform-Termux%20%7C%20Android-green.svg)]()
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-purple.svg)]()

Manoid is an experimental, fully autonomous chess engine designed from the ground up to train and run natively on high-end mobile hardware (specifically Snapdragon 8 Gen 3 via Termux). By combining a lightweight Dual-Head Neural Network with a localized MCTS planner and continuous Stockfish-driven distillation, Manoid effectively builds its own tactical and positional intuition directly on the edge.

## 🧠 Architecture
*   **Dual-Head Neural Network:** A hyper-efficient (~1.2M–1.5M parameters) network architecture.
    *   **Value Head:** Positional evaluation mapped to centipawns, optimized via MSE loss.
    *   **Policy Head:** A flat 4096-dimensional action space (`nn.Linear(128, 4096)`) mapped via Cross-Entropy loss to predict move probability priors.
*   **Batching MCTS Planner:** Utilizes batched leaf-node evaluations for high-throughput Monte Carlo Tree Search, leveraging PUCT to aggressively prune low-probability lines.
*   **LoRA Weight Adaptation:** Low-Rank Adaptation style matrices enable memory-efficient backpropagation, allowing the engine to learn on-device without exhausting mobile RAM.

## ⚔️ Training Methodology & Curriculum
Manoid is trained iteratively through self-play and supervised distillation.
*   **Oracle Distillation:** Queries Stockfish continuously to bootstrap the policy (`bestmove`) and value (centipawn proxy) targets.
*   **Dynamic Sparring Curriculum:** Features a randomized 3-tier difficulty gauntlet against Stockfish (Levels 2–5) to prevent plateauing and overfitting to weak or perfect play.
*   **Phase 16.10 Guardrail Override:** Forces continuous Consolidation (TRAIN) steps after every episode, bypassing legacy Two-Strike plasticity gates to accelerate Policy Head distillation during deep grinds.

## 🔌 Interfaces & Integration
Manoid is not just a training loop; it is a fully playable engine.
*   **Universal Chess Interface (UCI):** Includes a standalone wrapper (`uci_manoid.py`) that handles standard UCI protocol commands (`uci`, `isready`, `position`, `go`, etc.).
*   **Mobile GUI Integration:** Designed to hook directly into mobile chess apps like **DroidFish** using a `socat` network loopback bridge (`127.0.0.1:3333`).

## 📂 Project Structure
*   `scripts/phase18_marathon.py` - The core "Deep Grind" continuous training loop.
*   `uci_manoid.py` - Standalone UCI engine wrapper.
*   `scripts/phase14_mcts_planner.py` - The Dual-Head Network and Batching MCTS implementation.
*   `stream_marathon.sh` - Color-coded telemetry monitor for observing autonomous training.
*   `monitor_marathon.sh` - Lightweight passive pulse check for the background marathon.

## 🚀 Quickstart & Usage Guide

### 1. Launch the Training Marathon
Run the self-play distillation loop in the background:
```bash
python -u scripts/phase18_marathon.py >> marathon.log 2>&1 &
```
*Monitor the training telemetry in real-time:*
```bash
./stream_marathon.sh
```

### 2. Run the UCI Protocol Interactively
Test the engine manually via standard input:
```bash
python uci_manoid.py
```
*Inside the prompt, test standard commands:*
```text
uci
isready
position startpos moves e2e4 e7e5
go
quit
```

### 3. Deploy Mobile Socat Bridge (DroidFish Integration)
To expose Manoid to Android GUI apps, launch a local TCP socket using `socat`:
```bash
socat TCP-LISTEN:3333,reuseaddr,fork EXEC:"python uci_manoid.py"
```
*Next, open DroidFish, add a network engine at `127.0.0.1` port `3333`, and start playing!*
