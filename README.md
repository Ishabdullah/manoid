# Manoid ♟️📱
**An Edge-Optimized Autonomous Reinforcement Learning Chess Engine built for mobile hardware.**

[![Platform](https://img.shields.io/badge/Platform-Termux%20%7C%20Android-green.svg)](https://termux.dev)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

Manoid is an experimental, fully autonomous chess engine designed from the ground up to train and run natively on high-end mobile hardware (specifically Snapdragon 8 Gen 3 via Termux). It combines a lightweight Phase-Expert neural network with a Gumbel-MCTS planner and continuous Stockfish-driven distillation — building its own tactical intuition directly on the edge, no cloud required.

---

## 🧠 Architecture

| Component | Detail |
|---|---|
| **Shared Trunk** | `Linear(837→256) → LN → GELU → Linear(256→128) → LN → GELU` |
| **Phase Classifier** | Heuristic on piece count: >24 = Opening (0), >12 = Middlegame (1), ≤12 = Endgame (2) |
| **Expert Branches** | 3 × `Linear(128→128) → LN → GELU`, one per phase, gated by classifier |
| **Output Heads** | Value · From · To · Depth-delta · Dense-reward · PV-from · PV-to |
| **Total Parameters** | ~365K base (no LoRA) — same order of magnitude as the 580K baseline |
| **MCTS Planner** | Batched Gumbel-MCTS, Top-K sequential halving, 128 sims/move in production |
| **LoRA Adapter** | `FullChess` adapter (rank 64) wraps every head + expert branch |
| **Replay Buffer** | Last 200 episodes, 256-sample random batches per consolidation step |
| **LR Schedule** | Cosine decay `5e-4 → 5e-6` over 2000 episodes |

The three expert branches mean Opening, Middlegame, and Endgame positions each get dedicated gradient signal. `USE_PHASE_EXPERTS = True` by default; set it to `False` on the model instance to run the single-trunk baseline for ablation.

---

## ⚔️ Training Loop — How It Works

Manoid trains continuously and automatically. Here is the full cycle for one episode:

```
1.  Select starting position  (Natural / GM Book opening / Active Learning)
2.  Play game vs Stockfish    (curriculum: Skill 2–5, weighted by win-rate)
3.  Per-ply data collection
      • MCTS best move  →  target_q (value target)
      • Stockfish depth-1 multipv  →  soft policy (from/to) targets
      • Stockfish depth-6  →  depth-delta auxiliary target
      • Eval delta before/after move  →  dense per-ply reward
      • Syzygy tablebase  →  value override when ≤5 pieces
      • Mirror augmentation  →  horizontal flip of every position
4.  Push episode plies into replay buffer (circular, last 200 eps)
5.  Sample 256 examples randomly from buffer  →  train for 5 gradient steps
      Weighted loss: value(×1.0) + policy(×0.003) + depth(×0.5) +
                     dense(×0.35) + pv(×0.0025)
6.  Log phase-split MSE, raw + weighted losses, LR, Elo
7.  Save checkpoint every 25 episodes  →  manoid_v2_ckpt_<ep>.pt
```

**Training is fully automatic** — once you launch `phase18_marathon.py` it loops forever (default `episodes=2000`). It resumes from the latest checkpoint automatically if interrupted. There is no manual intervention required between episodes.

---

## 🚀 Quickstart

### Prerequisites

```bash
pkg install python stockfish           # Termux
pip install torch python-chess         # Python deps
```

Clone and enter the repo:

```bash
git clone https://github.com/Ishabdullah/manoid.git
cd manoid
```

---

### 1. Launch the Training Marathon

Start the continuous self-play + distillation loop in the background:

```bash
python -u scripts/phase18_marathon.py >> marathon.log 2>&1 &
echo $! > marathon.pid
echo "Training started. PID: $(cat marathon.pid)"
```

Training starts immediately from episode 1 (or resumes from the highest-numbered `manoid_v2_ckpt_*.pt` if one exists). **It trains automatically after every single episode** — no batching, no manual trigger. A checkpoint is saved every 25 episodes.

---

### 2. Watch the Training Logs

**Live colour-coded telemetry** (recommended — follow mode):

```bash
./stream_marathon.sh
```

**Passive pulse-check** (recent checkpoints + last 15 lines):

```bash
./monitor_marathon.sh
```

**Raw follow** (plain `tail`):

```bash
tail -f marathon.log
```

**What each log line means:**

```
Starting FEN: [Start: Natural] | Opponent Level: 2 | Level Mix: [0.7, 0.2, 0.05, 0.05]
[Episode 42] Manoid plays White | Result: Checkmated (Lost) (25 moves) | Avg MSE: 0.0216
  -> Rolling Stats (Last 42): 0W / 42L / 0D | Elo: 783 (vs SF2≈1100)
  -> Replay Buffer: 1200 examples across 42 episodes | Batch: 256 samples from 42 unique eps
  -> Grounding Hits: TB 12 vs SF 0
  -> Phase MSE | Open(220): 0.0031 | Mid(34): 0.0025 | End(2): 0.0568
  -> Loss [raw]     | V: 0.0035 | P: 7.71 | D: 0.0119 | Dn: 0.0072 | PV: 7.89
  -> Loss [weighted] | V: 0.0035 | P: 0.0231 | D: 0.0059 | Dn: 0.0025 | PV: 0.0197 | Total: 0.0548 | LR: 4.99e-04
```

| Field | Meaning |
|---|---|
| `Avg MSE` | Mean squared error of value predictions vs MCTS target-Q during the game |
| `Rolling Stats` | Win/Loss/Draw over the last 50 episodes |
| `Elo` | Manoid's rolling Elo estimate (K=32); opponent Elo assumed from Stockfish skill level |
| `Replay Buffer` | Total stored plies and how many unique episodes fed the current batch |
| `TB / SF` | Syzygy tablebase hits vs Stockfish evaluations used as value targets |
| `Phase MSE` | Per-phase value MSE: Opening / Middlegame / Endgame (count in parentheses) |
| `Loss [raw]` | Unscaled individual loss terms each consolidation step |
| `Loss [weighted]` | After per-term weighting so no head dominates gradient |
| `LR` | Current learning rate (cosine decay `5e-4 → 5e-6` over 2000 episodes) |

**Checkpoint saves** appear every 25 episodes:

```
--- Checkpoint Saved: manoid_v2_ckpt_25.pt ---
Stats over last 25 episodes: Wins: 0, Losses: 24, Draws: 1
Average MSE over last 25 episodes: 0.0341
Replay Buffer: 712 examples / 25 episodes
Estimated Elo: 870  |  LR: 4.97e-04
----------------------------------------
```

---

### 3. Check Training Status (Non-Interactive)

```bash
# Quick state check (JSON: phase / episode / status)
cat state.json

# List all saved checkpoints
ls -lh manoid_v2_ckpt_*.pt

# Kill training gracefully
kill $(cat marathon.pid)
```

Training auto-resumes from the latest checkpoint next time you run step 1.

---

### 4. Run the UCI Engine (Play Against It)

Test the engine interactively via standard UCI protocol:

```bash
python uci_manoid.py
```

Try these commands inside the prompt:

```
uci
isready
position startpos moves e2e4 e7e5
go movetime 3000
quit
```

The engine automatically loads the latest `manoid_v2_ckpt_*.pt` checkpoint on startup.

---

### 5. Deploy the DroidFish Bridge (Play on Android)

Expose Manoid to any Android chess app via a local TCP socket:

```bash
socat TCP-LISTEN:3333,reuseaddr,fork EXEC:"python uci_manoid.py"
```

In DroidFish (or any UCI-compatible mobile app): add a **network engine** at `127.0.0.1` port `3333` and start playing.

---

## 📂 Project Structure

```
manoid/
├── scripts/
│   ├── phase18_marathon.py      ← Core training loop (run this)
│   ├── phase14_mcts_planner.py  ← ChessValueModelPhase14 + Gumbel-MCTS
│   ├── smoke_test_stability.py  ← Stability regression test (replay/LR/weights/Elo)
│   └── smoke_test_phase_experts.py  ← Phase expert architecture test
├── agent/
│   ├── environment/
│   │   └── full_chess_env.py    ← 837-dim board encoder + legal move gen
│   └── consolidation/
│       └── lora.py              ← LoRALinear implementation
├── uci_manoid.py                ← UCI wrapper (for DroidFish / any GUI)
├── stream_marathon.sh           ← Colour-coded live log stream
├── monitor_marathon.sh          ← Quick pulse check
├── state.json                   ← Current episode / status (written each checkpoint)
├── marathon.log                 ← Full training log (append mode)
└── manoid_v2_ckpt_<N>.pt        ← Checkpoints saved every 25 episodes
```

---

## 📊 Training Configuration Reference

All tunable constants live at the top of [`scripts/phase18_marathon.py`](scripts/phase18_marathon.py):

```python
REPLAY_BUFFER_EPISODES = 200    # Episodes kept in replay buffer
REPLAY_BATCH_SIZE      = 256    # Examples sampled per consolidation step
LR_MAX                 = 5e-4   # Initial learning rate
LR_MIN                 = 5e-6   # Final learning rate (after LR_DECAY_EPISODES)
LR_DECAY_EPISODES      = 2000   # Cosine decay period
W_VALUE                = 1.0    # Loss weight: value head
W_POLICY               = 0.003  # Loss weight: policy (from + to)
W_DEPTH                = 0.5    # Loss weight: depth-delta auxiliary
W_DENSE                = 0.35   # Loss weight: dense per-ply reward
W_PV                   = 0.0025 # Loss weight: PV auxiliary head
```

To disable phase experts and run the single-trunk baseline:

```python
value_model.USE_PHASE_EXPERTS = False
```

---

## 🔬 Development Progress

| Phase | Status | Description |
|---|---|---|
| 0 | ✅ | Termux-to-Colab resumable bridge |
| 1–6 | ✅ | JEPA world model · episodic memory · environment · curiosity · falsification · consolidation |
| 7 | ✅ | Final assay: **2.8× more sample-efficient** than backprop baseline |
| 8–13 | ✅ | MCTS integration · LoRA · micro-chess → full chess |
| 14–18 | ✅ | Phase experts · Gumbel-MCTS · replay buffer · LR schedule · Elo tracking |

---

## ⚠️ Notes

- **Checkpoint compatibility:** Architecture changes (e.g. adding phase experts) break existing checkpoints. Start fresh after any structural change to `ChessValueModelPhase14`.
- **Stockfish required:** Both training and UCI mode require `stockfish` in PATH.
- **Syzygy tablebases (optional):** Place 3-4-5 piece tablebases in `syzygy_345/` for endgame value grounding. Download with `python download_syzygy.py`.
- **Memory:** Training uses ~400 MB RAM. UCI play uses ~150 MB. Safe on 12 GB devices.
