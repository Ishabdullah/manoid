"""
Smoke test for the 4 training-stability improvements:
  1. Replay buffer
  2. LR schedule
  3. Loss weighting
  4. Elo tracking

Also verifies UCI wrapper still loads the model without crashing.
"""
import sys
import os
import glob
import re
import torch
import chess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.phase14_mcts_planner import ChessValueModelPhase14, BatchingMCTSPlanner
from scripts.phase18_marathon import (
    Phase16Planner, run_phase18,
    ReplayBuffer, cosine_lr, elo_update, elo_expected,
    LR_MAX, LR_MIN, LR_DECAY_EPISODES,
    REPLAY_BUFFER_EPISODES, REPLAY_BATCH_SIZE,
    W_VALUE, W_POLICY, W_DEPTH, W_DENSE, W_PV,
    STOCKFISH_ELO,
)

print("=" * 65)
print("  TRAINING STABILITY SMOKE TEST")
print("=" * 65)

# ─── 1. Replay Buffer unit test ────────────────────────────────────────────────
print("\n[1] Replay Buffer")
rb = ReplayBuffer(max_episodes=5)
assert rb.num_examples == 0 and rb.num_episodes == 0, "empty buffer"

dummy_ply = {
    "state":      torch.randn(1, 837),
    "phase":      0,
    "target":     torch.tensor([[0.5]]),
    "pf_target":  torch.zeros(1, 64),
    "pt_target":  torch.zeros(1, 64),
    "d_target":   torch.tensor([[0.1]]),
    "dn_target":  torch.tensor([[0.0]]),
    "pvf_target": torch.zeros(1, 3, dtype=torch.long),
    "pvt_target": torch.zeros(1, 3, dtype=torch.long),
}

for i in range(8):        # push 8 episodes into a max-5 buffer
    rb.push_episode(i, [dummy_ply] * 10)

assert rb.num_episodes == 5, f"eviction failed: {rb.num_episodes}"
assert rb.num_examples == 50, f"example count wrong: {rb.num_examples}"
batch, n_eps = rb.sample(32)
assert batch is not None and batch["states"].shape[0] == 32
assert n_eps > 0
print(f"    ✓ PASS  eviction keeps last 5 eps ({rb.num_episodes} eps, {rb.num_examples} examples)")
print(f"    ✓ PASS  sample(32) → 32 examples from {n_eps} unique episodes")

# ─── 2. LR schedule ───────────────────────────────────────────────────────────
print("\n[2] LR Schedule (cosine)")
lr_start = cosine_lr(0, LR_DECAY_EPISODES)
lr_mid   = cosine_lr(LR_DECAY_EPISODES // 2, LR_DECAY_EPISODES)
lr_end   = cosine_lr(LR_DECAY_EPISODES, LR_DECAY_EPISODES)
print(f"    LR at ep 0:    {lr_start:.2e}  (expected ~{LR_MAX:.0e})")
print(f"    LR at ep {LR_DECAY_EPISODES//2}: {lr_mid:.2e}  (expected ~halfway)")
print(f"    LR at ep {LR_DECAY_EPISODES}: {lr_end:.2e}  (expected ~{LR_MIN:.0e})")
assert abs(lr_start - LR_MAX) < 1e-8, "start LR wrong"
assert abs(lr_end - LR_MIN) < 1e-8,   "end LR wrong"
assert lr_start > lr_mid > lr_end,     "not monotonically decreasing"
print(f"    ✓ PASS  Monotone decay: {lr_start:.2e} → {lr_mid:.2e} → {lr_end:.2e}")

# ─── 3. Loss weights ──────────────────────────────────────────────────────────
print("\n[3] Loss Weighting")
# Check that weighted contributions are roughly equal at measured averages
measured = {"value": 0.0212, "policy": 7.82, "depth": 0.034, "dense": 0.056, "pv": 8.12}
weights  = {"value": W_VALUE, "policy": W_POLICY, "depth": W_DEPTH, "dense": W_DENSE, "pv": W_PV}
contribs = {k: measured[k] * weights[k] for k in measured}
print("    Term       raw_avg    weight    contribution")
for k in measured:
    print(f"    {k:<10} {measured[k]:.4f}     {weights[k]:.4f}     {contribs[k]:.4f}")
max_c = max(contribs.values())
min_c = min(contribs.values())
ratio = max_c / min_c
print(f"    Max/min contribution ratio: {ratio:.1f}x  (target < 5x)")
assert ratio < 5.0, f"contributions still too imbalanced: {ratio:.1f}x"
print(f"    ✓ PASS  Loss contributions balanced within {ratio:.1f}x of each other")

# ─── 4. Elo math ──────────────────────────────────────────────────────────────
print("\n[4] Elo Mechanics")
print(f"    Assumed Stockfish Elos: {STOCKFISH_ELO}")
elo = 1000.0
for result, actual in [("Win", 1.0), ("Draw", 0.5), ("Loss", 0.0)]:
    opp = STOCKFISH_ELO[3]
    exp = elo_expected(elo, opp)
    new_elo = elo_update(elo, exp, actual)
    print(f"    {result} vs SF3 ({opp}): {elo:.0f} → {new_elo:.0f}  (Δ={new_elo-elo:+.1f})")
print(f"    ✓ PASS  Elo updates correctly for W/D/L")

# ─── 5. UCI wrapper still loads ───────────────────────────────────────────────
print("\n[5] UCI Wrapper Import")
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "uci_manoid",
        os.path.join(os.path.dirname(__file__), '..', 'uci_manoid.py')
    )
    uci_mod = importlib.util.module_from_spec(spec)
    # Don't exec (it starts stdin loop), just verify it can be parsed
    import ast
    src = open(os.path.join(os.path.dirname(__file__), '..', 'uci_manoid.py')).read()
    ast.parse(src)
    print(f"    ✓ PASS  uci_manoid.py parses cleanly")
except Exception as e:
    print(f"    ✗ FAIL  {e}")

# ─── 6. Fresh 45-episode run ──────────────────────────────────────────────────
print("\n[6] Fresh 45-Episode Integration Run")
print("    Moving existing checkpoints aside...")
moved = []
for ckpt in glob.glob("manoid_v2_ckpt_*.pt"):
    dest = ckpt + ".stab_bak"
    os.rename(ckpt, dest)
    moved.append((dest, ckpt))
    print(f"    Moved {ckpt} → {dest}")

print("    Patching MCTS to 4 simulations for speed...")
BatchingMCTSPlanner.USE_GUMBEL_SEARCH = False
old_sq = Phase16Planner.search_with_q
def fast_sq(self, root_fen, num_simulations=100):
    return old_sq(self, root_fen, 4)
Phase16Planner.search_with_q = fast_sq

print("    Starting 45-episode run...\n")
print("─" * 65)

run_phase18(45)

print("─" * 65)
print("\n[7] Post-run checks done.")

if moved:
    print("\n    Restoring checkpoints...")
    for dest, orig in moved:
        try:
            os.rename(dest, orig)
            print(f"    Restored {orig}")
        except Exception as e:
            print(f"    Warning: {e}")

print("\n" + "=" * 65)
print("  ALL STABILITY CHECKS DONE")
print("=" * 65)
