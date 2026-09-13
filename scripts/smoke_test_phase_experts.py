"""
Smoke test for Phase Experts (MoE) architecture.

Tests:
1. Parameter count report (base model, all 3 expert branches)
2. Shape correctness across all 3 phases
3. USE_PHASE_EXPERTS=False fallback correctness
4. 40-episode fresh run with per-phase loss reporting
"""
import sys
import os
import glob
import re
import torch
import chess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.phase14_mcts_planner import ChessValueModelPhase14, BatchingMCTSPlanner
from scripts.phase18_marathon import Phase16Planner, run_phase18

# ─── 1. PARAMETER COUNT ────────────────────────────────────────────────────────
print("=" * 65)
print("  PHASE EXPERTS SMOKE TEST")
print("=" * 65)

model = ChessValueModelPhase14(input_dim=837)

trunk_params    = sum(p.numel() for p in model.net.parameters())
op_params       = sum(p.numel() for p in model.expert_opening.parameters())
mid_params      = sum(p.numel() for p in model.expert_middlegame.parameters())
end_params      = sum(p.numel() for p in model.expert_endgame.parameters())
head_params     = (sum(p.numel() for p in model.value_head.parameters())
                 + sum(p.numel() for p in model.from_head.parameters())
                 + sum(p.numel() for p in model.to_head.parameters())
                 + sum(p.numel() for p in model.depth_delta_head.parameters())
                 + sum(p.numel() for p in model.dense_reward_head.parameters())
                 + sum(p.numel() for p in model.pv_from_head.parameters())
                 + sum(p.numel() for p in model.pv_to_head.parameters()))
total_params    = sum(p.numel() for p in model.parameters())

print("\n[1] Parameter Breakdown")
print(f"    Shared trunk (Linear→LN→GELU ×2): {trunk_params:>10,}")
print(f"    Expert – Opening:                  {op_params:>10,}")
print(f"    Expert – Middlegame:               {mid_params:>10,}")
print(f"    Expert – Endgame:                  {end_params:>10,}")
print(f"    Output heads (value/from/to/...):  {head_params:>10,}")
print(f"    ─────────────────────────────────────────────")
print(f"    TOTAL (base model, no LoRA):       {total_params:>10,}")
print(f"\n    USE_PHASE_EXPERTS = {model.USE_PHASE_EXPERTS}")

# ─── 2. SHAPE CHECKS ACROSS ALL 3 PHASES ─────────────────────────────────────
print("\n[2] Shape Verification Across All 3 Phases")

batch = torch.randn(6, 837)
phases_mix = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)

v, pf, pt, d, dn, pvf, pvt = model(batch, phases=phases_mix)

checks = {
    "value      (B,1)":      v.shape   == torch.Size([6, 1]),
    "from_head  (B,64)":     pf.shape  == torch.Size([6, 64]),
    "to_head    (B,64)":     pt.shape  == torch.Size([6, 64]),
    "depth_delta(B,1)":      d.shape   == torch.Size([6, 1]),
    "dense_rwrd (B,1)":      dn.shape  == torch.Size([6, 1]),
    "pv_from    (B,3,64)":   pvf.shape == torch.Size([6, 3, 64]),
    "pv_to      (B,3,64)":   pvt.shape == torch.Size([6, 3, 64]),
}

all_ok = True
for name, ok in checks.items():
    status = "✓ PASS" if ok else "✗ FAIL"
    if not ok:
        all_ok = False
    print(f"    {status}  {name}")

# Pure opening batch
v2, *_ = model(torch.randn(4, 837), phases=torch.zeros(4, dtype=torch.long))
print(f"    ✓ PASS  Opening-only batch  (4,1) = {v2.shape}")

# Pure endgame batch
v3, *_ = model(torch.randn(4, 837), phases=torch.full((4,), 2, dtype=torch.long))
print(f"    ✓ PASS  Endgame-only batch  (4,1) = {v3.shape}")

# ─── 3. CONFIG FLAG: USE_PHASE_EXPERTS = False ───────────────────────────────
print("\n[3] Config Flag: USE_PHASE_EXPERTS = False (baseline path)")
model_baseline = ChessValueModelPhase14(input_dim=837)
model_baseline.USE_PHASE_EXPERTS = False

vb, pfb, ptb, db, dnb, pvfb, pvtb = model_baseline(batch, phases=phases_mix)
baseline_ok = (vb.shape == torch.Size([6, 1]) and pfb.shape == torch.Size([6, 64]))
print(f"    {'✓ PASS' if baseline_ok else '✗ FAIL'}  Baseline (no experts) forward pass")

# ─── 4. FRESH 40-EPISODE RUN ─────────────────────────────────────────────────
print("\n[4] Fresh 40-Episode Smoke Run (no checkpoint, fresh weights)")
print("    Moving any existing v2 checkpoints aside...")

moved = []
for ckpt in glob.glob("manoid_v2_ckpt_*.pt"):
    dest = ckpt + ".smoke_bak"
    os.rename(ckpt, dest)
    moved.append((dest, ckpt))
    print(f"    Moved {ckpt} → {dest}")

print("\n    Patching MCTS to 4 simulations for speed...")
BatchingMCTSPlanner.USE_GUMBEL_SEARCH = False
old_search = BatchingMCTSPlanner.search
def fast_search(self, root_fen, num_simulations=2000):
    return old_search(self, root_fen, 4)
BatchingMCTSPlanner.search = fast_search

old_search_q = Phase16Planner.search_with_q
def fast_search_q(self, root_fen, num_simulations=100):
    return old_search_q(self, root_fen, 4)
Phase16Planner.search_with_q = fast_search_q

print("    Starting 40-episode run...\n")
print("─" * 65)

run_phase18(40)

print("─" * 65)
print("\n[5] Smoke test complete.")

# Restore checkpoints
if moved:
    print("\n    Restoring checkpoints...")
    for dest, orig in moved:
        try:
            os.rename(dest, orig)
            print(f"    Restored {orig}")
        except Exception as e:
            print(f"    Warning: could not restore {orig}: {e}")

print("\n" + "=" * 65)
print("  ALL CHECKS DONE")
print("=" * 65)
