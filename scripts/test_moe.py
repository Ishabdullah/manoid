import sys
import os
import torch
import chess
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.phase14_mcts_planner import BatchingMCTSPlanner, ChessValueModelPhase14
from scripts.phase18_marathon import Phase16Planner, run_phase18

import glob
for ckpt in glob.glob("manoid_v2_ckpt_*.pt"):
    print(f"Moving {ckpt} out of the way...")
    os.rename(ckpt, ckpt + ".bak")

# Print param count
model = ChessValueModelPhase14()
params = sum(p.numel() for p in model.parameters())
print(f"Total Params: {params}")

BatchingMCTSPlanner.USE_GUMBEL_SEARCH = False
old_search = BatchingMCTSPlanner.search
def fast_search(self, root_fen, num_simulations=2000):
    return old_search(self, root_fen, 2)
BatchingMCTSPlanner.search = fast_search

old_search_q = Phase16Planner.search_with_q
def fast_search_q(self, root_fen, num_simulations=100):
    return old_search_q(self, root_fen, 2)
Phase16Planner.search_with_q = fast_search_q

print("Running fast 30 episode test for MoE...")
run_phase18(30)
