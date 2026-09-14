import sys
import os
import torch
import chess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.phase18_marathon import run_phase18

# Monkey patch planner search to be super fast
from scripts.phase14_mcts_planner import BatchingMCTSPlanner
BatchingMCTSPlanner.USE_GUMBEL_SEARCH = False
old_search = BatchingMCTSPlanner.search
def fast_search(self, root_fen, num_simulations=2000):
    return old_search(self, root_fen, 2)
BatchingMCTSPlanner.search = fast_search

from scripts.phase18_marathon import Phase16Planner
old_search_q = Phase16Planner.search_with_q
def fast_search_q(self, root_fen, num_simulations=100):
    return old_search_q(self, root_fen, 2)
Phase16Planner.search_with_q = fast_search_q

print("Running fast 30 episode test for AL...")
run_phase18(380)
