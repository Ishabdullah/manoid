import sys
import os
import time
import torch
import chess
import chess.engine
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.environment.full_chess_env import FullChessEnv
from scripts.phase14_mcts_planner import ChessValueModelPhase14
from scripts.phase18_marathon import Phase16Planner

def run_experiment(use_gumbel, num_episodes=30):
    env = FullChessEnv()
    value_model = ChessValueModelPhase14()
    
    import glob
    import re
    ckpts = glob.glob("manoid_v2_ckpt_*.pt")
    if ckpts:
        ckpts.sort(key=lambda x: int(re.search(r"manoid_v2_ckpt_(\d+)\.pt", x).group(1)))
        latest_ckpt = ckpts[-1]
        state_dict = torch.load(latest_ckpt, weights_only=True)
        value_model.load_state_dict(state_dict, strict=False)
        print(f"Loaded {latest_ckpt}")
    else:
        print("No checkpoint found.")
        
    value_model.eval()
    
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    engine.configure({"Skill Level": 3})
    
    planner = Phase16Planner(env, value_model, engine, batch_size=8)
    planner.USE_GUMBEL_SEARCH = use_gumbel
    
    wins = 0
    losses = 0
    draws = 0
    total_moves = 0
    
    # 30 standard FENs or starting FEN
    starting_fens = [chess.STARTING_FEN] * num_episodes
    
    for ep in range(num_episodes):
        board = chess.Board(starting_fens[ep])
        moves = 0
        agent_color = chess.WHITE if ep % 2 == 0 else chess.BLACK
        
        while not board.is_game_over() and moves < 100:
            if board.turn == agent_color:
                best_move, _ = planner.search_with_q(board.fen(), num_simulations=50)
            else:
                result = engine.play(board, chess.engine.Limit(time=0.01))
                best_move = result.move
                
            board.push(best_move)
            moves += 1
            
        total_moves += moves
        result = board.result()
        if result == '1-0':
            if agent_color == chess.WHITE: wins += 1
            else: losses += 1
        elif result == '0-1':
            if agent_color == chess.BLACK: wins += 1
            else: losses += 1
        else:
            draws += 1
            
        print(f"[{'Gumbel' if use_gumbel else 'PUCT'}] Ep {ep+1}/{num_episodes}: {'W' if result == '1-0' and agent_color == chess.WHITE or result == '0-1' and agent_color == chess.BLACK else 'L' if result != '1/2-1/2' else 'D'} ({moves} moves)")
        
    engine.quit()
    print(f"--- {'Gumbel' if use_gumbel else 'PUCT'} Results ---")
    print(f"W: {wins} | L: {losses} | D: {draws}")
    print(f"Avg game length: {total_moves / num_episodes:.1f} moves")
    
if __name__ == "__main__":
    print("Running PUCT...")
    run_experiment(use_gumbel=False, num_episodes=30)
    print("\nRunning Gumbel...")
    run_experiment(use_gumbel=True, num_episodes=30)
