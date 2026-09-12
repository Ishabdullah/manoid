import sys
import os
import time
import chess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.environment.micro_chess_env import MicroChessEnv
from scripts.chess_mcts_planner import ChessMCTSPlanner

def run_smoke_test():
    print("=== Phase 9: Micro-Chess Bridge (Smoke Test) ===")
    print("Scenario: K+R vs K (Mate in 2)")
    print("FEN: k7/8/2K5/8/8/8/8/7R w - - 0 1")
    print("--------------------------------------------------")
    
    fen = "k7/8/2K5/8/8/8/8/7R w - - 0 1"
    env = MicroChessEnv(fen)
    planner = ChessMCTSPlanner(env)
    
    board = chess.Board(fen)
    
    step = 1
    while not board.is_checkmate() and step <= 5:
        start_time = time.time()
        print(f"[Step {step}] White to move. Board FEN: {board.fen()}")
        
        # MCTS
        best_move = planner.search(board.fen(), num_simulations=200)
        
        end_time = time.time()
        
        print(f"  -> MCTS 200 iterations completed in {end_time - start_time:.2f} seconds.")
        print(f"  -> Chosen Move: {board.san(best_move)}")
        
        board.push(best_move)
        
        if board.is_checkmate():
            print("\n[Final Result]")
            print(f"SUCCESS! Checkmate delivered on step {step}.")
            break
            
        # Black's turn
        start_time = time.time()
        black_move = planner.search(board.fen(), num_simulations=200)
        end_time = time.time()
        
        print(f"  -> Black replies: {board.san(black_move)} (thought for {end_time - start_time:.2f}s)")
        board.push(black_move)
        
        step += 1
        
    if not board.is_checkmate():
        print("\n[Final Result]")
        print("FAILED to deliver checkmate within 5 steps.")

if __name__ == "__main__":
    run_smoke_test()
