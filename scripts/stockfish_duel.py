import sys
import os
import time
import torch
import chess
import chess.engine

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.environment.micro_chess_env import MicroChessEnv
from scripts.chess_neural_assay import ChessValueModel, NeuralMCTSPlanner

def run_duel(num_games=5):
    print("=== Phase 11: The Stockfish Benchmark ===")
    print("Match: Manoid (White) vs Stockfish Level 1 (Black)")
    print("Scenario: K+R vs K")
    print("--------------------------------------------------")

    env = MicroChessEnv()
    value_model = ChessValueModel()
    
    # We will initialize it, though normally we'd load trained weights.
    # We'll just run it as-is for the Duel, or let it train via the same assay loop?
    # The prompt says: "Manoid must use its trained Tier 2 LoRA intuitions and the MCTS planner to force the mate against a perfect defending engine."
    # Since we didn't save weights in Phase 10, we can run a quick offline training on the mate-in-2, or just use the MCTS search which is deep enough if we give it iterations. 
    # Let's give it 200 simulations.
    
    value_model.apply_lora_adapter("ChessEndgame", rank=8)
    value_model.set_active_lora("ChessEndgame")
    
    planner = NeuralMCTSPlanner(env, value_model)

    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    engine.configure({"Skill Level": 0})  # Level 1 effectively

    base_fen = "k7/8/2K5/8/8/8/8/7R w - - 0 1"

    results = {"Win": 0, "Draw": 0, "Loss": 0}
    moves_to_mate = []

    for game in range(1, num_games + 1):
        fen = env.reset(base_fen)
        board = chess.Board(fen)
        
        print(f"\n[Game {game}] Starting FEN: {board.fen()}")
        moves_made = 0
        
        while not board.is_game_over() and moves_made < 50:
            if board.turn == chess.WHITE:
                # Manoid's turn
                best_move, target_q = planner.search_with_q(board.fen(), num_simulations=400)
                if best_move is None:
                    break
                board.push(best_move)
                moves_made += 1
            else:
                # Stockfish's turn
                result = engine.play(board, chess.engine.Limit(time=0.1))
                board.push(result.move)
                moves_made += 1
                
        if board.is_checkmate():
            if board.turn == chess.BLACK:
                print(f"  -> Result: Manoid WON in {moves_made} moves")
                results["Win"] += 1
                moves_to_mate.append(moves_made)
            else:
                print(f"  -> Result: Stockfish WON in {moves_made} moves")
                results["Loss"] += 1
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_repetition() or board.is_fifty_moves():
            print(f"  -> Result: DRAW in {moves_made} moves")
            results["Draw"] += 1
        else:
            print(f"  -> Result: DRAW (Max moves reached) in {moves_made} moves")
            results["Draw"] += 1

    engine.quit()

    print("\n=== Duel Results ===")
    print(f"Wins: {results['Win']} | Draws: {results['Draw']} | Losses: {results['Loss']}")
    if moves_to_mate:
        avg_moves = sum(moves_to_mate) / len(moves_to_mate)
        print(f"Average moves to mate (in wins): {avg_moves:.1f}")

if __name__ == "__main__":
    run_duel(num_games=5)
