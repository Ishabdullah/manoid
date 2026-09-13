import argparse
import sys
import os
import time
import random
import torch
import chess
import chess.engine

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.environment.full_chess_env import FullChessEnv
from scripts.phase14_mcts_planner import ChessValueModelPhase14, BatchingMCTSPlanner, ChessNodePhase14

class InteractivePlanner(BatchingMCTSPlanner):
    def search_with_info(self, root_fen, num_simulations=1000):
        legal_moves = self.env.get_legal_actions(root_fen)
        if not legal_moves:
            return None, 0.0, 0.0
            
        root = ChessNodePhase14(root_fen, untried_moves=list(legal_moves))
        
        sims_completed = 0
        start_time = time.time()
        
        while sims_completed < num_simulations:
            batch_nodes = []
            while len(batch_nodes) < self.batch_size and sims_completed + len(batch_nodes) < num_simulations:
                node = root
                while len(node.untried_moves) == 0 and len(node.children) > 0:
                    node = self.select(node)
                if len(node.untried_moves) > 0:
                    node = self.expand(node)
                batch_nodes.append(node)
                
            if not batch_nodes:
                break
                
            values = self.evaluate_batch(batch_nodes)
            for node, value in zip(batch_nodes, values):
                self.backpropagate(node, value)
                
            sims_completed += len(batch_nodes)
            
        elapsed = time.time() - start_time
            
        if not root.children:
            return random.choice(legal_moves), 0.0, elapsed
            
        best_n = max(child.N for child in root.children.values())
        best_actions = [m for m, child in root.children.items() if child.N == best_n]
        best_move = max(best_actions, key=lambda m: root.children[m].W / root.children[m].N if root.children[m].N > 0 else float('-inf'))
        target_q = root.children[best_move].W / root.children[best_move].N if root.children[best_move].N > 0 else 0.0
        
        return best_move, target_q, elapsed

def main():
    parser = argparse.ArgumentParser(description="Play against Manoid in the terminal.")
    parser.add_argument("--checkpoint", type=str, default="manoid_grandmaster_ckpt_50.pt", help="Path to checkpoint file")
    parser.add_argument("--color", type=str, default="white", choices=["white", "black"], help="Your color (white or black)")
    args = parser.parse_args()
    
    print(f"Loading Manoid Brain: {args.checkpoint}...")
    
    env = FullChessEnv()
    value_model = ChessValueModelPhase14(input_dim=837)
    value_model.apply_lora_adapter("FullChess", rank=64)
    value_model.set_active_lora("FullChess")
    
    if os.path.exists(args.checkpoint):
        value_model.load_state_dict(torch.load(args.checkpoint))
        print("Checkpoint loaded successfully!")
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Using untrained model.")
        
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    planner = InteractivePlanner(env, value_model, engine, batch_size=32)
    
    board = chess.Board()
    human_color = chess.WHITE if args.color.lower() == "white" else chess.BLACK
    
    print("\n--- Match Started! ---")
    print("Type your move in SAN (e.g., e4, Nf3) or UCI (e.g., e2e4). Type 'quit' to exit.")
    
    while not board.is_game_over():
        print("\n" + board.unicode(borders=True) + "\n")
        
        if board.turn == human_color:
            try:
                move_str = input("Your move: ").strip()
            except EOFError:
                break
                
            if move_str.lower() in ["quit", "exit"]:
                print("Game aborted by user.")
                break
                
            try:
                move = board.parse_san(move_str)
                board.push(move)
            except ValueError:
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        board.push(move)
                    else:
                        print("Illegal UCI move. Try again.")
                except ValueError:
                    print("Invalid move format. Try SAN (e4) or UCI (e2e4).")
        else:
            print("Manoid is thinking...")
            best_move, q_val, elapsed = planner.search_with_info(board.fen(), num_simulations=1000)
            if best_move is None:
                break
                
            print(f"[Telemetry] Time: {elapsed:.2f}s | Simulations: 1000 | Root Q-Value: {q_val:.4f}")
            print(f"Manoid plays: {board.san(best_move)} ({best_move.uci()})")
            board.push(best_move)
            
    if board.is_game_over():
        print("\n" + board.unicode(borders=True) + "\n")
        print("Game Over:", board.result())
        
    engine.quit()

if __name__ == "__main__":
    main()
