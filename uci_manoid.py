import sys
import os
import torch
import chess
import chess.engine
import warnings

# Suppress PyTorch warnings for clean UCI communication
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from agent.environment.full_chess_env import FullChessEnv
from scripts.phase14_mcts_planner import ChessValueModelPhase14, BatchingMCTSPlanner

CHECKPOINT_PATH = "manoid_grandmaster_ckpt_500.pt"

def main():
    board = chess.Board()
    env = FullChessEnv()
    
    # Initialize the Dual-Head Model
    value_model = ChessValueModelPhase14(input_dim=837)
    value_model.apply_lora_adapter("FullChess", rank=64)
    value_model.set_active_lora("FullChess")
    
    try:
        # Load with strict=False in case policy weights are absent from older saves
        value_model.load_state_dict(torch.load(CHECKPOINT_PATH), strict=False)
    except Exception as e:
        pass
        
    value_model.eval()

    # Initialize Proxy Engine for MCTS rollout evaluations
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    engine.configure({"Skill Level": 5})

    planner = BatchingMCTSPlanner(env, value_model, engine, batch_size=32)

    while True:
        try:
            line = sys.stdin.readline()
        except EOFError:
            break
            
        if not line:
            break
            
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        command = parts[0]
        
        if command == "uci":
            print("id name Manoid")
            print("id author The Architect")
            print("uciok")
            sys.stdout.flush()
            
        elif command == "isready":
            print("readyok")
            sys.stdout.flush()
            
        elif command == "ucinewgame":
            board = chess.Board()
            
        elif command == "position":
            if len(parts) > 1 and parts[1] == "startpos":
                board = chess.Board()
                moves_idx = -1
                if "moves" in parts:
                    moves_idx = parts.index("moves")
                    for m in parts[moves_idx+1:]:
                        board.push(chess.Move.from_uci(m))
            elif len(parts) > 1 and parts[1] == "fen":
                try:
                    moves_idx = parts.index("moves")
                    fen_str = " ".join(parts[2:moves_idx])
                    board = chess.Board(fen_str)
                    for m in parts[moves_idx+1:]:
                        board.push(chess.Move.from_uci(m))
                except ValueError:
                    fen_str = " ".join(parts[2:])
                    board = chess.Board(fen_str)
                    
        elif command == "go":
            best_move = planner.search(board.fen(), num_simulations=400)
            if best_move is None:
                legal = list(board.legal_moves)
                if legal:
                    best_move = legal[0].uci()
                else:
                    best_move = "0000"
            else:
                if type(best_move) is chess.Move:
                    best_move = best_move.uci()
                elif type(best_move) is str:
                    pass
                else:
                    best_move = chess.Move.from_uci(str(best_move)).uci()
                    
            print(f"bestmove {best_move}")
            sys.stdout.flush()
            
        elif command == "quit":
            engine.quit()
            break

if __name__ == "__main__":
    main()
