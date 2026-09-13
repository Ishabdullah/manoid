import sys
import os
import torch
import chess
import chess.engine
import warnings
import glob
import re
import shutil

# Suppress PyTorch warnings for clean UCI communication
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from agent.environment.full_chess_env import FullChessEnv
from scripts.phase14_mcts_planner import ChessValueModelPhase14, BatchingMCTSPlanner

def get_latest_checkpoint():
    checkpoints = glob.glob("manoid_v2_ckpt_*.pt")
    if not checkpoints:
        return None
    
    def extract_number(filename):
        match = re.search(r'_(\d+)\.pt$', filename)
        return int(match.group(1)) if match else -1
        
    return max(checkpoints, key=extract_number)

def main():
    board = chess.Board()
    env = FullChessEnv()
    
    # Initialize the Dual-Head Model
    value_model = ChessValueModelPhase14(input_dim=837)
    value_model.apply_lora_adapter("FullChess", rank=64)
    value_model.set_active_lora("FullChess")
    
    ckpt_path = get_latest_checkpoint()
    if ckpt_path:
        try:
            value_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
            print(f"info string Loaded checkpoint: {ckpt_path}", file=sys.stderr)
        except Exception as e:
            print(f"info string FAILED to load checkpoint {ckpt_path}: {e}", file=sys.stderr)
            print("info string WARNING: Falling back to randomly initialized weights!", file=sys.stderr)
    else:
        print("info string WARNING: No checkpoint found, using randomly initialized weights!", file=sys.stderr)
        
    value_model.eval()

    if shutil.which("stockfish") is None:
        print("Error: 'stockfish' binary not found in PATH.", file=sys.stderr)
        print("Please install it using 'pkg install stockfish' (Termux) or 'apt install stockfish' (Linux).", file=sys.stderr)
        sys.exit(1)

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
            import time
            start_time = time.time()
            
            num_simulations = 400
            wtime = btime = winc = binc = movetime = nodes = None
            
            i = 1
            while i < len(parts):
                if parts[i] == "wtime" and i + 1 < len(parts): wtime = int(parts[i+1]); i+=2
                elif parts[i] == "btime" and i + 1 < len(parts): btime = int(parts[i+1]); i+=2
                elif parts[i] == "winc" and i + 1 < len(parts): winc = int(parts[i+1]); i+=2
                elif parts[i] == "binc" and i + 1 < len(parts): binc = int(parts[i+1]); i+=2
                elif parts[i] == "movetime" and i + 1 < len(parts): movetime = int(parts[i+1]); i+=2
                elif parts[i] == "nodes" and i + 1 < len(parts): nodes = int(parts[i+1]); i+=2
                else: i+=1
                
            if nodes is not None:
                num_simulations = nodes
            elif movetime is not None:
                num_simulations = max(10, int((movetime / 1000.0) * 100))
            elif wtime is not None and btime is not None:
                time_left = wtime if board.turn == chess.WHITE else btime
                inc = winc if board.turn == chess.WHITE and winc else (binc if board.turn == chess.BLACK and binc else 0)
                allocated_time = (time_left / 40.0) + (inc * 0.5)
                num_simulations = max(10, int((allocated_time / 1000.0) * 100))

            best_move = planner.search(board.fen(), num_simulations=num_simulations)
            
            end_time = time.time()
            print(f"info string Search took {end_time - start_time:.2f}s for {num_simulations} sims")
            sys.stdout.flush()
            
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
            
        elif command == "stop":
            # Since our planner.search is synchronous and blocks the read loop,
            # 'stop' will likely only be read after 'go' finishes unless implemented with threading.
            # However, for UCI compliance, we parse it and if reached without searching, just return a move.
            legal = list(board.legal_moves)
            if legal:
                best_move = legal[0].uci()
            else:
                best_move = "0000"
            print(f"bestmove {best_move}")
            sys.stdout.flush()

        elif command == "quit":
            engine.quit()
            break

if __name__ == "__main__":
    main()
