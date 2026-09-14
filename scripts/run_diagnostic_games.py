import sys
import os
import time
import math
import torch
import chess
import chess.engine
import chess.pgn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.environment.full_chess_env import FullChessEnv
from scripts.phase14_mcts_planner import ChessValueModelPhase14
from scripts.phase18_marathon import Phase16Planner

def run_diagnostics():
    env = FullChessEnv()
    value_model = ChessValueModelPhase14(input_dim=837)
    value_model.apply_lora_adapter("FullChess", rank=64)
    value_model.set_active_lora("FullChess")
    
    ckpt_path = "manoid_v2_ckpt_1700.pt"
    value_model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True), strict=False)
    value_model.eval()
    print(f"Loaded weights from {ckpt_path}")
    
    # Engine for play (Stockfish Level 1)
    play_engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    play_engine.configure({"Skill Level": 1})
    
    # Engine for depth 10 analysis
    analysis_engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    
    planner = Phase16Planner(env, value_model, play_engine, batch_size=32)
    planner.USE_GUMBEL_SEARCH = True
    planner.GUMBEL_LOGGING_ENABLED = False
    planner.USE_QUIESCENCE = True
    planner.GUMBEL_LOGGING_ENABLED = True # enable for printing
    
    game_configs = [
        {"desc": "Game 1 (Natural Start, Manoid=White)", "fen": chess.STARTING_FEN, "color": chess.WHITE},
        {"desc": "Game 2 (Natural Start, Manoid=Black)", "fen": chess.STARTING_FEN, "color": chess.BLACK},
        {"desc": "Game 3 (GM Book 1.e4 e5, Manoid=White)", "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", "color": chess.WHITE},
        {"desc": "Game 4 (GM Book 1.e4 c5, Manoid=Black)", "fen": "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2", "color": chess.BLACK},
        {"desc": "Game 5 (Natural Start, Manoid=White)", "fen": chess.STARTING_FEN, "color": chess.WHITE},
    ]
    
    for g_idx, cfg in enumerate(game_configs, 1):
        print("\n" + "="*80)
        print(f"=== {cfg['desc']} ===")
        print(f"Starting FEN: {cfg['fen']}")
        print("="*80)
        
        board = chess.Board(cfg['fen'])
        agent_color = cfg['color']
        
        pgn_game = chess.pgn.Game()
        pgn_game.headers["Event"] = f"Manoid Diagnostic {g_idx}"
        pgn_game.headers["White"] = "Manoid (v2_ckpt_1700)" if agent_color == chess.WHITE else "Stockfish Skill 1"
        pgn_game.headers["Black"] = "Stockfish Skill 1" if agent_color == chess.WHITE else "Manoid (v2_ckpt_1700)"
        if cfg['fen'] != chess.STARTING_FEN:
            pgn_game.headers["SetUp"] = "1"
            pgn_game.headers["FEN"] = cfg['fen']
            
        pgn_node = pgn_game
        
        manoid_turn_count = 0
        moves_history = []
        
        while not board.is_game_over() and len(moves_history) < 150:
            is_manoid_turn = (board.turn == agent_color)
            fen_before = board.fen()
            
            if is_manoid_turn:
                manoid_turn_count += 1
                best_move, target_q = planner.search_with_q(fen_before, num_simulations=128)
                
                # Check legality
                is_legal = best_move in board.legal_moves
                assert is_legal, f"ILLEGAL MOVE DETECTED: {best_move} in {fen_before}"
                
                # If within first 10 Manoid turns, print detailed info
                if manoid_turn_count <= 10:
                    print(f"\n--- [Manoid Turn {manoid_turn_count}] (Ply {len(moves_history)+1}) ---")
                    print(f"FEN: {fen_before}")
                    san_move = board.san(best_move)
                    print(f"Manoid Selected: {best_move.uci()} ({san_move}) [Legal: {is_legal}, Q: {target_q:.4f}]")
                    
                    # For all games, let us analyze top 3 moves with Stockfish depth 10
                    sf_info = analysis_engine.analyse(board, chess.engine.Limit(depth=10), multipv=3)
                    print(f"Stockfish (Depth 10) Top Moves:")
                    manoid_eval_found = None
                    for rank, entry in enumerate(sf_info, 1):
                        sf_m = entry["pv"][0]
                        sf_san = board.san(sf_m)
                        score = entry["score"].white() if agent_color == chess.WHITE else entry["score"].black()
                        cp_str = f"{score.score():+d} cp" if not score.is_mate() else f"Mate {score.mate()}"
                        print(f"  #{rank}: {sf_m.uci()} ({sf_san}) -> {cp_str}")
                        if sf_m == best_move:
                            manoid_eval_found = cp_str
                    
                    # If Manoid's move was not in top 3, get its eval specifically
                    if manoid_eval_found is None:
                        board.push(best_move)
                        eval_after = analysis_engine.analyse(board, chess.engine.Limit(depth=10))
                        board.pop()
                        # After pushing, score is from opponent perspective, so invert
                        sc_after = eval_after["score"].white() if agent_color == chess.WHITE else eval_after["score"].black()
                        cp_str_after = f"{sc_after.score():+d} cp" if not sc_after.is_mate() else f"Mate {sc_after.mate()}"
                        print(f"  -> Manoid move actual eval: {cp_str_after}")
                        
                move_to_play = best_move
            else:
                res = play_engine.play(board, chess.engine.Limit(time=0.01))
                move_to_play = res.move
                
            san = board.san(move_to_play)
            moves_history.append((board.turn, move_to_play, san))
            board.push(move_to_play)
            pgn_node = pgn_node.add_variation(move_to_play)
            
        result = board.result()
        pgn_game.headers["Result"] = result
        
        print("\n" + "-"*60)
        print(f"Game {g_idx} Finished: Result = {result} in {len(moves_history)} moves ({len(moves_history)//2} full moves)")
        if board.is_checkmate():
            winner = "Manoid" if ((board.turn != agent_color)) else "Stockfish Skill 1"
            print(f"Outcome: Checkmate! Winner: {winner}")
        elif board.is_stalemate():
            print("Outcome: Stalemate")
        elif board.is_insufficient_material():
            print("Outcome: Draw (Insufficient Material)")
        elif board.is_repetition():
            print("Outcome: Draw (Repetition)")
        elif board.is_fifty_moves():
            print("Outcome: Draw (50-move rule)")
        else:
            print("Outcome: Max moves reached")
            
        print("\n--- FULL MOVE SEQUENCE (SAN) ---")
        full_moves_san = []
        temp_b = chess.Board(cfg['fen'])
        move_num = temp_b.fullmove_number
        if temp_b.turn == chess.BLACK:
            full_moves_san.append(f"{move_num}... {moves_history[0][2]}")
            temp_b.push(moves_history[0][1])
            start_i = 1
            move_num += 1
        else:
            start_i = 0
            
        for i in range(start_i, len(moves_history), 2):
            w_san = moves_history[i][2]
            if i + 1 < len(moves_history):
                b_san = moves_history[i+1][2]
                full_moves_san.append(f"{move_num}. {w_san} {b_san}")
            else:
                full_moves_san.append(f"{move_num}. {w_san}")
            move_num += 1
            
        print(" ".join(full_moves_san))
        
        print("\n--- FULL PGN ---")
        print(str(pgn_game))
        print("-"*60)

    play_engine.quit()
    analysis_engine.quit()

if __name__ == "__main__":
    run_diagnostics()
