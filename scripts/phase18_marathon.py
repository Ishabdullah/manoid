import sys
import os
import time
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.engine
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.environment.full_chess_env import FullChessEnv
from scripts.phase14_mcts_planner import ChessValueModelPhase14, BatchingMCTSPlanner, ChessNodePhase14

class Phase16Planner(BatchingMCTSPlanner):
    def search_with_q(self, root_fen, num_simulations=100):
        legal_moves = self.env.get_legal_actions(root_fen)
        if not legal_moves:
            return None, 0.0
            
        root = ChessNodePhase14(root_fen, untried_moves=list(legal_moves))
        
        sims_completed = 0
        while sims_completed < num_simulations:
            batch_nodes = []
            
            while len(batch_nodes) < self.batch_size and sims_completed + len(batch_nodes) < num_simulations:
                node = root
                
                # Selection
                while len(node.untried_moves) == 0 and len(node.children) > 0:
                    node = self.select(node)
                    
                # Expansion
                if len(node.untried_moves) > 0:
                    node = self.expand(node)
                
                batch_nodes.append(node)
                
            if not batch_nodes:
                break
                
            values = self.evaluate_batch(batch_nodes)
            
            for node, value in zip(batch_nodes, values):
                self.backpropagate(node, value)
                
            sims_completed += len(batch_nodes)
            
        if not root.children:
            return random.choice(legal_moves), 0.0
            
        best_n = max(child.N for child in root.children.values())
        best_actions = [m for m, child in root.children.items() if child.N == best_n]
        best_move = max(best_actions, key=lambda m: root.children[m].W / root.children[m].N if root.children[m].N > 0 else float('-inf'))
        target_q = root.children[best_move].W / root.children[best_move].N if root.children[best_move].N > 0 else 0.0
        return best_move, target_q

def run_phase18(episodes=2000):
    print("=== Phase 18: The Deep Grind ===")
    
    env = FullChessEnv()
    value_model = ChessValueModelPhase14(input_dim=837)
    
    value_model.apply_lora_adapter("FullChess", rank=64)
    value_model.set_active_lora("FullChess")
    
    try:
        value_model.load_state_dict(torch.load("manoid_grandmaster_ckpt_500.pt"), strict=False)
    except:
        pass
        
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    engine.configure({"Skill Level": 3})
    
    mse_history = []
    trainable_params = [p for p in value_model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=5e-4)

    wins = 0
    losses = 0
    draws = 0

    opening_book = [
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
        "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",
        "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",
        "rnbqkb1r/pppppp1p/5np1/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
        "rnbqk2r/ppppppbp/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq - 1 3",
        "rnbqkb1r/pppppppp/5n2/8/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 1 2",
        "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",
        "rnbqkbnr/pppppppp/8/8/8/2P5/PP1PPPPP/RNBQKBNR b KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/6P1/PPPPPP1P/RNBQKBNR b KQkq - 0 1",
        "rnbqkbnr/ppppp1pp/8/5p2/4P3/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 2",
        "rnbqkbnr/pppppppp/8/8/8/1P6/P1PPPPPP/RNBQKBNR b KQkq - 0 1"
    ]

    for ep in range(501, episodes + 1):
        if random.random() < 0.50:
            start_fen = chess.STARTING_FEN
            start_desc = "[Start: Natural]"
        else:
            start_fen = random.choice(opening_book)
            start_desc = f"[Start: GM Book: {start_fen}]"
            
        opponent_level = random.choice([2, 3, 4, 5])
        engine.configure({"Skill Level": opponent_level})
        print(f"Starting FEN: {start_desc} | Opponent Level: {opponent_level}")
        board = chess.Board(start_fen)
        agent_color = chess.WHITE if ep % 2 == 1 else chess.BLACK
        desc = "White" if agent_color == chess.WHITE else "Black"
        
        planner = Phase16Planner(env, value_model, engine, batch_size=32)
        
        episodic_states = []
        episodic_targets = []
        episodic_policy_targets = []
        ep_mse_sum = 0.0
        moves_made = 0
        
        while not board.is_game_over() and moves_made < 200:
            if board.turn == agent_color:
                best_move, target_q = planner.search_with_q(board.fen(), num_simulations=128)
                if best_move is None:
                    break
                    
                tensor_state = env.encode_state(board.fen())
                with torch.no_grad():
                    v_pred, p_logits = value_model(tensor_state)
                    raw_pred = v_pred.item()
                
                loss_val = (raw_pred - target_q) ** 2
                ep_mse_sum += loss_val
                
                info = engine.analyse(board, chess.engine.Limit(depth=1))
                sf_best_move = info["pv"][0] if "pv" in info and info["pv"] else best_move
                target_action_idx = sf_best_move.from_square * 64 + sf_best_move.to_square
                
                episodic_states.append(tensor_state)
                episodic_targets.append(torch.tensor([[target_q]], dtype=torch.float32))
                episodic_policy_targets.append(torch.tensor([target_action_idx], dtype=torch.long))
                
                board.push(best_move)
            else:
                result = engine.play(board, chess.engine.Limit(time=0.01))
                if result.move is None:
                    break
                board.push(result.move)
            moves_made += 1
            
        final_state_desc = ""
        if board.is_checkmate():
            if board.turn == agent_color:
                final_state_desc = "Checkmated (Lost)"
                losses += 1
            else:
                final_state_desc = "Checkmate (Won)"
                wins += 1
        elif board.is_stalemate() or board.is_repetition() or board.is_fifty_moves():
            final_state_desc = "Draw/Stalemate"
            draws += 1
        else:
            final_state_desc = "Max moves"
            draws += 1 # Or just count as incomplete, treat as draw for stats
            
        ep_mse = ep_mse_sum / max(1, len(episodic_states)) if episodic_states else 0.0
        mse_history.append(ep_mse)
        
        print(f"[Episode {ep}] Manoid plays {desc} | Result: {final_state_desc} ({moves_made} moves) | Avg MSE: {ep_mse:.4f}")
        
        skip_consolidation = False
        print("  -> Plasticity Override: Forcing TRAIN for Policy Bootstrapping.")
        
        # Override original plasticity gate for Phase 16.10
        # if len(mse_history) < 3:
        #     print("  -> Plasticity: Insufficient history. Defaulting to TRAIN.")
        # else:
        #     prev2 = mse_history[-3]
        #     prev1 = mse_history[-2]
        #     curr = mse_history[-1]
        #     
        #     if curr > (prev1 * 10.0) and prev1 > 0.001:
        #         print("  -> Plasticity: Relative Magnitude Bypass detected! Triggering TRAIN.")
        #     elif curr > prev1 and prev1 > prev2:
        #         print("  -> Plasticity: Two-Strike regression detected. Triggering TRAIN.")
        #     else:
        #         print("  -> Plasticity: Error stable/recovering. SKIPPING consolidation.")
        #         skip_consolidation = True
                
        if not skip_consolidation and episodic_states:
            states_tensor = torch.cat(episodic_states, dim=0)
            targets_tensor = torch.cat(episodic_targets, dim=0)
            policy_targets_tensor = torch.cat(episodic_policy_targets, dim=0)
            
            value_model.train()
            for _ in range(5):
                optimizer.zero_grad()
                v_preds, p_logits = value_model(states_tensor)
                v_loss = nn.functional.mse_loss(v_preds, targets_tensor)
                p_loss = nn.functional.cross_entropy(p_logits, policy_targets_tensor)
                loss = v_loss + p_loss
                loss.backward()
                optimizer.step()
            value_model.eval()
            print(f"  -> Consolidation Complete | Final Value Loss: {v_loss.item():.4f} | Final Policy Loss: {p_loss.item():.4f}")

        if ep % 25 == 0:
            ckpt_name = f"manoid_grandmaster_ckpt_{ep}.pt"
            torch.save(value_model.state_dict(), ckpt_name)
            avg_mse_25 = sum(mse_history[-25:]) / 25
            print(f"\n--- Checkpoint Saved: {ckpt_name} ---")
            print(f"Stats over last 25 episodes: Wins: {wins}, Losses: {losses}, Draws: {draws}")
            print(f"Average MSE over last 25 episodes: {avg_mse_25:.4f}")
            print("----------------------------------------\n")
            # reset stats for next 25
            wins = 0
            losses = 0
            draws = 0

    torch.save(value_model.state_dict(), "full_chess_lora_weights_final.pt")
    engine.quit()

if __name__ == "__main__":
    run_phase18()
