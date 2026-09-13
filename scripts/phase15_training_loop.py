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

class Phase15Planner(BatchingMCTSPlanner):
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

def run_phase15(episodes=5):
    print("=== Phase 15: The Grandmaster Curriculum ===")
    
    env = FullChessEnv()
    value_model = ChessValueModelPhase14(input_dim=837)
    
    value_model.apply_lora_adapter("FullChess", rank=64)
    value_model.set_active_lora("FullChess")
    
    try:
        value_model.load_state_dict(torch.load("full_chess_lora_weights.pt"))
    except:
        pass
        
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    engine.configure({"Skill Level": 5})
    
    mse_history = []
    trainable_params = [p for p in value_model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=5e-4)

    for ep in range(1, episodes + 1):
        board = chess.Board()
        agent_color = chess.WHITE if ep % 2 == 1 else chess.BLACK
        desc = "White" if agent_color == chess.WHITE else "Black"
        
        planner = Phase15Planner(env, value_model, engine, batch_size=32)
        
        episodic_states = []
        episodic_targets = []
        ep_mse_sum = 0.0
        moves_made = 0
        
        while not board.is_game_over() and moves_made < 200: # limit moves to keep assay fast
            if board.turn == agent_color:
                best_move, target_q = planner.search_with_q(board.fen(), num_simulations=128)
                if best_move is None:
                    break
                    
                tensor_state = env.encode_state(board.fen())
                with torch.no_grad():
                    raw_pred = value_model(tensor_state).item()
                
                loss_val = (raw_pred - target_q) ** 2
                ep_mse_sum += loss_val
                
                episodic_states.append(tensor_state)
                episodic_targets.append(torch.tensor([[target_q]], dtype=torch.float32))
                
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
            else:
                final_state_desc = "Checkmate (Won)"
        elif board.is_stalemate() or board.is_repetition() or board.is_fifty_moves():
            final_state_desc = "Draw/Stalemate"
        else:
            final_state_desc = "Max moves"
            
        ep_mse = ep_mse_sum / max(1, len(episodic_states)) if episodic_states else 0.0
        mse_history.append(ep_mse)
        
        print(f"[Episode {ep}] Manoid plays {desc} | Result: {final_state_desc} ({moves_made} moves) | Avg MSE: {ep_mse:.4f}")
        
        skip_consolidation = False
        if len(mse_history) < 3:
            print("  -> Plasticity: Insufficient history. Defaulting to TRAIN.")
        else:
            prev2 = mse_history[-3]
            prev1 = mse_history[-2]
            curr = mse_history[-1]
            
            if curr > (prev1 * 10.0) and prev1 > 0.001:
                print("  -> Plasticity: Relative Magnitude Bypass detected! Triggering TRAIN.")
            elif curr > prev1 and prev1 > prev2:
                print("  -> Plasticity: Two-Strike regression detected. Triggering TRAIN.")
            else:
                print("  -> Plasticity: Error stable/recovering. SKIPPING consolidation.")
                skip_consolidation = True
                
        if not skip_consolidation and episodic_states:
            states_tensor = torch.cat(episodic_states, dim=0)
            targets_tensor = torch.cat(episodic_targets, dim=0)
            
            value_model.train()
            for _ in range(5):
                optimizer.zero_grad()
                preds = value_model(states_tensor)
                loss = nn.functional.mse_loss(preds, targets_tensor)
                loss.backward()
                optimizer.step()
            value_model.eval()

    torch.save(value_model.state_dict(), "full_chess_lora_weights_phase15.pt")
    engine.quit()

if __name__ == "__main__":
    run_phase15()
