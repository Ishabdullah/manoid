import sys
import os
import time
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import chess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.environment.micro_chess_env import MicroChessEnv
from scripts.chess_mcts_planner import ChessMCTSPlanner, ChessNode
from agent.consolidation.lora import LoRALinear

class ChessValueModel(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        self._lora_adapters = nn.ModuleDict()
        self._active_lora = None

    def apply_lora_adapter(self, concept_name, rank=4):
        if concept_name in self._lora_adapters:
            return
        import copy
        net_copy = copy.deepcopy(self.net)
        original_linear = net_copy[6]
        net_copy[6] = LoRALinear(original_linear, rank=rank)
        self._lora_adapters[concept_name] = net_copy

    def set_active_lora(self, concept_name):
        self._active_lora = concept_name

    def forward(self, x):
        if self._active_lora is not None:
            return self._lora_adapters[self._active_lora](x)
        return self.net(x)

class NeuralMCTSPlanner(ChessMCTSPlanner):
    def __init__(self, env, value_model):
        super().__init__(env)
        self.value_model = value_model
        
    def evaluate(self, node):
        board = chess.Board(node.fen)
        if board.is_checkmate():
            return 1.0 if board.turn == chess.BLACK else -1.0
        if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition() or board.is_fifty_moves():
            return -1.0
            
        tensor_state = self.env.encode_state(node.fen)
        with torch.no_grad():
            value = self.value_model(tensor_state).item()
        return value

    def search_with_q(self, root_fen, num_simulations=50):
        legal_moves = self.env.get_legal_actions(root_fen)
        if not legal_moves:
            return None, 0.0
            
        root = ChessNode(root_fen, untried_moves=list(legal_moves))
        
        for sim in range(num_simulations):
            node = root
            while len(node.untried_moves) == 0 and len(node.children) > 0:
                node = self.select(node)
            if len(node.untried_moves) > 0:
                node = self.expand(node)
            value = self.evaluate(node)
            self.backpropagate(node, value)
            
        if not root.children:
            return random.choice(legal_moves), 0.0
            
        best_n = max(child.N for child in root.children.values())
        best_actions = [m for m, child in root.children.items() if child.N == best_n]
        best_move = max(best_actions, key=lambda m: root.children[m].W / root.children[m].N if root.children[m].N > 0 else float('-inf'))
        target_q = root.children[best_move].W / root.children[best_move].N if root.children[best_move].N > 0 else 0.0
        
        return best_move, target_q

def run_neural_assay(episodes=60):
    print("=== Phase 10: Closed-Loop Neural Consolidation ===")
    print("Environment: Micro-Chess (K+R vs K)")
    print("Model: ChessValueModel with LoRA Adapter")
    print("--------------------------------------------------")
    
    env = MicroChessEnv()
    value_model = ChessValueModel()
    
    value_model.apply_lora_adapter("ChessEndgame", rank=8)
    value_model.set_active_lora("ChessEndgame")
    
    planner = NeuralMCTSPlanner(env, value_model)
    
    mse_history = []
    
    trainable_params = [p for p in value_model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=5e-4)
    
    base_fen = "k7/8/2K5/8/8/8/8/7R w - - 0 1"
    
    for ep in range(1, episodes + 1):
        fen = env.reset(base_fen)
        board = chess.Board(fen)
        
        episodic_states = []
        episodic_targets = []
        
        ep_mse_sum = 0.0
        moves_made = 0
        
        # Randomize start slightly to get diverse states
        if ep > 5:
           # scramble a bit
           board.push(random.choice(list(board.legal_moves)))
           if not board.is_game_over():
               board.push(random.choice(list(board.legal_moves)))

        while not board.is_game_over() and moves_made < 10:
            best_move, target_q = planner.search_with_q(board.fen(), num_simulations=400)
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
            moves_made += 1
            
            if board.is_game_over():
                break
                
            # Black's turn: play random move to speed up
            black_moves = list(board.legal_moves)
            if black_moves:
                board.push(random.choice(black_moves))
                moves_made += 1
            
        ep_mse = ep_mse_sum / max(1, moves_made // 2)
        mse_history.append(ep_mse)
        
        print(f"[Episode {ep}] Moves: {moves_made} | Avg MSE: {ep_mse:.4f}")
        
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
                print("  -> Plasticity: Twice-up regression detected. Triggering TRAIN.")
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

    torch.save(value_model.state_dict(), "chess_lora_weights.pt")
    print("Saved trained weights to chess_lora_weights.pt")

if __name__ == "__main__":
    run_neural_assay(episodes=30)
