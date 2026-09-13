import sys
import os
import time
import math
import random
import torch
import torch.nn as nn
import chess
import chess.engine
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.environment.full_chess_env import FullChessEnv
from agent.consolidation.lora import LoRALinear

class ChessValueModelPhase14(nn.Module):
    def __init__(self, input_dim=837):
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

    def apply_lora_adapter(self, concept_name, rank=64):
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

class ChessNodePhase14:
    def __init__(self, fen, parent=None, move=None, untried_moves=None):
        self.fen = fen
        self.parent = parent
        self.move = move
        self.N = 0
        self.W = 0.0
        self.children = {}
        self.untried_moves = untried_moves if untried_moves is not None else []
        self.state_value = None  # To cache evaluation

class BatchingMCTSPlanner:
    def __init__(self, env, value_model, engine, batch_size=64):
        self.env = env
        self.value_model = value_model
        self.engine = engine
        self.exploration_weight = math.sqrt(2)
        self.batch_size = batch_size

    def search(self, root_fen, num_simulations=2000):
        legal_moves = self.env.get_legal_actions(root_fen)
        if not legal_moves:
            return None
            
        root = ChessNodePhase14(root_fen, untried_moves=list(legal_moves))
        
        sims_completed = 0
        while sims_completed < num_simulations:
            # Collect a batch of leaf nodes
            batch_nodes = []
            
            while len(batch_nodes) < self.batch_size and sims_completed + len(batch_nodes) < num_simulations:
                node = root
                
                # 1. Selection
                while len(node.untried_moves) == 0 and len(node.children) > 0:
                    node = self.select(node)
                    
                # 2. Expansion
                if len(node.untried_moves) > 0:
                    node = self.expand(node)
                
                batch_nodes.append(node)
                
            if not batch_nodes:
                break
                
            # 3. Batch Evaluation
            values = self.evaluate_batch(batch_nodes)
            
            # 4. Backpropagation
            for node, value in zip(batch_nodes, values):
                self.backpropagate(node, value)
                
            sims_completed += len(batch_nodes)
            
        if not root.children:
            return random.choice(legal_moves)
            
        best_n = max(child.N for child in root.children.values())
        best_actions = [m for m, child in root.children.items() if child.N == best_n]
        best_move = max(best_actions, key=lambda m: root.children[m].W / root.children[m].N if root.children[m].N > 0 else float('-inf'))
        return best_move

    def select(self, node):
        board = chess.Board(node.fen)
        is_white_turn = (board.turn == chess.WHITE)
        
        best_score = float('-inf')
        best_child = None
        for move, child in node.children.items():
            if child.N == 0:
                return child
                
            q_val = child.W / child.N
            if not is_white_turn:
                q_val = -q_val
                
            ucb_score = q_val + self.exploration_weight * math.sqrt(math.log(node.N) / child.N)
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        return best_child

    def expand(self, node):
        move = node.untried_moves.pop()
        next_fen = self.env.apply_action(node.fen, move)
        next_untried = self.env.get_legal_actions(next_fen)
        child_node = ChessNodePhase14(next_fen, parent=node, move=move, untried_moves=list(next_untried))
        node.children[move] = child_node
        return child_node

    def evaluate_batch(self, nodes: List[ChessNodePhase14]):
        values = []
        fens_to_eval = []
        node_indices = []
        
        for i, node in enumerate(nodes):
            board = chess.Board(node.fen)
            if board.is_checkmate():
                val = 1.0 if board.turn == chess.BLACK else -1.0
                values.append(val)
            elif board.is_stalemate() or board.is_insufficient_material() or board.is_repetition() or board.is_fifty_moves():
                values.append(0.0) # Draw
            else:
                values.append(None)
                fens_to_eval.append(node.fen)
                node_indices.append(i)
                
        if not fens_to_eval:
            return values
            
        # Parallel chunk evaluation with Stockfish + LoRA Model
        tensor_states = torch.cat([self.env.encode_state(fen) for fen in fens_to_eval], dim=0)
        
        with torch.no_grad():
            nn_preds = self.value_model(tensor_states).squeeze(-1).tolist()
            if isinstance(nn_preds, float):
                nn_preds = [nn_preds]
                
        # Stockfish Centipawn Bootstrapping
        sf_vals = []
        for fen in fens_to_eval:
            board = chess.Board(fen)
            info = self.engine.analyse(board, chess.engine.Limit(depth=4)) # Quick depth for fast batch eval
            score = info["score"].white()
            if score.is_mate():
                cp = 10000 if score.mate() > 0 else -10000
            else:
                cp = score.score()
            
            # Normalize centipawn to [-1.0, 1.0] proxy reward
            # 1000 cp = 1.0
            norm_score = max(-1.0, min(1.0, cp / 1000.0))
            sf_vals.append(norm_score)
            
        # Combine Neural Net and Stockfish (e.g. average, or just use SF for bootstrapping)
        # We'll use a mix to show we're bootstrapping
        for idx, nn_val, sf_val in zip(node_indices, nn_preds, sf_vals):
            combined_val = 0.5 * nn_val + 0.5 * sf_val
            values[idx] = combined_val
            
        return values

    def backpropagate(self, node, value):
        current_node = node
        while current_node is not None:
            current_node.N += 1
            current_node.W += value 
            current_node = current_node.parent

def run_smoke_test():
    print("=== Phase 14: Full-Scale Chess Upgrade & Centipawn Bootstrapping ===")
    
    env = FullChessEnv()
    
    # 837 features
    value_model = ChessValueModelPhase14(input_dim=837)
    value_model.apply_lora_adapter("FullChess", rank=64)
    value_model.set_active_lora("FullChess")
    
    # Init Stockfish
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    
    planner = BatchingMCTSPlanner(env, value_model, engine, batch_size=32)
    
    start_fen = chess.STARTING_FEN
    
    print("Starting deep search with 2000 simulations...")
    start_time = time.time()
    
    best_move = planner.search(start_fen, num_simulations=2000)
    
    elapsed = time.time() - start_time
    
    print(f"Chosen Opening Move: {best_move}")
    print(f"Time taken for 2000 simulations: {elapsed:.2f} seconds")
    print("Batching queue successfully processed leaf nodes.")
    
    engine.quit()

if __name__ == "__main__":
    run_smoke_test()
