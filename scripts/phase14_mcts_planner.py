import math
import random
import torch
import torch.nn as nn
import chess
import chess.engine
from typing import List

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
            nn.GELU()
        )
        self.value_head = nn.Linear(128, 1)
        self.policy_head = nn.Linear(128, 4096)
        self._lora_adapters = nn.ModuleDict()
        self._active_lora = None

    def apply_lora_adapter(self, concept_name, rank=64):
        if concept_name in self._lora_adapters:
            return
        import copy
        net_copy = nn.ModuleDict({
            'net': copy.deepcopy(self.net),
            'value_head': LoRALinear(copy.deepcopy(self.value_head), rank=rank),
            'policy_head': LoRALinear(copy.deepcopy(self.policy_head), rank=rank)
        })
        self._lora_adapters[concept_name] = net_copy

    def set_active_lora(self, concept_name):
        self._active_lora = concept_name

    def forward(self, x):
        if self._active_lora is not None:
            features = self._lora_adapters[self._active_lora]['net'](x)
            v = self._lora_adapters[self._active_lora]['value_head'](features)
            p = self._lora_adapters[self._active_lora]['policy_head'](features)
            return v, p
        features = self.net(x)
        return self.value_head(features), self.policy_head(features)


class ChessNodePhase14:
    def __init__(self, fen, parent=None, move=None, untried_moves=None, is_terminal_win=False, is_forcing=False):
        self.fen = fen
        self.parent = parent
        self.move = move
        self.N = 0
        self.W = 0.0
        self.children = {}
        self.untried_moves = untried_moves if untried_moves is not None else []
        self.move_probs = {}
        self.is_terminal_win = is_terminal_win
        self.is_forcing = is_forcing


class BatchingMCTSPlanner:
    def __init__(self, env, value_model, engine, batch_size=32):
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
                
            prior = node.move_probs.get(move, 1.0)
            ucb_score = q_val + self.exploration_weight * prior * math.sqrt(node.N) / (1 + child.N)
            
            if child.is_terminal_win:
                ucb_score += 1000.0
                
            if child.is_forcing:
                ucb_score += 0.2
                
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        return best_child

    def expand(self, node):
        move = node.untried_moves.pop()
        board = chess.Board(node.fen)
        is_forcing = board.is_capture(move) or board.gives_check(move) or (move.promotion is not None)
        
        next_fen = self.env.apply_action(node.fen, move)
        next_board = chess.Board(next_fen)
        is_terminal_win = next_board.is_checkmate()
        
        next_untried = self.env.get_legal_actions(next_fen)
        child_node = ChessNodePhase14(next_fen, parent=node, move=move, untried_moves=list(next_untried),
                                      is_terminal_win=is_terminal_win, is_forcing=is_forcing)
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
                values.append(0.0)
            else:
                values.append(None)
                fens_to_eval.append(node.fen)
                node_indices.append(i)
                
        if not fens_to_eval:
            return values
            
        tensor_states = torch.cat([self.env.encode_state(fen) for fen in fens_to_eval], dim=0)
        
        with torch.no_grad():
            v_preds, p_logits = self.value_model(tensor_states)
            v_preds = v_preds.squeeze(-1).tolist()
            if isinstance(v_preds, float):
                v_preds = [v_preds]
            p_probs = torch.softmax(p_logits, dim=-1).cpu().numpy()
                
        sf_vals = []
        for i, fen in enumerate(fens_to_eval):
            board = chess.Board(fen)
            info = self.engine.analyse(board, chess.engine.Limit(depth=1))
            score = info["score"].white()
            if score.is_mate():
                cp = 10000 if score.mate() > 0 else -10000
            else:
                cp = score.score()
            
            norm_score = max(-1.0, min(1.0, cp / 1000.0))
            sf_vals.append(norm_score)
            
            # Map policy probabilities to node
            node = nodes[node_indices[i]]
            legal_moves = list(board.legal_moves)
            for move in legal_moves:
                action_idx = move.from_square * 64 + move.to_square
                node.move_probs[move] = p_probs[i][action_idx]
            
            # Normalize probabilities for valid moves only
            total_prob = sum(node.move_probs.values())
            if total_prob > 0:
                for move in node.move_probs:
                    node.move_probs[move] /= total_prob
            
        for idx, nn_val, sf_val in zip(node_indices, v_preds, sf_vals):
            combined_val = 0.5 * nn_val + 0.5 * sf_val
            values[idx] = combined_val
            
        return values

    def backpropagate(self, node, value):
        current_node = node
        while current_node is not None:
            current_node.N += 1
            current_node.W += value 
            current_node = current_node.parent
