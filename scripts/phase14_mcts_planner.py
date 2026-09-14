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
        
        self.USE_PHASE_EXPERTS = True
        self.expert_opening = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        self.expert_middlegame = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        self.expert_endgame = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        
        self.value_head = nn.Linear(128, 1)
        self.from_head = nn.Linear(128, 64)
        self.to_head = nn.Linear(128, 64)
        self.depth_delta_head = nn.Linear(128, 1)
        self.dense_reward_head = nn.Linear(128, 1)
        self.pv_from_head = nn.Linear(128, 3 * 64)
        self.pv_to_head = nn.Linear(128, 3 * 64)
        self._lora_adapters = nn.ModuleDict()
        self._active_lora = None

    def apply_lora_adapter(self, concept_name, rank=64):
        if concept_name in self._lora_adapters:
            return
        import copy
        net_copy = nn.ModuleDict({
            'net': copy.deepcopy(self.net),
            'expert_opening': copy.deepcopy(self.expert_opening),
            'expert_middlegame': copy.deepcopy(self.expert_middlegame),
            'expert_endgame': copy.deepcopy(self.expert_endgame),
            'value_head': LoRALinear(copy.deepcopy(self.value_head), rank=rank),
            'from_head': LoRALinear(copy.deepcopy(self.from_head), rank=rank),
            'to_head': LoRALinear(copy.deepcopy(self.to_head), rank=rank),
            'depth_delta_head': LoRALinear(copy.deepcopy(self.depth_delta_head), rank=rank),
            'dense_reward_head': LoRALinear(copy.deepcopy(self.dense_reward_head), rank=rank),
            'pv_from_head': LoRALinear(copy.deepcopy(self.pv_from_head), rank=rank),
            'pv_to_head': LoRALinear(copy.deepcopy(self.pv_to_head), rank=rank)
        })
        self._lora_adapters[concept_name] = net_copy

    def set_active_lora(self, concept_name):
        self._active_lora = concept_name

    def forward(self, x, phases=None):
        if self._active_lora is not None:
            features = self._lora_adapters[self._active_lora]['net'](x)
            if getattr(self, 'USE_PHASE_EXPERTS', False) and phases is not None:
                out_features = torch.zeros_like(features)
                mask_op = (phases == 0)
                if mask_op.any(): out_features[mask_op] = self._lora_adapters[self._active_lora]['expert_opening'](features[mask_op])
                mask_mid = (phases == 1)
                if mask_mid.any(): out_features[mask_mid] = self._lora_adapters[self._active_lora]['expert_middlegame'](features[mask_mid])
                mask_end = (phases == 2)
                if mask_end.any(): out_features[mask_end] = self._lora_adapters[self._active_lora]['expert_endgame'](features[mask_end])
                features = out_features
                
            v = self._lora_adapters[self._active_lora]['value_head'](features)
            p_from = self._lora_adapters[self._active_lora]['from_head'](features)
            p_to = self._lora_adapters[self._active_lora]['to_head'](features)
            d = self._lora_adapters[self._active_lora]['depth_delta_head'](features)
            dense = self._lora_adapters[self._active_lora]['dense_reward_head'](features)
            pv_from = self._lora_adapters[self._active_lora]['pv_from_head'](features).view(-1, 3, 64)
            pv_to = self._lora_adapters[self._active_lora]['pv_to_head'](features).view(-1, 3, 64)
        else:
            features = self.net(x)
            if getattr(self, 'USE_PHASE_EXPERTS', False) and phases is not None:
                out_features = torch.zeros_like(features)
                mask_op = (phases == 0)
                if mask_op.any(): out_features[mask_op] = self.expert_opening(features[mask_op])
                mask_mid = (phases == 1)
                if mask_mid.any(): out_features[mask_mid] = self.expert_middlegame(features[mask_mid])
                mask_end = (phases == 2)
                if mask_end.any(): out_features[mask_end] = self.expert_endgame(features[mask_end])
                features = out_features
                
            v = self.value_head(features)
            p_from = self.from_head(features)
            p_to = self.to_head(features)
            d = self.depth_delta_head(features)
            dense = self.dense_reward_head(features)
            pv_from = self.pv_from_head(features).view(-1, 3, 64)
            pv_to = self.pv_to_head(features).view(-1, 3, 64)
            
        return v, p_from, p_to, d, dense, pv_from, pv_to


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

    USE_GUMBEL_SEARCH = True

    GUMBEL_LOGGING_ENABLED = True
    GUMBEL_MATCHES = 0
    GUMBEL_CALLS = 0

    def search(self, root_fen, num_simulations=2000):
        puct_move = None
        if self.USE_GUMBEL_SEARCH and getattr(self, 'GUMBEL_LOGGING_ENABLED', False) and getattr(self, 'GUMBEL_CALLS', 0) < 5000: # approx 50 episodes
            puct_move = self._puct_search(root_fen, num_simulations)
            
        if self.USE_GUMBEL_SEARCH:
            move, _ = self._gumbel_search(root_fen, num_simulations)
            if puct_move is not None:
                self.GUMBEL_CALLS = getattr(self, 'GUMBEL_CALLS', 0) + 1
                if move == puct_move:
                    self.GUMBEL_MATCHES = getattr(self, 'GUMBEL_MATCHES', 0) + 1
                if self.GUMBEL_CALLS % 10 == 0:
                    print(f"[Gumbel Instrumentation] Match Rate (Gumbel vs PUCT): {self.GUMBEL_MATCHES}/{self.GUMBEL_CALLS} ({(self.GUMBEL_MATCHES/self.GUMBEL_CALLS)*100:.1f}%)")
            return move
            
        return self._puct_search(root_fen, num_simulations)

    def _puct_search(self, root_fen, num_simulations=2000):
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

    def _gumbel_search(self, root_fen, num_simulations):
        legal_moves = self.env.get_legal_actions(root_fen)
        if not legal_moves:
            return None, 0.0
            
        root = ChessNodePhase14(root_fen, untried_moves=list(legal_moves))
        
        # 1. Evaluate root to get policy priors
        _ = self.evaluate_batch([root])
        
        # 2. Add Gumbel noise to log_priors
        gumbel_scores = []
        for move in legal_moves:
            prior = root.move_probs.get(move, 1e-8)
            noise = -math.log(-math.log(random.uniform(0.0001, 0.9999)))
            score = math.log(prior) + 0.3 * noise
            gumbel_scores.append((score, move))
            
        # 3. Select Top-K actions
        K = min(16, len(legal_moves))
        gumbel_scores.sort(reverse=True, key=lambda x: x[0])
        top_k_moves = [m for s, m in gumbel_scores[:K]]
        
        root.untried_moves = top_k_moves[:]
        candidates = top_k_moves[:]
        
        phases = max(1, int(math.ceil(math.log2(K))))
        sims_used = 0
        
        for phase in range(phases):
            is_last_phase = (phase == phases - 1)
            if is_last_phase:
                phase_budget = num_simulations - sims_used
            else:
                phase_budget = num_simulations // phases
                
            sims_completed = 0
            while sims_completed < phase_budget:
                batch_nodes = []
                while len(batch_nodes) < self.batch_size and sims_completed + len(batch_nodes) < phase_budget:
                    node = root
                    
                    while len(node.untried_moves) == 0 and len(node.children) > 0:
                        if node == root:
                            node = self._select_from_candidates(node, candidates)
                        else:
                            node = self.select(node)
                        if node is None:
                            break
                            
                    if node is None:
                        break
                        
                    if len(node.untried_moves) > 0:
                        node = self.expand(node)
                        
                    batch_nodes.append(node)
                    
                if not batch_nodes:
                    break
                    
                values = self.evaluate_batch(batch_nodes)
                
                for node, value in zip(batch_nodes, values):
                    self.backpropagate(node, value)
                    
                sims_completed += len(batch_nodes)
                sims_used += len(batch_nodes)
                
            if not is_last_phase and len(candidates) > 1:
                board = chess.Board(root.fen)
                is_white_turn = (board.turn == chess.WHITE)
                
                cand_scores = []
                for m in candidates:
                    child = root.children.get(m)
                    if child and child.N > 0:
                        q = child.W / child.N
                        if not is_white_turn:
                            q = -q
                    else:
                        q = float('-inf')
                    cand_scores.append((q, m))
                    
                cand_scores.sort(reverse=True, key=lambda x: x[0])
                num_to_keep = max(1, len(candidates) // 2)
                candidates = [m for q, m in cand_scores[:num_to_keep]]
                root.untried_moves = [m for m in root.untried_moves if m in candidates]
                
        if not root.children:
            return random.choice(legal_moves), 0.0
            
        best_n = max(child.N for child in root.children.values())
        best_actions = [m for m, child in root.children.items() if child.N == best_n]
        best_move = max(best_actions, key=lambda m: root.children[m].W / root.children[m].N if root.children[m].N > 0 else float('-inf'))
        target_q = root.children[best_move].W / root.children[best_move].N if root.children[best_move].N > 0 else 0.0
        return best_move, target_q

    def _select_from_candidates(self, node, candidates):
        board = chess.Board(node.fen)
        is_white_turn = (board.turn == chess.WHITE)
        
        best_score = float('-inf')
        best_child = None
        for move in candidates:
            child = node.children.get(move)
            if child is None:
                continue
                
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

    USE_2PLY_LOOKAHEAD = False

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
        
        phases = []
        for fen in fens_to_eval:
            board = chess.Board(fen)
            num_pieces = len(board.piece_map())
            if num_pieces > 24:
                phases.append(0)
            elif num_pieces > 12:
                phases.append(1)
            else:
                phases.append(2)
        phases_tensor = torch.tensor(phases, dtype=torch.long)
        
        with torch.no_grad():
            v_preds, p_from_logits, p_to_logits, _, _, _, _ = self.value_model(tensor_states, phases=phases_tensor)
            v_preds = v_preds.squeeze(-1).tolist()
            if isinstance(v_preds, float):
                v_preds = [v_preds]
            p_from_probs = torch.softmax(p_from_logits, dim=-1).cpu().numpy()
            p_to_probs = torch.softmax(p_to_logits, dim=-1).cpu().numpy()
                
        sf_vals = []
        for i, fen in enumerate(fens_to_eval):
            board = chess.Board(fen)
            
            # Map policy probabilities to node
            node = nodes[node_indices[i]]
            legal_moves = list(board.legal_moves)
            for move in legal_moves:
                prob = p_from_probs[i][move.from_square] * p_to_probs[i][move.to_square]
                node.move_probs[move] = prob
            
            # Normalize probabilities for valid moves only
            total_prob = sum(node.move_probs.values())
            if total_prob > 0:
                for move in node.move_probs:
                    node.move_probs[move] /= total_prob

            if self.USE_2PLY_LOOKAHEAD:
                is_white = (board.turn == chess.WHITE)
                best_minimax = float('-inf') if is_white else float('inf')
                
                reply_fens = []
                terminal_vals = []
                for move in legal_moves:
                    board.push(move)
                    if board.is_checkmate():
                        terminal_vals.append(1.0 if board.turn == chess.BLACK else -1.0)
                        reply_fens.append(None)
                    elif board.is_game_over():
                        terminal_vals.append(0.0)
                        reply_fens.append(None)
                    else:
                        reply_info = self.engine.analyse(board, chess.engine.Limit(depth=1))
                        if "pv" in reply_info and reply_info["pv"]:
                            reply = reply_info["pv"][0]
                            board.push(reply)
                            if board.is_checkmate():
                                terminal_vals.append(1.0 if board.turn == chess.BLACK else -1.0)
                                reply_fens.append(None)
                            elif board.is_game_over():
                                terminal_vals.append(0.0)
                                reply_fens.append(None)
                            else:
                                terminal_vals.append(None)
                                reply_fens.append(board.fen())
                            board.pop()
                        else:
                            terminal_vals.append(0.0)
                            reply_fens.append(None)
                    board.pop()
                
                valid_reply_fens = [f for f in reply_fens if f is not None]
                if valid_reply_fens:
                    rep_tensors = torch.cat([self.env.encode_state(f) for f in valid_reply_fens], dim=0)
                    rep_phases = []
                    for f in valid_reply_fens:
                        rb = chess.Board(f)
                        np_pieces = len(rb.piece_map())
                        if np_pieces > 24: rep_phases.append(0)
                        elif np_pieces > 12: rep_phases.append(1)
                        else: rep_phases.append(2)
                    rep_phases_tensor = torch.tensor(rep_phases, dtype=torch.long)
                    
                    with torch.no_grad():
                        rep_v_preds, _, _, _, _, _, _ = self.value_model(rep_tensors, phases=rep_phases_tensor)
                        rep_v_preds = rep_v_preds.squeeze(-1).tolist()
                        if isinstance(rep_v_preds, float):
                            rep_v_preds = [rep_v_preds]
                            
                    rep_sf_vals = []
                    for f in valid_reply_fens:
                        rb = chess.Board(f)
                        ri = self.engine.analyse(rb, chess.engine.Limit(depth=1))
                        sc = ri["score"].white()
                        cp = 10000 if sc.is_mate() and sc.mate() > 0 else (-10000 if sc.is_mate() else sc.score())
                        rep_sf_vals.append(max(-1.0, min(1.0, cp / 1000.0)))
                        
                    valid_reply_blended = [0.5 * nv + 0.5 * sv for nv, sv in zip(rep_v_preds, rep_sf_vals)]
                else:
                    valid_reply_blended = []
                    
                v_idx = 0
                for t_val in terminal_vals:
                    if t_val is not None:
                        val = t_val
                    else:
                        val = valid_reply_blended[v_idx]
                        v_idx += 1
                        
                    if is_white:
                        if val > best_minimax: best_minimax = val
                    else:
                        if val < best_minimax: best_minimax = val
                        
                sf_vals.append(best_minimax)
            else:
                if getattr(self, 'USE_QUIESCENCE', False):
                    self.QUIESCENCE_CALLS = getattr(self, 'QUIESCENCE_CALLS', 0) + 1
                    
                    def _q_search(b, d, alpha, beta):
                        is_w = b.turn == chess.WHITE
                        if b.is_game_over():
                            if b.is_checkmate(): return -1.0 if is_w else 1.0
                            return 0.0
                            
                        # Stand pat
                        ri = self.engine.analyse(b, chess.engine.Limit(depth=1))
                        sc = ri["score"].white()
                        cp = 10000 if sc.is_mate() and sc.mate() > 0 else (-10000 if sc.is_mate() else sc.score())
                        stand_pat = max(-1.0, min(1.0, cp / 1000.0))
                        
                        if d >= 4: return stand_pat
                        
                        in_check = b.is_check()
                        if not in_check:
                            if is_w:
                                alpha = max(alpha, stand_pat)
                                if alpha >= beta: return alpha
                            else:
                                beta = min(beta, stand_pat)
                                if beta <= alpha: return beta
                                
                        caps = []
                        for m in b.legal_moves:
                            if in_check or b.is_capture(m) or b.gives_check(m):
                                caps.append(m)
                                
                        if not caps: return stand_pat
                        
                        best_val = stand_pat if not in_check else (-float('inf') if is_w else float('inf'))
                        for m in caps:
                            b.push(m)
                            val = _q_search(b, d + 1, alpha, beta)
                            b.pop()
                            if is_w:
                                best_val = max(best_val, val)
                                alpha = max(alpha, best_val)
                            else:
                                best_val = min(best_val, val)
                                beta = min(beta, best_val)
                            if alpha >= beta: break
                        return best_val

                    # Get static first to compare
                    info_static = self.engine.analyse(board, chess.engine.Limit(depth=1))
                    sc_static = info_static["score"].white()
                    cp_s = 10000 if sc_static.is_mate() and sc_static.mate() > 0 else (-10000 if sc_static.is_mate() else sc_static.score())
                    static_val = max(-1.0, min(1.0, cp_s / 1000.0))
                    
                    has_forcing = any(board.is_capture(m) or board.gives_check(m) for m in board.legal_moves)
                    if has_forcing:
                        q_val = _q_search(board, 0, -float('inf'), float('inf'))
                        if abs(q_val - static_val) > 0.1:
                            self.QUIESCENCE_HITS = getattr(self, 'QUIESCENCE_HITS', 0) + 1
                            if getattr(self, 'GUMBEL_LOGGING_ENABLED', False):
                                print(f"[Quiescence] Leaf changed: {static_val:.3f} -> {q_val:.3f} (Hits: {self.QUIESCENCE_HITS}/{self.QUIESCENCE_CALLS})")
                        sf_vals.append(q_val)
                    else:
                        sf_vals.append(static_val)
                else:
                    info = self.engine.analyse(board, chess.engine.Limit(depth=1))
                    score = info["score"].white()
                    if score.is_mate():
                        cp = 10000 if score.mate() > 0 else -10000
                    else:
                        cp = score.score()
                    
                    norm_score = max(-1.0, min(1.0, cp / 1000.0))
                    sf_vals.append(norm_score)
            
        for idx, nn_val, sf_val in zip(node_indices, v_preds, sf_vals):
            if self.USE_2PLY_LOOKAHEAD:
                combined_val = sf_val
            else:
                combined_val = 0.5 * nn_val + 0.5 * sf_val
            values[idx] = combined_val
            
        return values

    def backpropagate(self, node, value):
        current_node = node
        while current_node is not None:
            current_node.N += 1
            current_node.W += value 
            current_node = current_node.parent
