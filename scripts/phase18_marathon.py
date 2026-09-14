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
import chess.syzygy
from collections import deque
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.environment.full_chess_env import FullChessEnv
from scripts.phase14_mcts_planner import ChessValueModelPhase14, BatchingMCTSPlanner, ChessNodePhase14

# ─── CONFIG CONSTANTS ──────────────────────────────────────────────────────────
REPLAY_BUFFER_EPISODES = 200      # Change 1: keep last N episodes in replay buffer
REPLAY_BATCH_SIZE      = 256      # Change 1: examples sampled per consolidation step

# Change 2: LR schedule — cosine decay from LR_MAX → LR_MIN over LR_DECAY_EPISODES
# Rationale: cosine gives a smooth, gradual decay that avoids sudden LR drops that
# can destabilise training.  Steps every episode; restarts are NOT used so the
# model settles steadily rather than bouncing.
LR_MAX            = 5e-4
LR_MIN            = 5e-6
LR_DECAY_EPISODES = 2000          # full cosine period matches the marathon length

# Adaptive loss weighting — replaces the stale fixed-weight scheme.
# Weight for each term = priority / (EMA_of_raw_magnitude + eps)
# so every term contributes ~equally regardless of absolute scale,
# and the weights self-correct as raw magnitudes shift during training.
LOSS_EMA_DECAY  = 0.99    # ~100-step memory (slower = more stable baseline)

# Priority multipliers applied on top of the normalised 1/EMA weight.
# policy/pv determine move selection → higher priority than pure aux heads.
PRIORITY_VALUE  = 1.0
PRIORITY_POLICY = 2.0     # 2× so policy gradient doesn't yield to aux heads
PRIORITY_DEPTH  = 0.7     # auxiliary — helps trunk but not move selection
PRIORITY_DENSE  = 0.7     # auxiliary — same
PRIORITY_PV     = 2.0     # 2× — PV head directly trains move ordering

# Safety clamp: computed weights are clamped so no term exceeds CLAMP_MAX ×
# the median term weight, nor falls below CLAMP_MIN × median.
# Prevents a temporary spike/collapse in one term's EMA from suppressing others.
WEIGHT_CLAMP_MAX = 10.0
WEIGHT_CLAMP_MIN = 0.1

# Change 4: Elo — assumed Stockfish skill level → approximate Elo mapping
# Basis: Stockfish skill level docs + community benchmarks on chess.com/lichess
#   Skill 0≈800, Skill 2≈1100, Skill 3≈1300, Skill 4≈1600, Skill 5≈1900
# We use skill levels 2/3/4/5 in training.
STOCKFISH_ELO = {1: 900, 2: 1100, 3: 1300, 4: 1600, 5: 1900}
ELO_K = 32           # standard K-factor for developing players


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def cosine_lr(episode: int, total_episodes: int) -> float:
    """Cosine decay from LR_MAX to LR_MIN over total_episodes."""
    progress = min(episode / max(1, total_episodes), 1.0)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + math.cos(math.pi * progress))


def elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def elo_update(rating: float, expected: float, actual: float, k: float = ELO_K) -> float:
    return rating + k * (actual - expected)


class AdaptiveLossWeighter:
    """EMA-normalised loss weighter with per-term priority and safety clamps.

    Algorithm per consolidation step:
      1. Update EMA of each raw loss term.
      2. Compute normalised weight[i] = priority[i] / (ema[i] + eps)
         so the *contribution* weight[i] × ema[i] ≈ priority[i] for every term.
         This means policy (priority 2.0) targets twice the gradient contribution
         of value (priority 1.0), regardless of their raw loss scales.
      3. Safety clamp: compute the expected contribution for each term
         (weight × ema). Clamp contributions to [median_contrib × CLAMP_MIN,
         median_contrib × CLAMP_MAX], then back-solve for the clamped weight.
         This prevents a temporarily tiny EMA (e.g. depth after fast learning)
         from ballooning its weight to dominate the gradient.
      4. Return (weights, emas) for logging.
    """

    def __init__(self, decay: float = LOSS_EMA_DECAY, eps: float = 1e-4):
        self.decay = decay
        self.eps   = eps
        # Bootstrap EMAs from the measured before-baseline averages so the
        # weighter starts in a sensible region instead of cold-start guessing.
        self._ema = {
            "value":  0.014,   # measured avg over last 100 eps
            "policy": 7.99,
            "depth":  0.024,
            "dense":  0.035,
            "pv":     8.18,
        }
        self._priorities = {
            "value":  PRIORITY_VALUE,
            "policy": PRIORITY_POLICY,
            "depth":  PRIORITY_DEPTH,
            "dense":  PRIORITY_DENSE,
            "pv":     PRIORITY_PV,
        }

    def step(self, raw: dict) -> tuple[dict, dict]:
        """Update EMAs and return (weights, emas) for logging."""
        d = self.decay
        for k, v in raw.items():
            self._ema[k] = d * self._ema[k] + (1.0 - d) * v

        # Step 1: normalised weights — each term contributes ≈ priority[i]
        weights = {k: self._priorities[k] / (self._ema[k] + self.eps)
                   for k in self._ema}

        # Step 2: safety clamp on *contributions* (w × ema), not raw weights.
        # This keeps policy (high raw magnitude, low weight) correctly
        # dominant over depth (low raw magnitude, high raw weight).
        contribs = {k: weights[k] * (self._ema[k] + self.eps) for k in weights}
        contrib_vals = sorted(contribs.values())
        median_c = contrib_vals[len(contrib_vals) // 2]
        lo_c = median_c * WEIGHT_CLAMP_MIN
        hi_c = median_c * WEIGHT_CLAMP_MAX
        for k in weights:
            clamped_c = max(lo_c, min(hi_c, contribs[k]))
            weights[k] = clamped_c / (self._ema[k] + self.eps)

        return weights, dict(self._ema)


class Phase16Planner(BatchingMCTSPlanner):

    def search_with_q(self, root_fen, num_simulations=100):
        puct_move = None
        if self.USE_GUMBEL_SEARCH and getattr(self, 'GUMBEL_LOGGING_ENABLED', False) and getattr(self, 'GUMBEL_CALLS', 0) < 5000:
            puct_move, _ = self._puct_search_with_q(root_fen, num_simulations)
            
        if self.USE_GUMBEL_SEARCH:
            move, target_q = self._gumbel_search(root_fen, num_simulations)
            if puct_move is not None:
                self.GUMBEL_CALLS = getattr(self, 'GUMBEL_CALLS', 0) + 1
                if move == puct_move:
                    self.GUMBEL_MATCHES = getattr(self, 'GUMBEL_MATCHES', 0) + 1
                if self.GUMBEL_CALLS % 10 == 0:
                    print(f"[Gumbel Instrumentation] Match Rate (Gumbel vs PUCT): {self.GUMBEL_MATCHES}/{self.GUMBEL_CALLS} ({(self.GUMBEL_MATCHES/self.GUMBEL_CALLS)*100:.1f}%)")
            return move, target_q
            
        return self._puct_search_with_q(root_fen, num_simulations)

    def _puct_search_with_q(self, root_fen, num_simulations=100):
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

import json

def write_state(ep, status="running"):
    try:
        with open("state.json", "w") as f:
            json.dump({
                "phase": 18,
                "episode": ep,
                "status": status
            }, f)
    except Exception:
        pass


# ─── REPLAY BUFFER ─────────────────────────────────────────────────────────────
# Each entry stores one ply's data as a dict of already-built tensors so we
# don't need to reconstruct anything at sample time.
class ReplayBuffer:
    """Circular buffer of individual training examples from the last N episodes."""

    def __init__(self, max_episodes: int):
        self.max_episodes = max_episodes
        # Each episode is stored as a list of per-ply dicts; the outer deque
        # holds (episode_id, list_of_ply_dicts).  When full we pop the oldest.
        self._episodes: deque = deque()
        self._total_examples = 0

    def push_episode(self, ep_id: int, plies: list):
        """Add one episode worth of plies.  Each ply is a dict of tensors."""
        if not plies:
            return
        self._episodes.append((ep_id, plies))
        self._total_examples += len(plies)
        while len(self._episodes) > self.max_episodes:
            _, old_plies = self._episodes.popleft()
            self._total_examples -= len(old_plies)

    def sample(self, batch_size: int):
        """Return a batch sampled uniformly across all stored plies.

        Returns (batch_dict, num_episodes_in_batch) where batch_dict contains
        batched tensors ready for forward().
        """
        # Flatten all plies with their source episode id
        all_plies = []
        ep_ids_present = set()
        for ep_id, plies in self._episodes:
            for ply in plies:
                all_plies.append((ep_id, ply))
                ep_ids_present.add(ep_id)

        if not all_plies:
            return None, 0

        k = min(batch_size, len(all_plies))
        chosen = random.sample(all_plies, k)
        sampled_ep_ids = set(ep_id for ep_id, _ in chosen)
        plies_only = [p for _, p in chosen]

        batch = {
            "states":      torch.cat([p["state"]      for p in plies_only], dim=0),
            "phases":      torch.tensor([p["phase"]   for p in plies_only], dtype=torch.long),
            "targets":     torch.cat([p["target"]     for p in plies_only], dim=0),
            "pf_targets":  torch.cat([p["pf_target"]  for p in plies_only], dim=0),
            "pt_targets":  torch.cat([p["pt_target"]  for p in plies_only], dim=0),
            "d_targets":   torch.cat([p["d_target"]   for p in plies_only], dim=0),
            "dn_targets":  torch.cat([p["dn_target"]  for p in plies_only], dim=0),
            "pvf_targets": torch.cat([p["pvf_target"] for p in plies_only], dim=0),
            "pvt_targets": torch.cat([p["pvt_target"] for p in plies_only], dim=0),
        }
        return batch, len(sampled_ep_ids)

    @property
    def num_episodes(self):
        return len(self._episodes)

    @property
    def num_examples(self):
        return self._total_examples


# ─── MAIN TRAINING LOOP ────────────────────────────────────────────────────────

def run_phase18(episodes=2000):
    print("=== Phase 18: The Deep Grind ===")
    print(f"[Config] LR: cosine {LR_MAX:.0e} → {LR_MIN:.0e} over {LR_DECAY_EPISODES} eps")
    print(f"[Config] Replay buffer: {REPLAY_BUFFER_EPISODES} eps, batch {REPLAY_BATCH_SIZE}")
    print(f"[Config] Loss weighting: adaptive EMA (decay={LOSS_EMA_DECAY}) | "
          f"priorities value={PRIORITY_VALUE} policy={PRIORITY_POLICY} "
          f"depth={PRIORITY_DEPTH} dense={PRIORITY_DENSE} pv={PRIORITY_PV} | "
          f"clamp {WEIGHT_CLAMP_MIN}x–{WEIGHT_CLAMP_MAX}x median")
    
    env = FullChessEnv()
    value_model = ChessValueModelPhase14(input_dim=837)
    
    value_model.apply_lora_adapter("FullChess", rank=64)
    value_model.set_active_lora("FullChess")
        
    import glob
    import re
    ckpts = glob.glob("manoid_v2_ckpt_*.pt")
    ckpts_sorted = []
    for c in ckpts:
        m = re.search(r"manoid_v2_ckpt_(\d+)\.pt", c)
        if m:
            ckpts_sorted.append((int(m.group(1)), c))
    ckpts_sorted.sort(key=lambda x: x[0])
    
    latest_ep = 0
    latest_ckpt = None
    if ckpts_sorted:
        latest_ep, latest_ckpt = ckpts_sorted[-1]
        
    recent_state_dicts = []
    for ep_num, c in ckpts_sorted[-3:]:
        recent_state_dicts.append(torch.load(c, map_location='cpu'))

    if latest_ckpt:
        print(f"Resuming from checkpoint: {latest_ckpt} at episode {latest_ep}")
        value_model.load_state_dict(recent_state_dicts[-1], strict=False)
    else:
        print("No V2 checkpoint found, starting fresh.")
        
    write_state(latest_ep, status="running")
        
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    engine.configure({"Skill Level": 3})
    
    tablebase = None
    if os.path.exists("syzygy_345"):
        try:
            tablebase = chess.syzygy.open_tablebase("syzygy_345")
            print("Syzygy tablebases loaded successfully.")
        except Exception as e:
            print("Failed to load Syzygy tablebases:", e)
    
    mse_history = []

    # ── Change 2: LR starts at LR_MAX; updated each episode ──────────────────
    current_lr = cosine_lr(latest_ep, LR_DECAY_EPISODES)
    trainable_params = [p for p in value_model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=current_lr)
    print(f"[Config] Initial LR = {current_lr:.2e}")

    wins = 0
    losses = 0
    draws = 0
    rolling_results = []

    # ── Change 1: Replay buffer ───────────────────────────────────────────────
    replay_buffer = ReplayBuffer(max_episodes=REPLAY_BUFFER_EPISODES)

    # ── Adaptive loss weighter (replaces fixed W_VALUE/W_POLICY/etc) ──────────
    loss_weighter = AdaptiveLossWeighter(decay=LOSS_EMA_DECAY)

    # ── Change 4: Elo tracking ────────────────────────────────────────────────
    # Note: starting at 738 based on known state from previous logs, since
    # we aren't serialising this variable to disk yet.
    manoid_elo = 738.0

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
        "rnbqkbnr/pppppppp/8/8/8/1P6/P1PPPPPP/RNBQKBNR b KQkq - 0 1",
        "4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1"
    ]

    for ep in range(latest_ep + 1, episodes + 1):

        # ── Change 2: Update LR at the start of each episode ─────────────────
        current_lr = cosine_lr(ep, LR_DECAY_EPISODES)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        ACTIVE_LEARNING_RATIO = 0.3
        chosen_by_al = False
        al_score = 0.0
        
        if len(recent_state_dicts) >= 2 and random.random() < ACTIVE_LEARNING_RATIO:
            candidates = opening_book + [chess.STARTING_FEN]
            candidate_tensors = torch.cat([env.encode_state(f) for f in candidates], dim=0)
            candidate_phases = []
            for f in candidates:
                rb = chess.Board(f)
                np_pieces = len(rb.piece_map())
                if np_pieces > 24: candidate_phases.append(0)
                elif np_pieces > 12: candidate_phases.append(1)
                else: candidate_phases.append(2)
            candidate_phases_tensor = torch.tensor(candidate_phases, dtype=torch.long)
            
            val_preds = []
            p_from_preds = []
            p_to_preds = []
            
            with torch.no_grad():
                original_sd = value_model.state_dict()
                for sd in recent_state_dicts:
                    value_model.load_state_dict(sd, strict=False)
                    v, pf, pt, _, _, _, _ = value_model(candidate_tensors, phases=candidate_phases_tensor)
                    val_preds.append(v)
                    p_from_preds.append(torch.softmax(pf, dim=-1))
                    p_to_preds.append(torch.softmax(pt, dim=-1))
                value_model.load_state_dict(original_sd, strict=False)
                
            val_preds = torch.stack(val_preds)
            p_from_preds = torch.stack(p_from_preds)
            p_to_preds = torch.stack(p_to_preds)
            
            val_variance = val_preds.var(dim=0).squeeze(-1)
            
            mean_p_from = p_from_preds.mean(dim=0)
            mean_p_to = p_to_preds.mean(dim=0)
            entropy_from = -(mean_p_from * torch.log(mean_p_from + 1e-8)).sum(dim=-1)
            entropy_to = -(mean_p_to * torch.log(mean_p_to + 1e-8)).sum(dim=-1)
            
            total_uncertainty = val_variance * 10.0 + (entropy_from + entropy_to) * 0.1
            
            best_idx = torch.argmax(total_uncertainty).item()
            start_fen = candidates[best_idx]
            al_score = total_uncertainty[best_idx].item()
            start_desc = f"[Start: AL (unc: {al_score:.4f}) {start_fen}]"
            chosen_by_al = True
        else:
            if random.random() < 0.50:
                start_fen = chess.STARTING_FEN
                start_desc = "[Start: Natural]"
            else:
                start_fen = random.choice(opening_book)
                start_desc = f"[Start: GM Book: {start_fen}]"
            
        # Curriculum ordering based on Elo
        if manoid_elo < 1000:
            level_choices = [1, 2, 3, 4]
            level_weights = [0.8, 0.2, 0.0, 0.0]
        else:
            w_count = rolling_results.count('W')
            win_rate = w_count / max(1, len(rolling_results))
            level_choices = [2, 3, 4, 5]
            level_weights = [0.7, 0.2, 0.05, 0.05]
            if win_rate > 0.4:
                level_weights = [0.2, 0.6, 0.15, 0.05]
            if win_rate > 0.6:
                level_weights = [0.05, 0.2, 0.6, 0.15]
                
        opponent_level = random.choices(level_choices, weights=level_weights, k=1)[0]
        engine.configure({"Skill Level": opponent_level})
        board = chess.Board(start_fen)
        agent_color = chess.WHITE if ep % 2 == 1 else chess.BLACK
        desc = "White" if agent_color == chess.WHITE else "Black"
        print(f"Starting FEN: {start_desc} | Opponent Level: {opponent_level} | Level Mix: {level_weights}")
        
        planner = Phase16Planner(env, value_model, engine, batch_size=32)
        planner.USE_QUIESCENCE = True
        
        # ── Per-episode buffers (used to build replay entries) ────────────────
        ep_plies = []          # list of per-ply dicts for replay buffer
        ep_mse_sum = 0.0
        moves_made = 0
        ep_tb_hits = 0
        ep_sf_hits = 0
        
        while not board.is_game_over() and moves_made < 200:
            curr_np = len(board.piece_map())
            curr_phase = 0
            if curr_np > 24: curr_phase = 0
            elif curr_np > 12: curr_phase = 1
            else: curr_phase = 2
            
            if board.turn == agent_color:
                best_move, target_q = planner.search_with_q(board.fen(), num_simulations=128)
                if best_move is None:
                    break
                    
                tensor_state = env.encode_state(board.fen())
                with torch.no_grad():
                    v_pred, p_from, p_to, d_pred, dense_pred, _, _ = value_model(tensor_state, phases=torch.tensor([curr_phase], dtype=torch.long))
                    raw_pred = v_pred.item()
                
                loss_val = (raw_pred - target_q) ** 2
                ep_mse_sum += loss_val
                
                # Soft policy targets & PV auxiliary head
                info = engine.analyse(board, chess.engine.Limit(depth=1), multipv=5)
                TEMPERATURE = 0.5
                
                soft_target_from = torch.zeros(64, dtype=torch.float32)
                soft_target_to = torch.zeros(64, dtype=torch.float32)
                evals = []
                move_indices = []
                
                pv_target_from = torch.zeros(3, dtype=torch.long)
                pv_target_to = torch.zeros(3, dtype=torch.long)
                
                if "pv" in info[0] and info[0]["pv"]:
                    pv_moves = info[0]["pv"][:3]
                    for idx, m in enumerate(pv_moves):
                        pv_target_from[idx] = m.from_square
                        pv_target_to[idx] = m.to_square
                    if len(pv_moves) > 0:
                        last_m = pv_moves[-1]
                        for idx in range(len(pv_moves), 3):
                            pv_target_from[idx] = last_m.from_square
                            pv_target_to[idx] = last_m.to_square
                
                for pv_info in info:
                    if "pv" in pv_info and pv_info["pv"]:
                        m = pv_info["pv"][0]
                        score = pv_info["score"].white()
                        if board.turn == chess.BLACK:
                            score = pv_info["score"].black()
                        cp = 10000 if score.is_mate() and score.mate() > 0 else (-10000 if score.is_mate() else score.score())
                        evals.append(cp)
                        move_indices.append(m)
                        
                if evals:
                    evals_tensor = torch.tensor(evals, dtype=torch.float32) / 100.0
                    probs = torch.softmax(evals_tensor / TEMPERATURE, dim=0)
                    for m, p in zip(move_indices, probs):
                        soft_target_from[m.from_square] += p
                        soft_target_to[m.to_square] += p
                else:
                    m = best_move
                    soft_target_from[m.from_square] = 1.0
                    soft_target_to[m.to_square] = 1.0
                
                # Dense reward per ply
                current_eval = info[0]["score"].white() if board.turn == chess.WHITE else info[0]["score"].black()
                current_cp = 10000 if current_eval.is_mate() and current_eval.mate() > 0 else (-10000 if current_eval.is_mate() else current_eval.score())
                current_cp = current_cp / 100.0
                
                board.push(best_move)
                info_after = engine.analyse(board, chess.engine.Limit(depth=1))
                after_eval = info_after["score"].white() if not board.turn == chess.WHITE else info_after["score"].black()
                after_cp = 10000 if after_eval.is_mate() and after_eval.mate() > 0 else (-10000 if after_eval.is_mate() else after_eval.score())
                after_cp = after_cp / 100.0
                
                dense_reward = (-after_cp) - current_cp
                dense_reward = max(-1.0, min(1.0, dense_reward * 0.1))
                board.pop()

                # Search depth distillation auxiliary head
                info_d6 = engine.analyse(board, chess.engine.Limit(depth=6))
                d6_score = info_d6["score"].white() if board.turn == chess.WHITE else info_d6["score"].black()
                d6_cp = 10000 if d6_score.is_mate() and d6_score.mate() > 0 else (-10000 if d6_score.is_mate() else d6_score.score())
                d6_cp = d6_cp / 100.0
                depth_delta = (d6_cp - current_cp) * 0.1
                depth_delta = max(-1.0, min(1.0, depth_delta))
                
                # Syzygy tablebase grounding
                if tablebase is not None and len(board.piece_map()) <= 5:
                    try:
                        wdl = tablebase.probe_wdl(board)
                        if wdl > 0: tb_val = 1.0
                        elif wdl < 0: tb_val = -1.0
                        else: tb_val = 0.0
                        target_q = tb_val
                        ep_tb_hits += 1
                    except Exception:
                        ep_sf_hits += 1
                else:
                    ep_sf_hits += 1

                # ── Change 1: build ply dict for replay buffer ────────────────
                ply_dict = {
                    "state":      tensor_state,
                    "phase":      curr_phase,
                    "target":     torch.tensor([[target_q]], dtype=torch.float32),
                    "pf_target":  soft_target_from.unsqueeze(0),
                    "pt_target":  soft_target_to.unsqueeze(0),
                    "d_target":   torch.tensor([[depth_delta]], dtype=torch.float32),
                    "dn_target":  torch.tensor([[dense_reward]], dtype=torch.float32),
                    "pvf_target": pv_target_from.unsqueeze(0),
                    "pvt_target": pv_target_to.unsqueeze(0),
                }
                ep_plies.append(ply_dict)
                
                # Mirror augmentation
                ENABLE_MIRROR = True
                if ENABLE_MIRROR:
                    mirrored_board = board.transform(chess.flip_horizontal)
                    mirrored_tensor_state = env.encode_state(mirrored_board.fen())
                    
                    mirrored_soft_target_from = torch.zeros(64, dtype=torch.float32)
                    mirrored_soft_target_to = torch.zeros(64, dtype=torch.float32)
                    for sq in range(64):
                        m_sq = sq - (sq % 8) + (7 - (sq % 8))
                        mirrored_soft_target_from[m_sq] = soft_target_from[sq]
                        mirrored_soft_target_to[m_sq] = soft_target_to[sq]
                    
                    mirrored_pv_from = torch.zeros(3, dtype=torch.long)
                    mirrored_pv_to = torch.zeros(3, dtype=torch.long)
                    for idx in range(3):
                        sq_f = pv_target_from[idx].item()
                        sq_t = pv_target_to[idx].item()
                        mirrored_pv_from[idx] = sq_f - (sq_f % 8) + (7 - (sq_f % 8))
                        mirrored_pv_to[idx] = sq_t - (sq_t % 8) + (7 - (sq_t % 8))
                    
                    if evals:
                        mir_ply = {
                            "state":      mirrored_tensor_state,
                            "phase":      curr_phase,
                            "target":     torch.tensor([[target_q]], dtype=torch.float32),
                            "pf_target":  mirrored_soft_target_from.unsqueeze(0),
                            "pt_target":  mirrored_soft_target_to.unsqueeze(0),
                            "d_target":   torch.tensor([[depth_delta]], dtype=torch.float32),
                            "dn_target":  torch.tensor([[dense_reward]], dtype=torch.float32),
                            "pvf_target": mirrored_pv_from.unsqueeze(0),
                            "pvt_target": mirrored_pv_to.unsqueeze(0),
                        }
                        ep_plies.append(mir_ply)
                
                board.push(best_move)
            else:
                result = engine.play(board, chess.engine.Limit(time=0.01))
                if result.move is None:
                    break
                board.push(result.move)
            moves_made += 1
            
        # ── Game outcome ──────────────────────────────────────────────────────
        final_state_desc = ""
        game_result_str = "D"  # default: draw
        if board.is_checkmate():
            if board.turn == agent_color:
                final_state_desc = "Checkmated (Lost)"
                losses += 1
                game_result_str = "L"
                rolling_results.append('L')
            else:
                final_state_desc = "Checkmate (Won)"
                wins += 1
                game_result_str = "W"
                rolling_results.append('W')
        elif board.is_stalemate() or board.is_repetition() or board.is_fifty_moves():
            final_state_desc = "Draw/Stalemate"
            draws += 1
            rolling_results.append('D')
        else:
            final_state_desc = "Max moves"
            draws += 1
            rolling_results.append('D')
            
        if len(rolling_results) > 50:
            rolling_results.pop(0)

        # ── Change 4: Elo update ──────────────────────────────────────────────
        opp_elo = STOCKFISH_ELO.get(opponent_level, 1300)
        expected_score = elo_expected(manoid_elo, opp_elo)
        actual_score = 1.0 if game_result_str == "W" else (0.5 if game_result_str == "D" else 0.0)
        manoid_elo = elo_update(manoid_elo, expected_score, actual_score)

        # ── Change 1: Push this episode into replay buffer ────────────────────
        replay_buffer.push_episode(ep, ep_plies)
            
        ep_mse = ep_mse_sum / max(1, len(ep_plies)) if ep_plies else 0.0
        mse_history.append(ep_mse)
        
        print(f"[Episode {ep}] Manoid plays {desc} | Result: {final_state_desc} ({moves_made} moves) | Avg MSE: {ep_mse:.4f}")
        w = rolling_results.count('W')
        l = rolling_results.count('L')
        d = rolling_results.count('D')
        print(f"  -> Rolling Stats (Last {len(rolling_results)}): {w}W / {l}L / {d}D | Elo: {manoid_elo:.0f} (vs SF{opponent_level}≈{opp_elo})")
        
        skip_consolidation = False
        print("  -> Plasticity Override: Forcing TRAIN for Policy Bootstrapping.")
                
        # ── Change 1: Sample from replay buffer instead of current episode ────
        if not skip_consolidation and replay_buffer.num_examples > 0:
            batch, num_unique_eps = replay_buffer.sample(REPLAY_BATCH_SIZE)

            if batch is not None:
                states_tensor     = batch["states"]
                phases_tensor     = batch["phases"]
                targets_tensor    = batch["targets"]
                pf_targets_tensor = batch["pf_targets"]
                pt_targets_tensor = batch["pt_targets"]
                d_targets_tensor  = batch["d_targets"]
                dn_targets_tensor = batch["dn_targets"]
                pvf_targets_tensor= batch["pvf_targets"]
                pvt_targets_tensor= batch["pvt_targets"]

                # Report replay buffer health
                print(f"  -> Replay Buffer: {replay_buffer.num_examples} examples "
                      f"across {replay_buffer.num_episodes} episodes | "
                      f"Batch: {states_tensor.shape[0]} samples from {num_unique_eps} unique eps")

                value_model.train()
                
                phase_v_losses = {0: 0.0, 1: 0.0, 2: 0.0}
                phase_counts = {0: 0, 1: 0, 2: 0}
                
                for _ in range(5):
                    optimizer.zero_grad()
                    v_preds, p_from_logits, p_to_logits, d_preds, dense_preds, pv_f_logits, pv_t_logits = \
                        value_model(states_tensor, phases=phases_tensor)
                    
                    # ── Compute raw losses ────────────────────────────────────
                    v_loss_raw    = nn.functional.mse_loss(v_preds, targets_tensor)
                    pf_loss_raw   = nn.functional.cross_entropy(p_from_logits, pf_targets_tensor)
                    pt_loss_raw   = nn.functional.cross_entropy(p_to_logits,   pt_targets_tensor)
                    p_loss_raw    = pf_loss_raw + pt_loss_raw
                    d_loss_raw    = nn.functional.mse_loss(d_preds,   d_targets_tensor)
                    dn_loss_raw   = nn.functional.mse_loss(dense_preds, dn_targets_tensor)
                    pvf_loss_raw  = nn.functional.cross_entropy(pv_f_logits.transpose(1, 2), pvf_targets_tensor)
                    pvt_loss_raw  = nn.functional.cross_entropy(pv_t_logits.transpose(1, 2), pvt_targets_tensor)
                    pv_loss_raw   = pvf_loss_raw + pvt_loss_raw

                    # ── Adaptive weighting: EMA-normalised + priority ─────────
                    raw_magnitudes = {
                        "value":  v_loss_raw.item(),
                        "policy": p_loss_raw.item(),
                        "depth":  d_loss_raw.item(),
                        "dense":  dn_loss_raw.item(),
                        "pv":     pv_loss_raw.item(),
                    }
                    w, emas = loss_weighter.step(raw_magnitudes)

                    v_contrib  = w["value"]  * v_loss_raw
                    p_contrib  = w["policy"] * p_loss_raw
                    d_contrib  = w["depth"]  * d_loss_raw
                    dn_contrib = w["dense"]  * dn_loss_raw
                    pv_contrib = w["pv"]     * pv_loss_raw
                    loss = v_contrib + p_contrib + d_contrib + dn_contrib + pv_contrib

                    loss.backward()
                    optimizer.step()
                    
                    with torch.no_grad():
                        sq_errs = (v_preds - targets_tensor) ** 2
                        for ph in range(3):
                            mask = (phases_tensor == ph)
                            if mask.any():
                                phase_v_losses[ph] = sq_errs[mask].mean().item()
                                phase_counts[ph] = mask.sum().item()
                            
                value_model.eval()
                print(f"  -> Grounding Hits: TB {ep_tb_hits} vs SF {ep_sf_hits}")
                print(f"  -> Phase MSE | Open({phase_counts[0]}): {phase_v_losses[0]:.4f} | "
                      f"Mid({phase_counts[1]}): {phase_v_losses[1]:.4f} | "
                      f"End({phase_counts[2]}): {phase_v_losses[2]:.4f}")
                # Raw losses (unchanged format — enables trend comparison across runs)
                print(f"  -> Loss [raw]     | V: {v_loss_raw.item():.4f} | "
                      f"P: {p_loss_raw.item():.4f} | "
                      f"D: {d_loss_raw.item():.4f} | "
                      f"Dn: {dn_loss_raw.item():.4f} | "
                      f"PV: {pv_loss_raw.item():.4f}")
                # Adaptive weights computed this step (watch these evolve)
                print(f"  -> Loss [weights]  | V: {w['value']:.4f} | "
                      f"P: {w['policy']:.4f} | "
                      f"D: {w['depth']:.4f} | "
                      f"Dn: {w['dense']:.4f} | "
                      f"PV: {w['pv']:.4f}")
                # EMA baselines the weights are computed from
                print(f"  -> EMA [magnitudes]| V: {emas['value']:.4f} | "
                      f"P: {emas['policy']:.4f} | "
                      f"D: {emas['depth']:.4f} | "
                      f"Dn: {emas['dense']:.4f} | "
                      f"PV: {emas['pv']:.4f}")
                # Weighted contributions + totals
                print(f"  -> Loss [weighted] | V: {v_contrib.item():.4f} | "
                      f"P: {p_contrib.item():.4f} | "
                      f"D: {d_contrib.item():.4f} | "
                      f"Dn: {dn_contrib.item():.4f} | "
                      f"PV: {pv_contrib.item():.4f} | "
                      f"Total: {loss.item():.4f} | "
                      f"LR: {current_lr:.2e}")

        if ep % 25 == 0:
            ckpt_name = f"manoid_v2_ckpt_{ep}.pt"
            sd = value_model.state_dict()
            torch.save(sd, ckpt_name)
            recent_state_dicts.append({k: v.cpu().clone() for k, v in sd.items()})
            if len(recent_state_dicts) > 3:
                recent_state_dicts.pop(0)
            avg_mse_25 = sum(mse_history[-25:]) / 25
            print(f"\n--- Checkpoint Saved: {ckpt_name} ---")
            print(f"Stats over last 25 episodes: Wins: {wins}, Losses: {losses}, Draws: {draws}")
            print(f"Average MSE over last 25 episodes: {avg_mse_25:.4f}")
            print(f"Replay Buffer: {replay_buffer.num_examples} examples / {replay_buffer.num_episodes} episodes")
            print(f"Estimated Elo: {manoid_elo:.0f}  |  LR: {current_lr:.2e}")
            print("----------------------------------------\n")
            wins = 0
            losses = 0
            draws = 0
            write_state(ep, status="running")

    torch.save(value_model.state_dict(), "full_chess_lora_weights_final.pt")
    write_state(episodes, status="done")
    engine.quit()

if __name__ == "__main__":
    run_phase18()
