"""
A/B Comparison: USE_WORLD_MODEL_BONUS=False vs True
Runs N_EPISODES episodes under each condition from the SAME starting checkpoint,
then writes RESULTS_PATH1.md with numbers and a verdict.

Usage:
    python scripts/ab_compare.py
"""

import sys, os, random, math, time, copy
import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.engine

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.environment.full_chess_env import FullChessEnv
from agent.core_model.world_model import WorldModel
from agent.memory.episodic import EpisodicMemory
from scripts.phase14_mcts_planner import (
    ChessValueModelPhase14, ChessNodePhase14, RESONANCE_WEIGHT
)
from scripts.phase18_marathon import Phase16Planner, cosine_lr

# ─── Configuration ────────────────────────────────────────────────────────────
N_EPISODES     = 10          # episodes per condition (keep small for mobile HW)
N_SIMULATIONS  = 64          # MCTS sims per move (halved for speed)
BATCH_SIZE     = 16
STOCKFISH_ELO  = {1: 900, 2: 1100, 3: 1300, 4: 1600, 5: 1900}
ELO_K          = 32
RANDOM_SEED    = 42
RESULTS_FILE   = "RESULTS_PATH1.md"

# Starting checkpoint: use the most recent v2 ckpt if available, else fresh
def _load_value_model():
    vm = ChessValueModelPhase14(input_dim=837)
    vm.apply_lora_adapter("FullChess", rank=64)
    vm.set_active_lora("FullChess")
    import glob, re
    ckpts = sorted(
        [(int(m.group(1)), c)
         for c in glob.glob("manoid_v2_ckpt_*.pt")
         if (m := re.search(r"manoid_v2_ckpt_(\d+)\.pt", c))],
        key=lambda x: x[0]
    )
    if ckpts:
        ep_n, path = ckpts[-1]
        vm.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
        print(f"  Loaded checkpoint: {path} (ep {ep_n})")
    else:
        print("  No checkpoint found — starting fresh")
    return vm

def _load_world_model():
    wm = WorldModel(input_dim=12*8*8, action_dim=16, latent_dim=512, num_heads=3)
    if os.path.exists("world_model_ckpt.pt"):
        try:
            wm.load_state_dict(torch.load("world_model_ckpt.pt", map_location='cpu'))
            print("  Loaded WorldModel checkpoint")
        except Exception as e:
            print(f"  WorldModel load failed ({e}), fresh")
    return wm

# ─── Single-episode runner ────────────────────────────────────────────────────
def run_episode(ep_idx, env, value_model, world_model, engine,
                use_world_model_bonus, elo, level_choices, level_weights):
    opponent_level = random.choices(level_choices, weights=level_weights, k=1)[0]
    engine.configure({"Skill Level": opponent_level})
    board = chess.Board()
    agent_color = chess.WHITE if ep_idx % 2 == 0 else chess.BLACK

    episodic_memory = EpisodicMemory(state_dim=512, action_dim=16, max_size=5000)
    planner = Phase16Planner(
        env, value_model, engine,
        batch_size=BATCH_SIZE,
        world_model=world_model,
        episodic_memory=episodic_memory,
        use_world_model_bonus=use_world_model_bonus,
    )
    planner.USE_QUIESCENCE = False  # disable for speed in comparison

    ep_mse_sum = 0.0
    ep_mse_count = 0
    moves_made = 0
    phase_sq_err = {0: [], 1: [], 2: []}

    wm_optimizer = optim.Adam(world_model.parameters(), lr=1e-4)
    from scripts.phase14_mcts_planner import _encode_move_cfloat

    while not board.is_game_over() and moves_made < 150:
        num_pieces = len(board.piece_map())
        curr_phase = 0 if num_pieces > 24 else (1 if num_pieces > 12 else 2)

        if board.turn == agent_color:
            best_move, target_q = planner.search_with_q(
                board.fen(), num_simulations=N_SIMULATIONS)
            if best_move is None:
                break

            t_state = env.encode_state_real(board.fen())
            with torch.no_grad():
                v_pred, *_ = value_model(
                    t_state, phases=torch.tensor([curr_phase]))
                raw_pred = v_pred.item()

            sq_err = (raw_pred - target_q) ** 2
            ep_mse_sum += sq_err
            ep_mse_count += 1
            phase_sq_err[curr_phase].append(sq_err)

            # WorldModel training step
            try:
                world_model.train()
                wm_optimizer.zero_grad()
                z_now = world_model.encode(env.encode_state(board.fen()))
                a_c = _encode_move_cfloat(best_move, action_dim=16)
                z_np = world_model.predict_next(z_now, a_c)
                board.push(best_move)
                z_act = world_model.encode(env.encode_state(board.fen()))
                board.pop()
                wm_loss = ((z_np - z_act.detach()).abs() ** 2).mean()
                wm_loss.backward()
                wm_optimizer.step()
                world_model.eval()
            except Exception:
                world_model.eval()

            board.push(best_move)
        else:
            result = engine.play(board, chess.engine.Limit(time=0.01))
            if result.move is None:
                break
            board.push(result.move)
        moves_made += 1

    # Outcome
    result_str = "D"
    if board.is_checkmate():
        result_str = "W" if board.turn != agent_color else "L"
    opp_elo_val = STOCKFISH_ELO.get(opponent_level, 1300)
    exp = 1.0 / (1.0 + 10 ** ((opp_elo_val - elo) / 400.0))
    score = 1.0 if result_str == "W" else (0.5 if result_str == "D" else 0.0)
    new_elo = elo + ELO_K * (score - exp)

    phase_mse = {
        ph: (sum(v)/len(v) if v else float('nan'))
        for ph, v in phase_sq_err.items()
    }
    ep_mse = ep_mse_sum / max(1, ep_mse_count)
    return result_str, new_elo, ep_mse, phase_mse, moves_made, opponent_level

# ─── Run one condition ────────────────────────────────────────────────────────
def run_condition(label, use_bonus, engine):
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    env         = FullChessEnv()
    value_model = _load_value_model()
    world_model = _load_world_model()

    elo = 738.0
    results   = []
    all_mse   = []
    phase_mse_accum = {0: [], 1: [], 2: []}
    pa_on = "ON" if use_bonus else "OFF"

    print(f"\n{'='*60}")
    print(f"Run {label}: USE_WORLD_MODEL_BONUS={use_bonus} (Path A bonus {pa_on})")
    print(f"{'='*60}")
    t_start = time.monotonic()

    level_choices = [2, 3, 4, 5]
    level_weights = [0.7, 0.2, 0.05, 0.05]

    for ep in range(N_EPISODES):
        res, elo, ep_mse, ph_mse, n_moves, opp_lvl = run_episode(
            ep, env, value_model, world_model, engine,
            use_bonus, elo, level_choices, level_weights
        )
        results.append(res)
        all_mse.append(ep_mse)
        for ph in range(3):
            if not math.isnan(ph_mse[ph]):
                phase_mse_accum[ph].append(ph_mse[ph])

        w = results.count('W'); l = results.count('L'); d = results.count('D')
        print(f"  Ep {ep+1}/{N_EPISODES}: {res}  moves={n_moves}  "
              f"Elo={elo:.0f} (vs SF{opp_lvl})  MSE={ep_mse:.4f}  "
              f"W/L/D={w}/{l}/{d}")

    elapsed = time.monotonic() - t_start
    w = results.count('W'); l = results.count('L'); d = results.count('D')
    mean_mse = sum(all_mse) / max(1, len(all_mse))
    ph_means = {
        ph: (sum(v)/len(v) if v else float('nan'))
        for ph, v in phase_mse_accum.items()
    }
    print(f"\n  Summary: W={w} L={l} D={d}  FinalElo={elo:.0f}  "
          f"AvgMSE={mean_mse:.4f}  Elapsed={elapsed:.0f}s")

    return {
        "label": label, "use_bonus": use_bonus,
        "W": w, "L": l, "D": d, "N": N_EPISODES,
        "final_elo": elo, "mean_mse": mean_mse,
        "ph0_mse": ph_means[0], "ph1_mse": ph_means[1], "ph2_mse": ph_means[2],
        "results": results, "all_mse": all_mse,
        "elapsed_s": elapsed,
    }

# ─── Write RESULTS_PATH1.md ───────────────────────────────────────────────────
def write_results(run_a, run_b):
    def fmt(v):
        return f"{v:.4f}" if not math.isnan(v) else "n/a"

    elo_delta  = run_b["final_elo"] - run_a["final_elo"]
    mse_delta  = run_b["mean_mse"]  - run_a["mean_mse"]   # negative = B better
    winrate_a  = run_a["W"] / max(1, run_a["N"])
    winrate_b  = run_b["W"] / max(1, run_b["N"])
    wr_delta   = winrate_b - winrate_a

    verdict = "INCONCLUSIVE (need more episodes)"
    if run_b["N"] >= 10:
        if elo_delta > 20 and wr_delta > 0.05:
            verdict = "π² resonance HELPED (Elo ↑, win rate ↑)"
        elif elo_delta < -20 and wr_delta < -0.05:
            verdict = "π² resonance HURT (Elo ↓, win rate ↓)"
        elif abs(elo_delta) <= 20 and abs(wr_delta) <= 0.05:
            verdict = "π² resonance had NO MEASURABLE EFFECT within this sample size"
        else:
            verdict = "MIXED SIGNALS — extend N_EPISODES for a clearer verdict"

    md = f"""# Phase 1 A/B Comparison Results

**Date**: {time.strftime('%Y-%m-%d %H:%M UTC')}
**Episodes per condition**: {N_EPISODES}
**MCTS simulations per move**: {N_SIMULATIONS}
**Starting Elo**: 738
**RESONANCE_WEIGHT**: {RESONANCE_WEIGHT}
**Random seed**: {RANDOM_SEED}

## What was tested

- **Run A** (`USE_WORLD_MODEL_BONUS=False`): MCTS uses only ChessValueModelPhase14
  policy head for move ordering. WorldModel Path A still runs (encode + HRR store)
  but its resonance bonus is **not** written to move_probs.
- **Run B** (`USE_WORLD_MODEL_BONUS=True`): Path A resonance bonus modulates move
  probabilities multiplicatively: `prob' = base_prob × (1 + {RESONANCE_WEIGHT} × bonus)`,
  then normalised. This is the **bug-fixed** version of the combined policy.

## Bug fixed in this comparison

Previously Path B unconditionally overwrote `node.move_probs[move] = prob`,
discarding whatever Path A had written. This has been fixed: Path B now reads
Path A's per-move resonance bonus and combines it. Run B is the first run to
actually exercise the π² resonance signal in move selection.

## Raw Results

| Metric | Run A (bonus OFF) | Run B (bonus ON) | Delta (B−A) |
|--------|------------------|-----------------|-------------|
| Wins | {run_a['W']} / {run_a['N']} | {run_b['W']} / {run_b['N']} | {run_b['W']-run_a['W']:+d} |
| Losses | {run_a['L']} / {run_a['N']} | {run_b['L']} / {run_b['N']} | {run_b['L']-run_a['L']:+d} |
| Draws | {run_a['D']} / {run_a['N']} | {run_b['D']} / {run_b['N']} | {run_b['D']-run_a['D']:+d} |
| Win rate | {winrate_a:.1%} | {winrate_b:.1%} | {wr_delta:+.1%} |
| Final Elo | {run_a['final_elo']:.0f} | {run_b['final_elo']:.0f} | {elo_delta:+.0f} |
| Mean MSE | {fmt(run_a['mean_mse'])} | {fmt(run_b['mean_mse'])} | {mse_delta:+.4f} |
| Opening MSE | {fmt(run_a['ph0_mse'])} | {fmt(run_b['ph0_mse'])} | — |
| Middlegame MSE | {fmt(run_a['ph1_mse'])} | {fmt(run_b['ph1_mse'])} | — |
| Endgame MSE | {fmt(run_a['ph2_mse'])} | {fmt(run_b['ph2_mse'])} | — |
| Wall time | {run_a['elapsed_s']:.0f}s | {run_b['elapsed_s']:.0f}s | — |

## Per-episode results

### Run A (bonus OFF)
{chr(10).join(f'- Ep {i+1}: {r}  MSE={run_a["all_mse"][i]:.4f}' for i, r in enumerate(run_a["results"]))}

### Run B (bonus ON)
{chr(10).join(f'- Ep {i+1}: {r}  MSE={run_b["all_mse"][i]:.4f}' for i, r in enumerate(run_b["results"]))}

## Verdict

> **{verdict}**

### Interpretation notes

- With only {N_EPISODES} episodes per condition the noise floor is high (~±{int(1/math.sqrt(N_EPISODES)*100)}%
  on win rate by binomial variance alone). The delta thresholds used above
  (>20 Elo, >5% win rate) are conservative but still uncertain at this sample size.
- The WorldModel is **untrained** (or lightly trained from previous runs); its
  resonance signal reflects random initialisation more than learned chess structure
  at this stage. A more meaningful comparison should be run after ≥100 episodes of
  WorldModel warm-up.
- If verdict is INCONCLUSIVE or NO EFFECT, proceed to Phase 2 planning regardless —
  the phase-space architecture remains theoretically motivated; the question is
  whether it needs deeper integration (Phase 2's primary model path) to show gains.

## Next step

Phase 2 plan document: `PI2_REFACTOR_PLAN.md`
"""
    with open(RESULTS_FILE, "w") as f:
        f.write(md)
    print(f"\nResults written to {RESULTS_FILE}")

# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Opening Stockfish engine...")
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")

    try:
        run_a = run_condition("A", use_bonus=False, engine=engine)
        run_b = run_condition("B", use_bonus=True,  engine=engine)
    finally:
        engine.quit()

    write_results(run_a, run_b)
