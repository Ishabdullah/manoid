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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.environment.micro_chess_env import MicroChessEnv
from scripts.chess_mcts_planner import ChessNode
from scripts.chess_neural_assay import ChessValueModel

class Phase12MCTSPlanner:
    def __init__(self, env, value_model, role="predator", agent_color=chess.WHITE):
        self.env = env
        self.value_model = value_model
        self.exploration_weight = math.sqrt(2)
        self.role = role
        self.agent_color = agent_color

    def evaluate(self, node):
        board = chess.Board(node.fen)
        
        if board.is_checkmate():
            if board.turn == self.agent_color:
                return -1.0 # Agent mated
            else:
                return 1.0 # Agent delivered mate
        if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition() or board.is_fifty_moves() or board.is_game_over():
            return -1.0 if self.role == "predator" else 0.5
            
        tensor_state = self.env.encode_state(node.fen)
        with torch.no_grad():
            value = self.value_model(tensor_state).item()
        return value

    def search_with_q(self, root_fen, num_simulations=50):
        legal_moves = list(chess.Board(root_fen).legal_moves)
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

    def select(self, node):
        board = chess.Board(node.fen)
        is_agent_turn = (board.turn == self.agent_color)
        
        best_score = float('-inf')
        best_child = None
        for move, child in node.children.items():
            if child.N == 0:
                return child
                
            q_val = child.W / child.N
            if not is_agent_turn:
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
        child_node = ChessNode(next_fen, parent=node, move=move, untried_moves=list(next_untried))
        node.children[move] = child_node
        return child_node

    def backpropagate(self, node, value):
        current_node = node
        while current_node is not None:
            current_node.N += 1
            current_node.W += value 
            current_node = current_node.parent

def run_phase12(episodes=50):
    print("=== Phase 12: Adversarial Self-Play & Contextual Rewards ===")
    
    env = MicroChessEnv()
    value_model = ChessValueModel()
    
    value_model.apply_lora_adapter("ChessEndgame", rank=8)
    value_model.set_active_lora("ChessEndgame")
    
    try:
        value_model.load_state_dict(torch.load("chess_lora_weights.pt"))
        print("[System] Loaded initial weights from chess_lora_weights.pt")
    except:
        pass
        
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    engine.configure({"Skill Level": 5})
    
    scenarios = [
        {"role": "predator", "color": chess.WHITE, "fen": "k7/8/2K5/8/8/8/8/7R w - - 0 1", "desc": "Predator (White)"},
        {"role": "predator", "color": chess.BLACK, "fen": "K7/8/2k5/8/8/8/8/7r b - - 0 1", "desc": "Predator (Black)"},
        {"role": "prey", "color": chess.WHITE, "fen": "K7/8/2k5/8/8/8/8/7r w - - 0 1", "desc": "Prey (White)"},
        {"role": "prey", "color": chess.BLACK, "fen": "k7/8/2K5/8/8/8/8/7R b - - 0 1", "desc": "Prey (Black)"},
    ]

    mse_history = []
    trainable_params = [p for p in value_model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=5e-4)

    for ep in range(1, episodes + 1):
        scenario = random.choice(scenarios)
        role = scenario["role"]
        agent_color = scenario["color"]
        board = chess.Board(scenario["fen"])
        
        # slight scramble
        if ep > 5:
            for _ in range(random.randint(1, 4)):
                if not board.is_game_over():
                    board.push(random.choice(list(board.legal_moves)))
        
        planner = Phase12MCTSPlanner(env, value_model, role=role, agent_color=agent_color)
        
        episodic_states = []
        episodic_targets = []
        ep_mse_sum = 0.0
        moves_made = 0
        
        while not board.is_game_over() and moves_made < 30:
            if board.turn == agent_color:
                best_move, target_q = planner.search_with_q(board.fen(), num_simulations=200)
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
                result = engine.play(board, chess.engine.Limit(time=0.05))
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
        
        print(f"[Episode {ep}] {scenario['desc']} | Result: {final_state_desc} ({moves_made} moves) | Avg MSE: {ep_mse:.4f}")
        
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

    torch.save(value_model.state_dict(), "chess_lora_weights_phase12.pt")
    engine.quit()

if __name__ == "__main__":
    run_phase12()
