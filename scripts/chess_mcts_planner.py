import math
import random
import chess
from agent.environment.micro_chess_env import MicroChessEnv

class ChessNode:
    def __init__(self, fen, parent=None, move=None, untried_moves=None):
        self.fen = fen
        self.parent = parent
        self.move = move
        self.N = 0
        self.W = 0.0
        self.children = {}
        self.untried_moves = untried_moves if untried_moves is not None else []

class ChessMCTSPlanner:
    def __init__(self, env):
        self.env = env
        self.exploration_weight = math.sqrt(2)

    def search(self, root_fen, num_simulations=200):
        legal_moves = self.env.get_legal_actions(root_fen)
        if not legal_moves:
            return None
            
        root = ChessNode(root_fen, untried_moves=list(legal_moves))
        
        for sim in range(num_simulations):
            node = root
            
            # 1. Selection
            while len(node.untried_moves) == 0 and len(node.children) > 0:
                node = self.select(node)
                
            # 2. Expansion
            if len(node.untried_moves) > 0:
                node = self.expand(node)
                
            # 3. Rollout / Evaluation (No random rollout, just immediate heuristic to prevent noise)
            value = self.evaluate(node)
            
            # 4. Backpropagation
            self.backpropagate(node, value)
            
        if not root.children:
            return random.choice(legal_moves)
            
        # Tie-breaker using Q-value
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
                # Black wants to minimize White's score
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

    def evaluate(self, node):
        return self.env.evaluate_heuristic(node.fen)

    def backpropagate(self, node, value):
        current_node = node
        while current_node is not None:
            current_node.N += 1
            current_node.W += value 
            current_node = current_node.parent
