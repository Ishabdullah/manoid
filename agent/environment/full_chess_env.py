import chess
import torch
import numpy as np

class FullChessEnv:
    def __init__(self):
        pass

    def reset(self, fen=chess.STARTING_FEN):
        return fen

    def apply_action(self, fen, action):
        board = chess.Board(fen)
        if type(action) is str:
            action = chess.Move.from_uci(action)
        board.push(action)
        return board.fen()

    def get_legal_actions(self, fen):
        board = chess.Board(fen)
        return list(board.legal_moves)

    def encode_state(self, fen):
        board = chess.Board(fen)
        
        # 12 piece types * 64 squares = 768
        # piece order: P, N, B, R, Q, K (White), p, n, b, r, q, k (Black)
        piece_map = board.piece_map()
        
        tensor = np.zeros(768 + 4 + 1 + 64, dtype=np.float32)
        
        for sq, piece in piece_map.items():
            piece_idx = piece.piece_type - 1 # 0 to 5
            if piece.color == chess.BLACK:
                piece_idx += 6
            idx = sq * 12 + piece_idx
            tensor[idx] = 1.0
            
        offset = 768
        
        # Turn
        tensor[offset] = 1.0 if board.turn == chess.WHITE else 0.0
        offset += 1
        
        # Castling
        tensor[offset] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        tensor[offset+1] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        tensor[offset+2] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        tensor[offset+3] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
        offset += 4
        
        # En passant
        if board.ep_square is not None:
            tensor[offset + board.ep_square] = 1.0
            
        return torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)
    
    def evaluate_heuristic(self, fen):
        # Placeholder, handled by MCTS planner
        return 0.0
