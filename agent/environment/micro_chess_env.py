import chess
import torch

class MicroChessEnv:
    def __init__(self, fen=None):
        if fen:
            self.board = chess.Board(fen)
        else:
            self.board = chess.Board("k7/8/2K5/8/8/8/8/7R w - - 0 1")

    def reset(self, fen=None):
        if fen:
            self.board = chess.Board(fen)
        else:
            self.board = chess.Board("k7/8/2K5/8/8/8/8/7R w - - 0 1")
        return self.board.fen()

    def get_legal_actions(self, fen):
        """Tier 3: Absolute discrete legal facts."""
        board = chess.Board(fen)
        return list(board.legal_moves)

    def apply_action(self, fen, move):
        board = chess.Board(fen)
        board.push(move)
        return board.fen()

    def encode_state(self, fen):
        """
        Neural Translation Layer: Converts FEN to a flat tensor for Tier 2/WorldModel.
        """
        board = chess.Board(fen)
        tensor = torch.zeros(1, 768)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                color_offset = 0 if piece.color == chess.WHITE else 6
                piece_offset = piece.piece_type - 1
                idx = square * 12 + color_offset + piece_offset
                tensor[0, idx] = 1.0
        return tensor

    def evaluate_heuristic(self, fen, role="predator", agent_color=chess.WHITE):
        """
        Tier 2 proxy: The LoRA adapter's heuristic reward logic.
        """
        board = chess.Board(fen)
        
        if board.is_checkmate():
            if board.turn == agent_color:
                return -1.0 # Agent was checkmated
            else:
                return 1.0 # Agent delivered checkmate
                
        if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition() or board.is_fifty_moves():
            return -1.0 if role == "predator" else 0.5
            
        value = 0.0
        predator_color = agent_color if role == "predator" else (not agent_color)
        prey_color = not predator_color
        
        wk_square = board.king(predator_color)
        bk_square = board.king(prey_color)
        
        if wk_square is not None and bk_square is not None:
            wk_rank, wk_file = chess.square_rank(wk_square), chess.square_file(wk_square)
            bk_rank, bk_file = chess.square_rank(bk_square), chess.square_file(bk_square)
            
            distance = max(abs(wk_rank - bk_rank), abs(wk_file - bk_file))
            if role == "predator":
                value -= distance * 0.01
                if bk_rank == 0 or bk_rank == 7 or bk_file == 0 or bk_file == 7:
                    value += 0.1
            else:
                value += distance * 0.01
                if bk_rank == 0 or bk_rank == 7 or bk_file == 0 or bk_file == 7:
                    value -= 0.1
                
        return value
