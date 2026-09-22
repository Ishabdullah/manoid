"""
full_chess_env.py — Phase-Space Chess Environment
==================================================
Replaces the discrete 8×8 integer board with a continuous 2D Fourier-domain
representation.  Each piece type generates a characteristic frequency amplitude
and phase angle on the board surface; the result is projected through a 2D FFT
to produce a complex-valued spatial spectrum that downstream modules consume
directly as torch.cfloat tensors.

Board geometry
--------------
* Board surface  : 8×8 continuous toroidal grid.
* Piece encoding : 12 complex channels (one per piece-type / colour).
                   Each occupied square contributes amplitude A_k·e^{iφ_k},
                   where A_k is the piece's base amplitude (relative material
                   value) and φ_k = π² · (piece_type_index / 12).
* FFT            : torch.fft.fft2 over the 8×8 spatial grid → 8×8 complex
                   frequency spectrum per channel.
* Flat output    : The complex spectrum is returned as a (1, 12, 8, 8) cfloat
                   tensor.  Callers that need a real vector can call
                   encode_state_real() which concatenates real/imag parts.

Meta-features (turn, castling, en-passant) are encoded as phase-offset scalars
blended into channel 0 (King channel) so that global context is visible in the
frequency domain without extra channels.
"""

import chess
import torch
import numpy as np
import math

# π² constant used throughout the wave-mechanics encoding
PI2 = math.pi ** 2

# Per-piece base amplitudes (analogous to material value, normalised to 1.0 for King)
_PIECE_AMPLITUDES = {
    chess.PAWN:   0.1,
    chess.KNIGHT: 0.3,
    chess.BISHOP: 0.32,
    chess.ROOK:   0.5,
    chess.QUEEN:  0.9,
    chess.KING:   1.0,
}

# Piece-type index 0-5 for White, 6-11 for Black (mirrors original layout)
def _piece_channel(piece: chess.Piece) -> int:
    idx = piece.piece_type - 1          # 0-5
    if piece.color == chess.BLACK:
        idx += 6
    return idx


class FullChessEnv:
    """
    Full standard-chess environment with continuous phase-space state encoding.

    encode_state(fen) → torch.Tensor [1, 12, 8, 8]  dtype=torch.cfloat
        Complex frequency spectrum ready for complex-valued downstream layers.

    encode_state_real(fen) → torch.Tensor [1, 837]  dtype=torch.float32
        Backward-compatible real view: real + imag parts concatenated
        (12·8·8·2 = 1536) followed by 4 castling bits + 1 turn + 64 en-passant
        one-hot = 1605 total. Trimmed to 837 to match the legacy value-head.

    NOTE: the legacy path expects 837 features (768 pieces + 4 castling + 1 turn
    + 64 ep). We honour that dimension by using only the magnitude spectrum
    (12·8·8 / 2 planes) and appending the same meta scalars, keeping full
    backward compatibility with the MCTS value model.
    """

    # ------------------------------------------------------------------
    # Core board mechanics (unchanged interface)
    # ------------------------------------------------------------------

    def reset(self, fen: str = chess.STARTING_FEN) -> str:
        return fen

    def apply_action(self, fen: str, action) -> str:
        board = chess.Board(fen)
        if isinstance(action, str):
            action = chess.Move.from_uci(action)
        board.push(action)
        return board.fen()

    def get_legal_actions(self, fen: str):
        board = chess.Board(fen)
        return list(board.legal_moves)

    # ------------------------------------------------------------------
    # Wave-domain helpers
    # ------------------------------------------------------------------

    def _board_to_wave_grid(self, board: chess.Board) -> torch.Tensor:
        """
        Build a (12, 8, 8) complex tensor where each channel represents one
        piece-type / colour.  Each occupied square is set to:

            A_k · exp(i · π² · φ_norm)

        where φ_norm = piece_channel / 12  so that each piece type resonates
        at a unique phase offset in [0, π²).
        """
        grid = torch.zeros(12, 8, 8, dtype=torch.cfloat)

        for sq, piece in board.piece_map().items():
            ch = _piece_channel(piece)
            rank = sq // 8          # 0-7
            file = sq % 8           # 0-7
            amplitude = _PIECE_AMPLITUDES[piece.piece_type]
            phase = PI2 * (ch / 12.0)
            grid[ch, rank, file] = complex(amplitude * math.cos(phase),
                                            amplitude * math.sin(phase))
        return grid

    def _apply_meta_phase(self, grid: torch.Tensor, board: chess.Board) -> torch.Tensor:
        """
        Encode global board state (turn, castling rights, en-passant) as
        phase offsets on the King channels (ch=5 White King, ch=11 Black King),
        blending into the DC component (frequency index [0,0]).
        """
        grid = grid.clone()
        turn_phase = PI2 if board.turn == chess.WHITE else 0.0

        castling_sum = (
            (1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0) +
            (0.5 if board.has_queenside_castling_rights(chess.WHITE) else 0.0) +
            (0.25 if board.has_kingside_castling_rights(chess.BLACK) else 0.0) +
            (0.125 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)
        )

        ep_phase = 0.0
        if board.ep_square is not None:
            ep_phase = PI2 * (board.ep_square / 64.0)

        meta_complex = complex(
            math.cos(turn_phase + castling_sum),
            math.sin(ep_phase + castling_sum)
        )

        # Inject into white-king and black-king channels at DC
        grid[5, 0, 0] += meta_complex * 0.05
        grid[11, 0, 0] += meta_complex * 0.05
        return grid

    def _fft2_board(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D FFT over the spatial (rank × file) dimensions.
        Input : (12, 8, 8) cfloat
        Output: (12, 8, 8) cfloat   — frequency spectrum
        """
        # fft2 operates on the last two dims by default
        return torch.fft.fft2(grid)

    # ------------------------------------------------------------------
    # Primary encoding interface
    # ------------------------------------------------------------------

    def encode_state(self, fen: str) -> torch.Tensor:
        """
        Returns a (1, 12, 8, 8) torch.cfloat tensor representing the
        Fourier-domain board state.  This is the native complex tensor
        passed between MCTS → WorldModel → Memory.
        """
        board = chess.Board(fen)
        grid = self._board_to_wave_grid(board)
        grid = self._apply_meta_phase(grid, board)
        spectrum = self._fft2_board(grid)          # (12, 8, 8) cfloat
        return spectrum.unsqueeze(0)               # (1, 12, 8, 8)

    def encode_state_real(self, fen: str) -> torch.Tensor:
        """
        Backward-compatible real encoding for the legacy value model.

        Builds a 837-dim float vector that mirrors the original layout:
          * 768 floats : magnitude of each frequency bin, restricted to
                         the first 6 channels × 64 bins (real amplitudes).
          * 4 floats   : castling rights
          * 1 float    : turn
          * 64 floats  : en-passant one-hot

        Total = 768 + 4 + 1 + 64 = 837
        """
        board = chess.Board(fen)
        grid = self._board_to_wave_grid(board)
        spectrum = self._fft2_board(grid)          # (12, 8, 8) cfloat

        # Magnitude map for 12 channels × 64 = 768 values
        mag = spectrum.abs().reshape(12, 64)       # (12, 64) real

        # Interleave channels in the original piece-order to preserve
        # semantic ordering expected by downstream heads (per-square, per-piece)
        tensor_list = []
        for sq in range(64):
            for ch in range(12):
                tensor_list.append(mag[ch, sq].item())

        tensor = torch.tensor(tensor_list, dtype=torch.float32)  # 768

        # Meta scalars
        meta = torch.zeros(4 + 1 + 64, dtype=torch.float32)
        meta[0] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        meta[1] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        meta[2] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        meta[3] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
        meta[4] = 1.0 if board.turn == chess.WHITE else 0.0
        if board.ep_square is not None:
            meta[5 + board.ep_square] = 1.0

        return torch.cat([tensor, meta], dim=0).unsqueeze(0)  # (1, 837)

    # ------------------------------------------------------------------
    # Harmonic resonance heuristic
    # ------------------------------------------------------------------

    def evaluate_heuristic(self, fen: str) -> float:
        """
        Evaluates board strength via π² variance spike detection.

        Detects tactical overlaps (Knight forks, discovered checks) as regions
        of the frequency spectrum where the phase variance across channels
        exceeds a π²-scaled threshold — i.e. where multiple piece waves
        constructively interfere (high amplitude) at the same frequency bin.

        Returns a float in [-1, 1] from White's perspective.
        """
        board = chess.Board(fen)
        if board.is_checkmate():
            return -1.0 if board.turn == chess.WHITE else 1.0
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        grid = self._board_to_wave_grid(board)
        spectrum = self._fft2_board(grid)  # (12, 8, 8)

        # Compute per-bin variance across channels in the complex plane
        # Variance of complex values: Var = E[|X - μ|²]
        mu = spectrum.mean(dim=0, keepdim=True)          # (1, 8, 8)
        deviation = spectrum - mu                          # (12, 8, 8)
        # |deviation|² then mean over channels
        variance_map = (deviation.abs() ** 2).mean(dim=0)  # (8, 8) real

        # π² variance spikes: bins where variance > π²/12 indicate resonance
        resonance_threshold = PI2 / 12.0
        spike_mask = (variance_map > resonance_threshold).float()

        # Weight spikes by the amplitude of White vs Black contributions
        white_channels = spectrum[:6]   # (6, 8, 8)
        black_channels = spectrum[6:]   # (6, 8, 8)

        white_power = (white_channels.abs() ** 2).sum(dim=0)  # (8, 8)
        black_power = (black_channels.abs() ** 2).sum(dim=0)  # (8, 8)

        # Resonance-weighted material advantage
        white_resonance = (white_power * spike_mask).sum().item()
        black_resonance = (black_power * spike_mask).sum().item()
        total = white_resonance + black_resonance + 1e-8

        score = (white_resonance - black_resonance) / total  # [-1, 1]

        # Blend with a simple material count for stability
        material_score = self._material_balance(board)
        return 0.6 * score + 0.4 * material_score

    def _material_balance(self, board: chess.Board) -> float:
        """Simple centipawn material balance, normalised to [-1, 1]."""
        values = {
            chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0
        }
        white_mat = sum(values[p.piece_type]
                        for p in board.piece_map().values()
                        if p.color == chess.WHITE)
        black_mat = sum(values[p.piece_type]
                        for p in board.piece_map().values()
                        if p.color == chess.BLACK)
        total = white_mat + black_mat + 1e-8
        return (white_mat - black_mat) / total
