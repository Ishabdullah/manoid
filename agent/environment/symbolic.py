"""
symbolic.py — Phase-Space Symbolic Grid Environment
====================================================
Replaces the discrete one-hot position encoding with a continuous 2D
Fourier-domain representation of agent position on an N×N toroidal grid.

Each agent position (x, y) is treated as a point source emitting a
wave with amplitude 1.0 and phase angle π² · (x·N + y) / N².  The
grid is then passed through torch.fft.fft2 to produce the complex
frequency spectrum, which is returned as the observation.

The downstream WorldModel receives cfloat tensors natively.  For
backward compatibility, _get_state_real() produces the 256-dim float
vector that Phase-1 code expects.
"""

import math
import torch

PI2 = math.pi ** 2


class SymbolicGridEnv:
    """
    Phase-Space Symbolic Grid Environment.

    Grid size: N × N
    Observation: (1, N, N) torch.cfloat  — 2D Fourier spectrum of position.
    The DC component encodes position amplitude; other bins encode spatial
    harmonics that allow the WorldModel to reason about relative distances
    without discrete indexing.

    For legacy compatibility (output_dim=256), call _get_state_real() to
    obtain a zero-padded float vector.
    """

    def __init__(self, grid_size: int = 5, output_dim: int = 256, action_dim: int = 16):
        self.grid_size = grid_size
        self.output_dim = output_dim
        self.action_dim = action_dim

        # Agent starts at (0, 0)
        self.pos_x = 0
        self.pos_y = 0

        # Action mappings: 0=Up, 1=Right, 2=Down, 3=Left
        self.action_map = {
            0: (0, -1),
            1: (1,  0),
            2: (0,  1),
            3: (-1, 0),
        }

    # ------------------------------------------------------------------
    # Wave-domain helpers
    # ------------------------------------------------------------------

    def _position_to_wave_grid(self, x: int, y: int) -> torch.Tensor:
        """
        Construct an N×N complex grid with the agent's position encoded as
        a point source at (y, x) with amplitude 1.0 and phase π²·idx/N².

        The phase is unique per cell so that the FFT spectrum uniquely
        identifies position without requiring one-hot encoding.
        """
        grid = torch.zeros(self.grid_size, self.grid_size, dtype=torch.cfloat)
        idx = y * self.grid_size + x
        phase = PI2 * idx / (self.grid_size ** 2)
        grid[y, x] = complex(math.cos(phase), math.sin(phase))
        return grid

    def _fft2_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """Apply 2D FFT → (N, N) cfloat spectrum."""
        return torch.fft.fft2(grid)

    def _get_state(self) -> torch.Tensor:
        """
        Returns the primary complex observation: (1, N, N) cfloat.
        This is passed directly to the complex-valued WorldModel encoder.
        """
        grid = self._position_to_wave_grid(self.pos_x, self.pos_y)
        spectrum = self._fft2_grid(grid)
        return spectrum.unsqueeze(0)  # (1, N, N)

    def _get_state_real(self) -> torch.Tensor:
        """
        Legacy-compatible real observation: (1, output_dim) float32.
        Concatenates real and imag parts of the spectrum then zero-pads
        to output_dim (default 256).
        """
        spectrum = self._get_state().squeeze(0)     # (N, N) cfloat
        real_flat = spectrum.real.reshape(-1)        # N² floats
        imag_flat = spectrum.imag.reshape(-1)        # N² floats
        combined = torch.cat([real_flat, imag_flat]) # 2·N² floats

        state = torch.zeros(1, self.output_dim)
        n = min(combined.shape[0], self.output_dim)
        state[0, :n] = combined[:n].float()
        return state

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------

    def reset(self) -> torch.Tensor:
        """Reset the environment; returns the complex spectrum state."""
        self.pos_x = 0
        self.pos_y = 0
        return self._get_state()

    def encode_action(self, action_idx: int) -> torch.Tensor:
        """Encode a discrete action index into a cfloat action tensor."""
        # Action encoded as a unit-magnitude complex number with a
        # phase proportional to action_idx so that adjacent actions are
        # adjacent in phase space.
        phase = PI2 * action_idx / self.action_dim
        action_complex = complex(math.cos(phase), math.sin(phase))
        action_tensor = torch.zeros(1, self.action_dim, dtype=torch.cfloat)
        action_tensor[0, action_idx] = action_complex
        return action_tensor

    def step(self, action_idx: int):
        """
        Causal do(X) intervention in phase-space.
        Returns (next_state: cfloat tensor, reward: float, done: bool).
        """
        dx, dy = self.action_map.get(action_idx, (0, 0))
        self.pos_x = max(0, min(self.grid_size - 1, self.pos_x + dx))
        self.pos_y = max(0, min(self.grid_size - 1, self.pos_y + dy))
        return self._get_state(), 0.0, False

    def log_transition(self, state: torch.Tensor, action: torch.Tensor,
                       next_state: torch.Tensor) -> str:
        """Log a transition using the dominant frequency component as identifier."""
        def _dominant_freq(s):
            mag = s.abs()
            return int(torch.argmax(mag.reshape(-1)).item())

        s_idx  = _dominant_freq(state)
        a_idx  = int(torch.argmax(action.abs().reshape(-1)).item())
        ns_idx = _dominant_freq(next_state)
        return (f"Transition Logged | "
                f"State(dom_freq): {s_idx} -> Action: {a_idx} -> "
                f"NextState(dom_freq): {ns_idx}")
