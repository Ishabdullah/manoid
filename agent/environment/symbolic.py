import torch

class SymbolicGridEnv:
    """
    Phase 3: Custom Symbolic Grid Environment.
    A simple deterministic gridworld for pure causal interventions `do(X)`.
    
    Grid size: N x N
    State mapping: To maintain compatibility with Phase 1 (input_dim=256), 
    the environment emits a 256-dimensional zero-padded one-hot vector 
    representing the agent's absolute position.
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
            1: (1, 0),
            2: (0, 1),
            3: (-1, 0)
        }
        
    def reset(self) -> torch.Tensor:
        """Reset the environment and return the initial state."""
        self.pos_x = 0
        self.pos_y = 0
        return self._get_state()
        
    def _get_state(self) -> torch.Tensor:
        """Construct the high-dimensional observation vector."""
        state = torch.zeros(1, self.output_dim)
        # Flatten the 2D coordinate into a 1D index
        idx = self.pos_y * self.grid_size + self.pos_x
        state[0, idx] = 1.0
        return state
        
    def encode_action(self, action_idx: int) -> torch.Tensor:
        """Encode a discrete action index into the expected continuous action_dim."""
        action = torch.zeros(1, self.action_dim)
        action[0, action_idx] = 1.0
        return action

    def step(self, action_idx: int):
        """
        The causal do(X) intervention.
        Updates internal state deterministically and returns (next_state, reward, done).
        """
        dx, dy = self.action_map.get(action_idx, (0, 0))
        
        # Apply movement with bounds checking
        self.pos_x = max(0, min(self.grid_size - 1, self.pos_x + dx))
        self.pos_y = max(0, min(self.grid_size - 1, self.pos_y + dy))
        
        # In this minimal sandbox, we just return the new state. 
        # Rewards and 'done' are kept trivial for now since we are studying causal state prediction.
        return self._get_state(), 0.0, False

    def log_transition(self, state, action, next_state):
        """Architectural requirement: Log real transitions."""
        # Simple logging format for the Verifier
        s_idx = torch.argmax(state).item()
        a_idx = torch.argmax(action).item()
        ns_idx = torch.argmax(next_state).item()
        return f"Transition Logged | State(pos): {s_idx} -> Action: {a_idx} -> NextState(pos): {ns_idx}"
