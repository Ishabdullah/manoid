import torch
import torch.nn.functional as F
import os

class EpisodicMemory:
    """
    Fast-Binding Episodic Cache (Phase 2).
    A pure PyTorch Tensor Buffer enabling instant one-shot storage and retrieval
    without retraining the WorldModel weights.
    Schema: [state, action, outcome, confidence]
    """
    def __init__(self, state_dim: int = 512, action_dim: int = 16, max_size: int = 10000, device: str = 'cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_size = max_size
        self.device = device
        
        # Pre-allocate tensor buffers
        self.states = torch.zeros((max_size, state_dim), device=device)
        self.actions = torch.zeros((max_size, action_dim), device=device)
        self.outcomes = torch.zeros((max_size, state_dim), device=device)
        self.confidences = torch.zeros((max_size, 1), device=device)
        
        self.ptr = 0
        self.size = 0
        
    def store(self, state: torch.Tensor, action: torch.Tensor, outcome: torch.Tensor, confidence: torch.Tensor):
        """Store a one-shot experience in the ring buffer."""
        # Ensure 2D shapes (batch, dim)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)
            outcome = outcome.unsqueeze(0)
            confidence = confidence.unsqueeze(0)
            
        batch_size = state.shape[0]
        
        for i in range(batch_size):
            self.states[self.ptr] = state[i]
            self.actions[self.ptr] = action[i]
            self.outcomes[self.ptr] = outcome[i]
            self.confidences[self.ptr] = confidence[i]
            
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
            
    def retrieve(self, query_state: torch.Tensor, query_action: torch.Tensor, k: int = 1):
        """
        Query the memory for the most similar past experiences based on cosine similarity
        of the concatenated (state, action) vectors.
        """
        if self.size == 0:
            return None, None, None
            
        if query_state.dim() == 1:
            query_state = query_state.unsqueeze(0)
            query_action = query_action.unsqueeze(0)
            
        # Get valid portion of memory
        valid_states = self.states[:self.size]
        valid_actions = self.actions[:self.size]
        
        # Concatenate state and action for context matching
        q_concat = torch.cat([query_state, query_action], dim=-1)
        mem_concat = torch.cat([valid_states, valid_actions], dim=-1)
        
        # L2 Normalize for cosine similarity
        q_norm = F.normalize(q_concat, p=2, dim=-1)
        mem_norm = F.normalize(mem_concat, p=2, dim=-1)
        
        # Matrix multiplication computes pairwise cosine similarities
        similarities = torch.mm(q_norm, mem_norm.t())
        
        # Retrieve top-k matches
        top_k = min(k, self.size)
        top_sims, top_indices = torch.topk(similarities, k=top_k, dim=-1)
        
        retrieved_outcomes = self.outcomes[top_indices]
        retrieved_confidences = self.confidences[top_indices]
        
        return retrieved_outcomes, retrieved_confidences, top_sims

    def save(self, filepath: str):
        """Persist memory buffer to disk."""
        torch.save({
            'states': self.states[:self.size],
            'actions': self.actions[:self.size],
            'outcomes': self.outcomes[:self.size],
            'confidences': self.confidences[:self.size],
            'ptr': self.ptr,
            'size': self.size
        }, filepath)
        
    def load(self, filepath: str):
        """Load memory buffer from disk."""
        if os.path.exists(filepath):
            data = torch.load(filepath, map_location=self.device)
            self.size = data['size']
            self.ptr = data['ptr']
            self.states[:self.size] = data['states']
            self.actions[:self.size] = data['actions']
            self.outcomes[:self.size] = data['outcomes']
            self.confidences[:self.size] = data['confidences']
