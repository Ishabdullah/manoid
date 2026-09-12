import torch
import sys
import os

# Add parent directory to path so we can import agent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.core_model.world_model import WorldModel

def verify_world_model():
    print("--- Phase 1: Live Verifier ---")
    
    # Initialize model
    input_dim = 256
    action_dim = 16
    latent_dim = 512
    
    model = WorldModel(input_dim=input_dim, action_dim=action_dim, latent_dim=latent_dim)
    
    # Print parameter count to confirm size
    param_count = model.count_parameters()
    print(f"Model initialized with {param_count:,} trainable parameters.")
    
    # Create dummy observation (batch_size=4, input_dim=256)
    batch_size = 4
    obs_t = torch.randn(batch_size, input_dim)
    action_t = torch.randn(batch_size, action_dim)
    obs_t_next = torch.randn(batch_size, input_dim)
    
    print("\nExecuting forward pass...")
    
    # 1. Encode
    z_t = model.encode(obs_t)
    print(f"Encoded shape (z_t): {z_t.shape} | Expected: ({batch_size}, {latent_dim})")
    
    # 2. Predict next state
    z_pred_next = model.predict_next(z_t, action_t)
    print(f"Predicted next shape (z_pred_next): {z_pred_next.shape} | Expected: ({batch_size}, {latent_dim})")
    
    # 3. Encode actual next state (for surprise calculation)
    z_actual_next = model.encode(obs_t_next)
    
    # 4. Compute surprise
    surprise = model.compute_surprise(z_pred_next, z_actual_next)
    print(f"Surprise metric shape: {surprise.shape} | Expected: ({batch_size},)")
    print(f"Surprise values: {surprise.tolist()}")
    
    print("\n--- Live Verification Success! ---")

if __name__ == "__main__":
    verify_world_model()
