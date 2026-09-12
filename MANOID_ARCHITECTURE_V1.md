# MANOID Architecture V1: Cognitive Pipeline & System Baseline

## 3-Tier Cognitive Architecture

### Tier 1: Anti-Forgetting Local Memory Buffer (Replay)
The foundation of Manoid's experiential learning. Tier 1 acts as a continuous replay buffer, storing sequential episodes of states, actions, and rewards. It prevents catastrophic forgetting during continuous online learning by providing a stable reservoir of past experiences for consolidation, ensuring the agent retains previously mastered scenarios even as it learns new ones.

### Tier 2: Neural Intuition & LoRA Adapters
Tier 2 represents the agent's internalized intuition, parameterized via dynamic LoRA (Low-Rank Adaptation) layers injected into a core Value Model (e.g., `ChessValueModel`). 
- **Heuristic Normalization**: For Phase 12/13, rewards have been strictly bounded to the `[-1.0, 1.0]` scale to prevent gradient explosions. 
- **Concept Routing**: Specific tasks (like "ChessEndgame") map to specific LoRA adapters (`apply_lora_adapter("ChessEndgame", rank=8)`), allowing rapid, isolated skill acquisition without overwriting base knowledge.

### Tier 3: Symbolic Engine & Expansion
The absolute ground truth of the environment. Tier 3 interfaces directly with strict physical/logical rules (e.g., `python-chess` legal move generation). It acts as the flawless expansion mechanism for the MCTS planner, completely distinct from Tier 2's neural approximations.

## Advanced Guardrails & Planning

### The Hybrid Guardrail (Dynamic Plasticity Gate)
Controls when the agent initiates offline neural consolidation, driven strictly by relative error momentum rather than hardcoded thresholds:
- **Two-Strike Regression**: If the Mean Squared Error (MSE) of the Tier 2 predictions strictly increases for two consecutive episodes (`curr > prev1` AND `prev1 > prev2`), the gate opens and triggers `TRAIN`.
- **10x Magnitude Bypass**: A failsafe for sparse terminal discoveries. If the error suddenly spikes to 10x the previous episode (`curr > (prev1 * 10.0)`), consolidation triggers immediately, bypassing the two-strike requirement to instantly map the new critical state (e.g., a sudden Checkmate).

### MCTS Planner
The bridge between Tier 3 logic and Tier 2 intuition. 
- **UCB1 Selection**: Alternating minimax formulation (inverting Q-values for Black) augmented by the standard Upper Confidence Bound formula scaled for `[-1.0, 1.0]` heuristics.
- **Zero-Sum FEN State**: Board states are efficiently translated into 1x768 flat tensors, capturing the spatial layout of pieces to act as the direct input to the Tier 2 intuition model during leaf node rollout evaluation.

## Adversarial Self-Play Curriculum
- **Contextual Rewards**: Asymmetric reward matrix based on the assigned role. A "Predator" (K+R) is penalized for draws (-1.0), while the "Prey" (K) is rewarded for survival (+0.5).
- **Color Invariance**: The curriculum dynamically scrambles roles and colors (Predator White/Black, Prey White/Black) to force the neural engine to learn abstract spatial mating nets rather than memorizing color-specific piece arrays.
