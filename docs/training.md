# AlphaChess Training and Testing Design

This document details the design of the fault-tolerant training script, configuration management, and the testing script for AlphaChess.

## 1. Fault-Tolerant Training Script (`train.py`)

The training script is designed to be robust and efficient, enabling continuous learning and recovery from interruptions.

### 1.1. Self-Play Loop

- **Game Generation:** The training process begins with a self-play loop where the current iteration of the AlphaChess agent plays against itself. Each game is guided by the MCTS algorithm, which uses the neural network to make informed move decisions.
- **Data Collection:** During each self-play game, the following data points are collected for every move:
  - **Board State:** The encoded representation of the chess board before the move.
  - **MCTS Policy:** The probability distribution over legal moves derived from the MCTS visit counts (often referred to as the "improved policy" or "search policy"). This is a crucial target for the neural network's policy head.
  - **Game Outcome:** The final result of the game (+1 for a win, -1 for a loss, 0 for a draw) from the perspective of the player whose turn it was at that state.
- **Replay Buffer:** The collected (state, MCTS policy, game outcome) tuples are stored in a large circular replay buffer. This buffer serves as a dataset for training the neural network, ensuring a diverse set of experiences and breaking temporal correlations.

### 1.2. Neural Network Training Loop

- **Batch Sampling:** Periodically, batches of experiences are sampled from the replay buffer.
- **Reinforcement Learning Algorithm:** Proximal Policy Optimization (PPO) will be used to train the neural network. PPO is a policy gradient method that is known for its stability and performance.
- **Loss Function:** The total loss function will be a combination of three components:
  - **Policy Loss:** A cross-entropy loss between the neural network's predicted policy and the MCTS-derived policy from the replay buffer. This encourages the network to mimic the strong moves found by MCTS.
  - **Value Loss:** A mean squared error (MSE) loss between the neural network's predicted value and the actual game outcome. This trains the network to accurately predict the game result.
  - **L2 Regularization:** Applied to the network weights to prevent overfitting.
- **Optimizer:** An Adam optimizer (or similar adaptive learning rate optimizer) will be used for gradient descent.
- **Learning Rate Schedule:** A decaying learning rate schedule will be implemented to fine-tune the training process as it progresses.

### 1.3. Checkpointing and Resumption

- **Regular Checkpoints:** The model's state dictionary and the optimizer's state will be saved at regular intervals (e.g., after a certain number of training steps or self-play games).
- **Fault Tolerance:** This allows the training process to be interrupted and resumed from the latest or a specified checkpoint, preventing loss of progress due to system failures or manual stops.

### 1.4. Logging and Monitoring

- **TensorBoard Integration:** TensorBoard will be used to log various training metrics, including:
  - Policy loss, value loss, and total loss.
  - Average game outcomes (win/loss/draw rates).
  - Learning rate.
  - Potentially, histograms of network weights and activations.
  - This provides a visual interface for monitoring training progress and debugging.

## 2. Configuration Management

- **Format:** Configuration parameters will be stored in human-readable files, preferably YAML or JSON. This allows for easy modification and versioning of training settings.
- **Configurable Parameters:** The configuration file will define:
  - **Training Hyperparameters:**
    - `learning_rate`: Initial learning rate for the optimizer.
    - `batch_size`: Number of samples per training batch.
    - `num_training_steps`: Total number of optimization steps.
    - `num_self_play_games_per_iteration`: Number of games generated in each self-play phase.
    - `mcts_simulations_per_move`: Number of MCTS simulations performed for each move during self-play.
    - `replay_buffer_capacity`: Maximum size of the replay buffer.
    - `checkpoint_frequency`: How often to save model checkpoints.
  - **Model Architecture Parameters:**
    - `num_residual_blocks`: Number of residual blocks in the neural network trunk.
    - `num_filters`: Number of filters in convolutional layers.
  - **Game Environment Parameters:**
    - `game_rules`: Any specific chess rules or variants.
    - `time_limits`: Time limits per move during self-play (if applicable).
- **Command-Line Overrides:** The training script will support overriding configuration parameters directly via command-line arguments, facilitating quick experimentation without modifying the config file.

## 3. Testing Script (`test.py`)

A separate script will be provided to evaluate the trained AlphaChess agent in different scenarios.

### 3.1. Model Loading

- The testing script will load a specified trained model checkpoint, allowing evaluation of different versions of the agent.

### 3.2. Modes of Execution

- **Self-Play Mode:**
  - The agent plays against itself, similar to the training self-play phase, but without collecting data or performing training updates.
  - This mode is useful for evaluating the agent's current strength, observing its playstyle, and generating game statistics.
  - The number of games to play in this mode will be configurable.
- **Human-Play Mode (CLI):**
  - A minimal command-line interface will allow a human player to interact with the AlphaChess agent.
  - The human inputs moves using standard algebraic notation (e.g., `e2e4`, `Nf3`).
  - The agent uses its loaded neural network and MCTS to compute its optimal move.
  - The current board state will be printed to the console after each move, providing a visual representation of the game.
  - The script will report the game outcome (win/loss/draw) and potentially the final game record in PGN format.

## 4. Resource Management & Optimization (Recap)

- **GPU Acceleration:** All PyTorch computations will be configured to run on the GPU (`cuda`) to leverage the Nvidia 3070ti.
- **Mixed-Precision Training (FP16):** `torch.cuda.amp` will be used to reduce memory consumption and speed up training.
- **Batch Sizes:** Batch sizes for both self-play and training will be carefully tuned to fit within the 8GB VRAM.
- **MCTS Memory Management:** Techniques like node pruning or explicit garbage collection will be employed to keep MCTS tree memory usage within the 16GB RAM limit.
- **`torch.compile`:** This PyTorch 2.0+ feature will be explored to further optimize the execution speed of the neural network.
