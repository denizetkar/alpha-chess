# AlphaChess Current Implementation Status

This document provides an overview of the current state of the AlphaChess project's implementation, reflecting the features and algorithms that have been integrated and are functional as of the latest development. It serves to clarify the "current app state" in contrast to the "desired app state" outlined in other design documents (e.g., `architecture.md`, `mcts.md`, `neural_network.md`, `training.md`).

## 1. Core Components Implemented

The following core components, as described in the design documentation, have been implemented:

- **Chess Environment (`src/chess_env.py`)**:

  - Handles board state management, legal move generation, and game termination conditions using the `python-chess` library.
  - **Input Plane Generation**: Converts the board state into a 29-plane input for the neural network. This includes:
    - 12 planes for piece positions (White/Black, P, N, B, R, Q, K).
    - 2 planes for player to move.
    - 4 planes for castling rights.
    - 1 plane for en passant target square.
    - 1 plane for fifty-move rule counter (normalized).
    - 1 plane for fullmove number (normalized).
    - **8 planes for Move History**: The last 8 half-moves are now included as input planes, indicating the player to move for each historical state. This captures temporal information and helps handle situations like threefold repetition. Note: This is a simplified representation compared to AlphaZero's full historical board state planes.

- **Move Encoder/Decoder (`src/move_encoder.py`)**:

  - Implements the 73-plane encoding scheme, mapping chess moves to a fixed-size integer index (0-4671) and vice-versa. This is consistent with the neural network's policy head output.

- **Neural Network Model (`src/nn_model.py`)**:

  - Implemented using PyTorch with a dual-head architecture (policy and value heads).
  - **Input Layer**: Configured to accept 29 input planes, aligning with the `ChessEnv`'s output.
  - **Architecture**: Features an initial convolutional block and configurable residual blocks, followed by a policy head (outputting 4672 move probabilities) and a value head (outputting a single scalar game outcome).
  - **Hardware Optimizations**: The model is designed with VRAM management in mind and supports mixed-precision training (`torch.cuda.amp`) and `torch.compile` for performance.

- **Monte Carlo Tree Search (MCTS) (`src/mcts.py`)**:
  - Implements the four phases: Selection (using UCB), Expansion (using NN for priors), Simulation (NN value prediction), and Backpropagation.
  - **Exploration**: Dirichlet noise is added to root node priors during self-play for enhanced exploration.
  - **Memory Optimization**: A `max_depth` parameter has been introduced to limit the MCTS tree's depth during simulations, providing a basic form of pruning to manage memory usage.

## 2. Training Script (`train.py`)

The training script is designed for robustness and efficiency:

- **Self-Play Loop**: Generates games using the current agent, collecting (state, MCTS policy, game outcome) tuples.
- **Replay Buffer**: Stores collected game data in a `deque` for batch sampling.
- **Neural Network Training Loop**:
  - Trains the network using policy (cross-entropy) and value (MSE) losses.
  - Uses the Adam optimizer.
  - **L2 Regularization**: Applied via the `weight_decay` parameter in the Adam optimizer, consistent with the `c||theta||^2` term in AlphaZero's loss function.
  - Supports configurable learning rate schedules (cosine annealing, exponential).
- **Fault Tolerance**: Implements regular checkpointing of model and optimizer states, allowing training to be interrupted and resumed.
- **Logging and Monitoring**: Integrates with TensorBoard for tracking various training metrics.
- **Configuration Management**: Parameters are loaded from `config.yaml` and can be overridden via command-line arguments.
- **Hardware Optimizations**: Utilizes GPU acceleration, mixed-precision training (`torch.cuda.amp`), and `torch.compile` for performance.

## 3. Testing Script (`test.py`)

The testing script provides evaluation capabilities:

- **Model Loading**: Can load a specified trained model checkpoint.
- **Self-Play Mode**: Allows the agent to play against itself for evaluation and statistics gathering.
- **Human-Play Mode (CLI)**: Provides a basic command-line interface for human players to interact with the AlphaChess agent.
- **Hardware Optimizations**: Supports `torch.compile` for optimized execution during testing.

## 4. Unit Tests

All existing unit tests (`tests/test_chess_env.py`, `tests/test_mcts.py`, `tests/test_move_encoder.py`, `tests/test_nn_model.py`) have been thoroughly reviewed and updated. All tests are now passing, ensuring the correctness and reliability of the implemented features.
